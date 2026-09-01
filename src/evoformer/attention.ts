import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";
import { subgroupRange } from "../runtime/subgroups.js";
import { createTiledGemmShader, gemmGrid } from "../runtime/gemm.js";

export interface AttentionWeights {
  readonly queryNormScale: Float32Array;
  readonly queryNormOffset: Float32Array;
  readonly queryWeight: Float32Array;
  readonly keyWeight: Float32Array;
  readonly valueWeight: Float32Array;
  readonly gatingWeight: Float32Array;
  readonly gatingBias: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
}

export interface SeparatePairBias {
  readonly source: "separate";
  readonly activations: Float32Array;
  readonly channels: number;
  readonly layerNormScale: Float32Array;
  readonly layerNormOffset: Float32Array;
  readonly projectionWeight: Float32Array;
}

export interface NormalizedInputPairBias {
  readonly source: "normalized-input";
  readonly projectionWeight: Float32Array;
}

export type AttentionPairBias = SeparatePairBias | NormalizedInputPairBias;

export interface AttentionInput {
  readonly activations: Float32Array;
  readonly mask: Float32Array;
  readonly batch: number;
  readonly queryLength: number;
  readonly channels: number;
  readonly heads: number;
  readonly transpose?: boolean;
  readonly weights: AttentionWeights;
  readonly pairBias?: AttentionPairBias;
  readonly epsilon?: number;
}

export interface AttentionResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

export type AttentionFlashVariant = "auto" | "portable" | "register" | "register-2q" | "register-4q"
  | "subgroup-4x8"
  | "subgroup-key32"
  | "subgroup-8x16" | "subgroup-8x32" | "subgroup-8x64"
  | "subgroup-16x64" | "subgroup-32x64" | "subgroup-64x64";

export interface AttentionGpuOptions {
  /** Primarily useful for differential benchmarks; production callers should use auto. */
  readonly flashVariant?: AttentionFlashVariant;
}

export interface AttentionFlashKernel {
  readonly cacheKey: string;
  readonly shader: string;
  readonly queryTile: number;
  readonly variant: Exclude<AttentionFlashVariant, "auto">;
}

const GRID_WIDTH = 32_768;
const ceilDivide = (value: number, divisor: number): number => Math.ceil(value / divisor);

function validate(input: AttentionInput): void {
  const { batch, queryLength, channels, heads, activations, mask, weights } = input;
  if (![batch, queryLength, channels, heads].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("attention dimensions must be positive safe integers");
  }
  if (channels % heads !== 0 || channels / heads > 32) {
    throw new RangeError("the fused WebGPU attention path requires an integral head dimension no larger than 32");
  }
  const projected = channels;
  const expected: ReadonlyArray<readonly [string, Float32Array, number]> = [
    ["activations", activations, batch * queryLength * channels],
    ["mask", mask, batch * queryLength],
    ["queryNormScale", weights.queryNormScale, channels],
    ["queryNormOffset", weights.queryNormOffset, channels],
    ["queryWeight", weights.queryWeight, channels * projected],
    ["keyWeight", weights.keyWeight, channels * projected],
    ["valueWeight", weights.valueWeight, channels * projected],
    ["gatingWeight", weights.gatingWeight, channels * projected],
    ["gatingBias", weights.gatingBias, projected],
    ["outputWeight", weights.outputWeight, projected * channels],
    ["outputBias", weights.outputBias, channels],
  ];
  for (const [name, value, size] of expected) {
    if (value.length !== size) throw new RangeError(`${name} has ${value.length} values; expected ${size}`);
  }
  const pair = input.pairBias;
  if (pair?.source === "separate") {
    const pairExpected: ReadonlyArray<readonly [string, Float32Array, number]> = [
      ["pair activations", pair.activations, queryLength * queryLength * pair.channels],
      ["pair norm scale", pair.layerNormScale, pair.channels],
      ["pair norm offset", pair.layerNormOffset, pair.channels],
      ["pair projection", pair.projectionWeight, pair.channels * heads],
    ];
    for (const [name, value, size] of pairExpected) {
      if (value.length !== size) throw new RangeError(`${name} has ${value.length} values; expected ${size}`);
    }
  } else if (pair !== undefined && pair.projectionWeight.length !== channels * heads) {
    throw new RangeError("normalized-input pair projection has an invalid size");
  }
}

export interface PackedAttentionWeights { readonly data: Float32Array; readonly offsets: readonly number[]; }

/**
 * Scratch budget for one attention batch window.
 *
 * An attention operation holds five tensors of [batch * queries, channels]:
 * the normalized input, query, key, value and gate. Attention is independent
 * across batch entries, so covering the batch in windows bounds all five at
 * once. Windowing costs dispatches, not efficiency: even the narrowest window
 * this budget produces leaves the projection hundreds of GEMM row tiles tall.
 */
export const ATTENTION_WINDOW_TARGET_BYTES = 32 * 1024 * 1024;

export function attentionBatchWindow(
  batch: number, queries: number, channels: number,
  budgetBytes: number = ATTENTION_WINDOW_TARGET_BYTES,
): number {
  if (![batch, queries, channels, budgetBytes].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("attention window dimensions and budget must be positive safe integers");
  }
  const bytesPerBatchEntry = queries * channels * Float32Array.BYTES_PER_ELEMENT;
  return Math.max(1, Math.min(batch, Math.floor(budgetBytes / bytesPerBatchEntry)));
}

export function packAttentionWeights(input: AttentionInput): PackedAttentionWeights {
  const w = input.weights;
  const tensors: Float32Array[] = [
    w.queryNormScale, w.queryNormOffset, w.queryWeight, w.keyWeight, w.valueWeight,
    w.gatingWeight, w.gatingBias, w.outputWeight, w.outputBias,
  ];
  const pair = input.pairBias;
  if (pair?.source === "separate") {
    tensors.push(pair.layerNormScale, pair.layerNormOffset, pair.projectionWeight);
  } else if (pair !== undefined) {
    tensors.push(pair.projectionWeight);
  }
  const offsets: number[] = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const data = new Float32Array(size);
  tensors.forEach((tensor, index) => data.set(tensor, offsets[index]));
  return { data, offsets };
}

/**
 * @param batchWindow Batch entries covered by this dispatch, and where the
 * window starts. Defaults to the whole batch.
 */
export function createAttentionParameters(
  input: AttentionInput, offsets: readonly number[],
  batchWindow: { readonly offset: number; readonly count: number }
    = { offset: 0, count: input.batch },
): Uint32Array {
  const pairProjectionIndex = input.pairBias?.source === "separate" ? 11 : 9;
  return new Uint32Array([
    batchWindow.count, input.queryLength, input.channels, input.heads, input.channels / input.heads,
    input.transpose === true ? 1 : 0, input.pairBias === undefined ? 0 : 1,
    offsets[2]!, offsets[3]!, offsets[4]!, offsets[5]!, offsets[6]!, offsets[7]!, offsets[8]!,
    input.pairBias === undefined ? 0 : offsets[pairProjectionIndex]!,
    input.pairBias?.source === "separate" ? input.pairBias.channels : input.channels,
    batchWindow.offset, input.batch, 0, 0,
  ]);
}

/**
 * @param batchOffset First batch entry this dispatch covers, and `batchTotal`
 * the number in the whole operation. They differ from `batch` only when the
 * operation is split into windows; `batch` always counts this window.
 */
export function createAttentionNormParameters(
  rows: number, channels: number, scale: number, offset: number,
  transpose: boolean, batch: number, queries: number, epsilon: number,
  batchOffset = 0, batchTotal = batch,
): Uint8Array {
  const buffer = new ArrayBuffer(48);
  const view = new DataView(buffer);
  [rows, channels, scale, offset, transpose ? 1 : 0, batch, queries].forEach(
    (value, index) => view.setUint32(index * 4, value, true),
  );
  view.setFloat32(28, epsilon, true);
  view.setUint32(32, batchOffset, true);
  view.setUint32(36, batchTotal, true);
  return new Uint8Array(buffer);
}

export const ATTENTION_NORMALIZE_SHADER = `
struct NormParameters {
  rows: u32, channels: u32, scale: u32, offset: u32,
  transpose: u32, batch: u32, queries: u32, epsilon: f32,
  batch_offset: u32, batch_total: u32, padding: vec2<u32>,
};
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: NormParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

// Rows are numbered within this batch window; the source holds the whole batch.
fn source_row(row: u32) -> u32 {
  let b = p.batch_offset + row / p.queries;
  let q = row % p.queries;
  if (p.transpose == 0u) { return b * p.queries + q; }
  return q * p.batch_total + b;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= p.rows) { return; }
  let input_base = source_row(row) * p.channels;
  let output_base = row * p.channels;
  var sum = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) { sum += source[input_base + c]; }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(p.channels); }
  workgroupBarrier();
  var squared = 0.0;
  for (var c = local.x; c < p.channels; c += 64u) {
    let centered = source[input_base + c] - row_mean[0];
    squared += centered * centered;
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(p.channels) + p.epsilon);
  for (var c = local.x; c < p.channels; c += 64u) {
    output[output_base + c] = (source[input_base + c] - row_mean[0]) * inverse_std
      * weights[p.scale + c] + weights[p.offset + c];
  }
}`;

const COMMON = `
struct Parameters {
  // batch counts the entries in this window; batch_offset and batch_total
  // locate it inside the whole operation, which transposed layouts need.
  batch: u32, queries: u32, channels: u32, heads: u32,
  head_dim: u32, transpose: u32, has_pair_bias: u32,
  query_weight: u32, key_weight: u32, value_weight: u32,
  gating_weight: u32, gating_bias: u32, output_weight: u32,
  output_bias: u32, pair_weight: u32, pair_channels: u32,
  batch_offset: u32, batch_total: u32, padding: vec2<u32>,
};
const GRID_WIDTH: u32 = 32768u;
`;

/**
 * Query, key, value, and gate in one projection.
 *
 * The four matrices share the source rows, so they are contracted as a single
 * A x W of width 4 * projected and split apart in the epilogue: the column
 * index selects both which weight block to read and which output tensor to
 * write. Each invocation's four columns always fall inside one matrix because
 * every AlphaFold projection width is a multiple of four.
 */
/**
 * Mask lookup for one batch entry of the current window.
 *
 * Shared by every flash variant so the window arithmetic has one definition.
 * The transposed layout strides by the whole batch, not by the window.
 */
const MASK_INDEX = `
fn mask_index(batch: u32, key_index: u32) -> u32 {
  let b = p.batch_offset + batch;
  if (p.transpose == 0u) { return b * p.queries + key_index; }
  return key_index * p.batch_total + b;
}`;

export const ATTENTION_PROJECT_SHADER = createTiledGemmShader({
  preamble: `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> query: array<f32>;
@group(0) @binding(4) var<storage, read_write> key: array<f32>;
@group(0) @binding(5) var<storage, read_write> value: array<f32>;
@group(0) @binding(6) var<storage, read_write> gate: array<f32>;

fn projection_weight_offset(matrix: u32) -> u32 {
  if (matrix == 0u) { return p.query_weight; }
  if (matrix == 1u) { return p.key_weight; }
  if (matrix == 2u) { return p.value_weight; }
  return p.gating_weight;
}`,
  rows: "p.batch * p.queries",
  inner: "p.channels",
  columns: "4u * p.heads * p.head_dim",
  sourceElement: "source[row * p.channels + k]",
  weightElement: `weights[projection_weight_offset(column / (p.heads * p.head_dim))
        + k * p.heads * p.head_dim + column % (p.heads * p.head_dim)]`,
  store: `let projected = p.heads * p.head_dim;
          let matrix = column / projected;
          let index = row * projected + column % projected;
          if (matrix == 0u) { query[index] = element * inverseSqrt(f32(p.head_dim)); }
          else if (matrix == 1u) { key[index] = element; }
          else if (matrix == 2u) { value[index] = element; }
          else {
            let biased = element + weights[p.gating_bias + column % projected];
            gate[index] = 1.0 / (1.0 + exp(-biased));
          }`,
});

export const ATTENTION_PAIR_BIAS_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> pair: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.heads * p.queries * p.queries) { return; }
  let k = index % p.queries;
  let q = (index / p.queries) % p.queries;
  let head = index / (p.queries * p.queries);
  var result = 0.0;
  for (var c = 0u; c < p.pair_channels; c += 1u) {
    result += pair[(q * p.queries + k) * p.pair_channels + c]
      * weights[p.pair_weight + c * p.heads + head];
  }
  output[index] = result;
}`;

export const ATTENTION_FLASH_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 32>;
var<workgroup> state: array<f32, 3>;

${MASK_INDEX}

@compute @workgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let q_index = group.x;
  let batch_index = group.y;
  let head = group.z;
  let lane = local.x;
  if (q_index >= p.queries || batch_index >= p.batch || head >= p.heads) { return; }
  let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
  var accumulated = 0.0;
  var running_max = -1e30;
  var running_sum = 0.0;
  for (var k_index = 0u; k_index < p.queries; k_index += 1u) {
    let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * p.head_dim;
    partial[lane] = select(0.0, query[q_base + lane] * key[k_base + lane], lane < p.head_dim);
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride /= 2u) {
      if (lane < stride) { partial[lane] += partial[lane + stride]; }
      workgroupBarrier();
    }
    if (lane == 0u) {
      var logit = partial[0] + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
      if (p.has_pair_bias != 0u) {
        logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
      }
      logit = clamp(logit, -1e8, 1e8);
      let new_max = max(running_max, logit);
      state[0] = exp(running_max - new_max);
      state[1] = exp(logit - new_max);
      running_sum = running_sum * state[0] + state[1];
      running_max = new_max;
      state[2] = running_sum;
    }
    workgroupBarrier();
    if (lane < p.head_dim) {
      accumulated = accumulated * state[0] + state[1] * value[k_base + lane];
    }
    workgroupBarrier();
  }
  if (lane < p.head_dim) {
    output[q_base + lane] = (accumulated / state[2]) * gate[q_base + lane];
  }
}`;

/**
 * Portable flash attention with one complete head owned by each invocation.
 *
 * AlphaFold's attention head dimensions are 32 and 8, so the query and online
 * softmax accumulator fit in eight or two named vec4 registers. This removes
 * the workgroup barriers and tree reductions from the key loop without relying
 * on subgroup extensions, which Chrome-on-Metal does not currently expose.
 */
export function createAttentionRegisterFlashShader(headDim: number, queriesPerThread = 1): string {
  if (!Number.isSafeInteger(headDim) || headDim <= 0 || headDim > 32 || headDim % 4 !== 0) {
    throw new RangeError("register attention requires a positive head dimension divisible by four and at most 32");
  }
  if (![1, 2, 4].includes(queriesPerThread)) {
    throw new RangeError("register attention supports one, two, or four queries per invocation");
  }
  const vectors = headDim / 4;
  const slots = queriesPerThread;
  const perSlot = (slot: number, indent: string, body: (index: number) => string): string => Array.from(
    { length: vectors }, (_, index) => `${indent}${body(index)}`,
  ).join("\n");
  const eachSlot = (indent: string, body: (slot: number) => string): string => Array.from(
    { length: slots }, (_, slot) => body(slot).split("\n").map((line) => `${indent}${line}`).join("\n"),
  ).join("\n");
  return `${COMMON}
const HD4: u32 = ${vectors}u;
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

${MASK_INDEX}

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let batch_index = group.y;
  let head = group.z;
  if (batch_index >= p.batch || head >= p.heads) { return; }
  // Slots stride by the workgroup width so neighbouring invocations keep
  // reading neighbouring queries, and one key load serves ${slots} of them.
  let query_origin = group.x * ${64 * slots}u + local.x;
${eachSlot("  ", (slot) => `let q_index_${slot} = query_origin + ${slot * 64}u;
let live_${slot} = q_index_${slot} < p.queries;
let q_base_${slot} = ((batch_index * p.queries + select(0u, q_index_${slot}, live_${slot})) * p.heads + head) * HD4;
${perSlot(slot, "", (index) => `var qv_${slot}_${index} = query[q_base_${slot} + ${index}u];`)}
${perSlot(slot, "", (index) => `var acc_${slot}_${index} = vec4<f32>(0.0);`)}
var running_max_${slot} = -1e30;
var running_sum_${slot} = 0.0;`)}

  for (var k_index = 0u; k_index < p.queries; k_index += 1u) {
    let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * HD4;
${perSlot(0, "    ", (index) => `let kv${index} = key[k_base + ${index}u];`)}
${perSlot(0, "    ", (index) => `let vv${index} = value[k_base + ${index}u];`)}
    let masked = 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
${eachSlot("    ", (slot) => `{
  var score = 0.0;
${perSlot(slot, "  ", (index) => `score += dot(qv_${slot}_${index}, kv${index});`)}
  var logit = score + masked;
  if (p.has_pair_bias != 0u) {
    logit += pair_bias[(head * p.queries + select(0u, q_index_${slot}, live_${slot})) * p.queries + k_index];
  }
  logit = clamp(logit, -1e8, 1e8);
  let new_max = max(running_max_${slot}, logit);
  let previous_scale = exp(running_max_${slot} - new_max);
  let weight = exp(logit - new_max);
  running_sum_${slot} = running_sum_${slot} * previous_scale + weight;
  running_max_${slot} = new_max;
${perSlot(slot, "  ", (index) => `acc_${slot}_${index} = acc_${slot}_${index} * previous_scale + weight * vv${index};`)}
}`)}
  }

${eachSlot("  ", (slot) => `if (live_${slot}) {
${perSlot(slot, "  ", (index) => `output[q_base_${slot} + ${index}u] = (acc_${slot}_${index} / running_sum_${slot}) * gate[q_base_${slot} + ${index}u];`)}
}`)}
}`;
}

// Fast path for devices that guarantee one 32-lane subgroup. All lanes keep
// identical online-softmax state, eliminating workgroup barriers in the key loop.
export const ATTENTION_SUBGROUP_FLASH_SHADER = `enable subgroups;
enable subgroup_size_control;
${COMMON}
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> key_tile: array<vec4<f32>, 64>;
var<workgroup> value_tile: array<vec4<f32>, 64>;

${MASK_INDEX}

@compute @workgroup_size(32, 4, 1) @subgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let q_index = group.x * 4u + local.y;
  let batch_index = group.y;
  let head = group.z;
  let lane = local.x;
  let valid_query = q_index < p.queries;
  let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
  var accumulated = 0.0;
  var running_max = -1e30;
  var running_sum = 0.0;
  let linear_lane = local.y * 32u + local.x;
  for (var k0 = 0u; k0 < p.queries; k0 += 8u) {
    if (linear_lane < 64u) {
      let tile_key_index = linear_lane / 8u;
      let vector_index = linear_lane % 8u;
      let k_index = k0 + tile_key_index;
      key_tile[linear_lane] = vec4<f32>(0.0);
      if (k_index < p.queries) {
        let k_base = (((batch_index * p.queries + k_index) * p.heads + head) * 8u);
        key_tile[linear_lane] = key[k_base + vector_index];
      }
    } else {
      let item = linear_lane - 64u;
      let tile_key_index = item / 8u;
      let vector_index = item % 8u;
      let k_index = k0 + tile_key_index;
      value_tile[item] = vec4<f32>(0.0);
      if (k_index < p.queries) {
        let k_base = (((batch_index * p.queries + k_index) * p.heads + head) * 8u);
        value_tile[item] = value[k_base + vector_index];
      }
    }
    workgroupBarrier();
    for (var tile_key_index = 0u; tile_key_index < 8u; tile_key_index += 1u) {
      let k_index = k0 + tile_key_index;
      if (k_index < p.queries) {
        let vector_index = lane / 4u;
        let component = lane % 4u;
        var product = 0.0;
        if (valid_query) {
          product = query[q_base / 4u + vector_index][component]
            * key_tile[tile_key_index * 8u + vector_index][component];
        }
        var logit = subgroupAdd(product) + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
        if (valid_query && p.has_pair_bias != 0u) {
          logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
        }
        logit = clamp(logit, -1e8, 1e8);
        let new_max = max(running_max, logit);
        let previous_scale = exp(running_max - new_max);
        let weight = exp(logit - new_max);
        running_sum = running_sum * previous_scale + weight;
        if (valid_query) {
          accumulated = accumulated * previous_scale
            + weight * value_tile[tile_key_index * 8u + vector_index][component];
        }
        running_max = new_max;
      }
    }
    workgroupBarrier();
  }
  if (valid_query && lane < p.head_dim) {
    output[q_base + lane] = (accumulated / running_sum) * gate[q_base + lane];
  }
}`;

export function supportsAttentionSubgroups(device: GPUDevice, headDim = 32): boolean {
  // subgroup-size-control only permits selecting a width inside the device's
  // advertised range. SwiftShader, for example, exposes the feature while
  // fixing the range to [4, 4], so feature detection alone is insufficient.
  const range = subgroupRange(device);
  return headDim === 32
    && device.features.has("subgroups")
    && device.features.has("subgroup-size-control")
    && range !== undefined
    && range[0] <= 32
    && range[1] >= 32;
}

/**
 * Software analogues of the Pallas tiled flash-attention algorithm.
 *
 * Eight 32-lane subgroups cooperate on up to 64 query rows and 64 key rows.
 * One lane owns one head channel, and K/V are loaded once into workgroup
 * memory. Scores remain ephemeral: online softmax immediately folds each score
 * into the value accumulator. The 64x64 specialization is the closest Pallas
 * analogue; smaller query tiles trade data reuse for more parallelism.
 */
function createAttentionSubgroupTiledShader(queryTile: 8 | 16 | 32 | 64, keyTile: 16 | 32 | 64): string {
  const querySlots = queryTile / 8;
  return `enable subgroups;
enable subgroup_size_control;
${COMMON}
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> key_tile: array<vec4<f32>, ${keyTile * 8}>;
var<workgroup> value_tile: array<vec4<f32>, ${keyTile * 8}>;

${MASK_INDEX}

@compute @workgroup_size(32, 8, 1) @subgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
) {
  let batch_index = group.y;
  let head = group.z;
  let vector_index = subgroup_lane / 4u;
  let component = subgroup_lane % 4u;
  let linear_lane = local.y * 32u + local.x;
  var accumulated: array<f32, ${querySlots}>;
  var running_max: array<f32, ${querySlots}>;
  var running_sum: array<f32, ${querySlots}>;
  var query_component: array<f32, ${querySlots}>;

  for (var query_slot = 0u; query_slot < ${querySlots}u; query_slot += 1u) {
    let q_index = group.x * ${queryTile}u + local.y + query_slot * 8u;
    accumulated[query_slot] = 0.0;
    running_max[query_slot] = -1e30;
    running_sum[query_slot] = 0.0;
    query_component[query_slot] = 0.0;
    if (q_index < p.queries) {
      let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * 8u;
      query_component[query_slot] = query[q_base + vector_index][component];
    }
  }

  for (var k0 = 0u; k0 < p.queries; k0 += ${keyTile}u) {
    for (var tile_vector = linear_lane; tile_vector < ${keyTile * 8}u; tile_vector += 256u) {
      let tile_key_index = tile_vector / 8u;
      let tile_vector_index = tile_vector % 8u;
      let k_index = k0 + tile_key_index;
      key_tile[tile_vector] = vec4<f32>(0.0);
      value_tile[tile_vector] = vec4<f32>(0.0);
      if (k_index < p.queries) {
        let k_base = ((batch_index * p.queries + k_index) * p.heads + head) * 8u;
        key_tile[tile_vector] = key[k_base + tile_vector_index];
        value_tile[tile_vector] = value[k_base + tile_vector_index];
      }
    }
    workgroupBarrier();

    for (var query_slot = 0u; query_slot < ${querySlots}u; query_slot += 1u) {
      let q_index = group.x * ${queryTile}u + local.y + query_slot * 8u;
      let valid_query = q_index < p.queries;
      for (var tile_key_index = 0u; tile_key_index < ${keyTile}u; tile_key_index += 1u) {
        let k_index = k0 + tile_key_index;
        if (k_index < p.queries) {
          let product = query_component[query_slot]
            * key_tile[tile_key_index * 8u + vector_index][component];
          var logit = subgroupAdd(select(0.0, product, valid_query))
            + 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
          if (valid_query && p.has_pair_bias != 0u) {
            logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
          }
          logit = clamp(logit, -1e8, 1e8);
          let new_max = max(running_max[query_slot], logit);
          let previous_scale = exp(running_max[query_slot] - new_max);
          let weight = exp(logit - new_max);
          running_sum[query_slot] = running_sum[query_slot] * previous_scale + weight;
          running_max[query_slot] = new_max;
          if (valid_query) {
            accumulated[query_slot] = accumulated[query_slot] * previous_scale
              + weight * value_tile[tile_key_index * 8u + vector_index][component];
          }
        }
      }
    }
    workgroupBarrier();
  }

  for (var query_slot = 0u; query_slot < ${querySlots}u; query_slot += 1u) {
    let q_index = group.x * ${queryTile}u + local.y + query_slot * 8u;
    if (q_index < p.queries) {
      let q_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
      output[q_base + subgroup_lane] = (accumulated[query_slot] / running_sum[query_slot])
        * gate[q_base + subgroup_lane];
    }
  }
}`;
}

export const ATTENTION_SUBGROUP_8X16_SHADER = createAttentionSubgroupTiledShader(8, 16);
export const ATTENTION_SUBGROUP_8X32_SHADER = createAttentionSubgroupTiledShader(8, 32);
export const ATTENTION_SUBGROUP_8X64_SHADER = createAttentionSubgroupTiledShader(8, 64);
export const ATTENTION_SUBGROUP_16X64_SHADER = createAttentionSubgroupTiledShader(16, 64);
export const ATTENTION_SUBGROUP_32X64_SHADER = createAttentionSubgroupTiledShader(32, 64);
export const ATTENTION_SUBGROUP_64X64_SHADER = createAttentionSubgroupTiledShader(64, 64);

/**
 * Key-parallel flash attention for a 32-channel head.
 *
 * Every subgroup handles one query. Its lanes calculate 32 key scores in
 * parallel, reduce the tile softmax, and then shuffle those probabilities
 * across lanes while each lane accumulates one value/output channel.
 */
export const ATTENTION_SUBGROUP_KEY32_SHADER = `enable subgroups;
enable subgroup_size_control;
${COMMON}
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gate: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<f32>;
var<workgroup> key_tile: array<vec4<f32>, 256>;
var<workgroup> value_tile: array<vec4<f32>, 256>;

${MASK_INDEX}

@compute @workgroup_size(32, 8, 1) @subgroup_size(32)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_invocation_id) subgroup_lane: u32,
) {
  let q_index = group.x * 8u + local.y;
  let batch_index = group.y;
  let head = group.z;
  let valid_query = q_index < p.queries;
  let query_base = ((batch_index * p.queries + q_index) * p.heads + head) * 8u;
  let output_vector = subgroup_lane / 4u;
  let output_component = subgroup_lane % 4u;
  let linear_lane = local.y * 32u + local.x;
  var accumulated = 0.0;
  var running_max = -1e30;
  var running_sum = 0.0;

  for (var k0 = 0u; k0 < p.queries; k0 += 32u) {
    let load_key = linear_lane / 8u;
    let load_vector = linear_lane % 8u;
    let source_key = k0 + load_key;
    let transposed_index = load_vector * 32u + load_key;
    key_tile[transposed_index] = vec4<f32>(0.0);
    value_tile[transposed_index] = vec4<f32>(0.0);
    if (source_key < p.queries) {
      let source_base = ((batch_index * p.queries + source_key) * p.heads + head) * 8u;
      key_tile[transposed_index] = key[source_base + load_vector];
      value_tile[transposed_index] = value[source_base + load_vector];
    }
    workgroupBarrier();

    let k_index = k0 + subgroup_lane;
    let valid_key = k_index < p.queries;
    var logit = -1e30;
    if (valid_query && valid_key) {
      logit = 0.0;
      for (var vector = 0u; vector < 8u; vector += 1u) {
        logit += dot(query[query_base + vector], key_tile[vector * 32u + subgroup_lane]);
      }
      logit += 1e9 * (mask[mask_index(batch_index, k_index)] - 1.0);
      if (p.has_pair_bias != 0u) {
        logit += pair_bias[(head * p.queries + q_index) * p.queries + k_index];
      }
      logit = clamp(logit, -1e8, 1e8);
    }

    let tile_max = subgroupMax(logit);
    let probability = select(0.0, exp(logit - tile_max), valid_query && valid_key);
    let tile_sum = subgroupAdd(probability);
    var tile_weighted = 0.0;
    for (var source_lane = 0u; source_lane < 32u; source_lane += 1u) {
      let source_probability = subgroupShuffle(probability, source_lane);
      tile_weighted += source_probability
        * value_tile[output_vector * 32u + source_lane][output_component];
    }

    let new_max = max(running_max, tile_max);
    let previous_scale = exp(running_max - new_max);
    let tile_scale = exp(tile_max - new_max);
    accumulated = accumulated * previous_scale + tile_weighted * tile_scale;
    running_sum = running_sum * previous_scale + tile_sum * tile_scale;
    running_max = new_max;
    workgroupBarrier();
  }

  if (valid_query) {
    let output_base = ((batch_index * p.queries + q_index) * p.heads + head) * p.head_dim;
    output[output_base + subgroup_lane] = (accumulated / running_sum)
      * gate[output_base + subgroup_lane];
  }
}`;

export function supportsAttentionSubgroup64x64(device: GPUDevice, headDim = 32): boolean {
  return supportsAttentionSubgroups(device, headDim)
    && device.limits.maxComputeInvocationsPerWorkgroup >= 256
    && device.limits.maxComputeWorkgroupStorageSize >= 16_384;
}

export function selectAttentionFlashKernel(
  device: GPUDevice,
  headDim = 32,
  requested: AttentionFlashVariant = "auto",
): AttentionFlashKernel {
  const subgroup = supportsAttentionSubgroups(device, headDim);
  const subgroup64 = supportsAttentionSubgroup64x64(device, headDim);
  const variant = requested === "auto"
    ? (subgroup64 ? "subgroup-key32" : subgroup ? "subgroup-4x8" : "register")
    : requested;
  if (variant.startsWith("subgroup-") && variant !== "subgroup-4x8" && !subgroup64) {
    throw new Error(`the ${variant} attention kernel is unsupported by this device`);
  }
  if (variant === "subgroup-4x8" && !subgroup) {
    throw new Error("the subgroup-4x8 attention kernel is unsupported by this device");
  }
  if (variant === "subgroup-key32") {
    return {
      cacheKey: "attention:flash-subgroup-key32", shader: ATTENTION_SUBGROUP_KEY32_SHADER,
      queryTile: 8, variant,
    };
  }
  const tiled = variant === "subgroup-8x16"
    ? { queryTile: 8, shader: ATTENTION_SUBGROUP_8X16_SHADER }
    : variant === "subgroup-8x32"
      ? { queryTile: 8, shader: ATTENTION_SUBGROUP_8X32_SHADER }
      : variant === "subgroup-8x64"
        ? { queryTile: 8, shader: ATTENTION_SUBGROUP_8X64_SHADER }
        : variant === "subgroup-16x64"
          ? { queryTile: 16, shader: ATTENTION_SUBGROUP_16X64_SHADER }
          : variant === "subgroup-32x64"
            ? { queryTile: 32, shader: ATTENTION_SUBGROUP_32X64_SHADER }
            : variant === "subgroup-64x64"
              ? { queryTile: 64, shader: ATTENTION_SUBGROUP_64X64_SHADER }
              : undefined;
  if (tiled !== undefined) {
    return {
      cacheKey: `attention:flash-${variant}`, shader: tiled.shader,
      queryTile: tiled.queryTile, variant,
    };
  }
  if (variant === "subgroup-4x8") {
    return {
      cacheKey: "attention:flash-subgroup4x8", shader: ATTENTION_SUBGROUP_FLASH_SHADER,
      queryTile: 4, variant,
    };
  }
  if (variant === "register" || variant === "register-2q" || variant === "register-4q") {
    if (headDim % 4 !== 0) throw new Error("register attention requires a head dimension divisible by four");
    const queriesPerThread = variant === "register-4q" ? 4 : variant === "register-2q" ? 2 : 1;
    return {
      cacheKey: `attention:flash-registers-${headDim}-q${queriesPerThread}`,
      shader: createAttentionRegisterFlashShader(headDim, queriesPerThread),
      queryTile: 64 * queriesPerThread,
      variant,
    };
  }
  return { cacheKey: "attention:flash", shader: ATTENTION_FLASH_SHADER, queryTile: 1, variant };
}

function createAttentionOutputShader(residual: boolean): string {
  return createTiledGemmShader({
    preamble: `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;`,
    rows: "p.batch * p.queries",
    inner: "p.heads * p.head_dim",
    columns: "p.channels",
    sourceElement: "source[row * p.heads * p.head_dim + k]",
    weightElement: "weights[p.output_weight + k * p.channels + column]",
    // Rows are numbered within this batch window, and column attention consumes
    // a transposed view, so the result row is remapped both ways.
    store: `let b = p.batch_offset + row / p.queries;
          let q = row % p.queries;
          let output_row = select(b * p.queries + q, q * p.batch_total + b, p.transpose != 0u);
          output[output_row * p.channels + column] ${residual ? "+=" : "="}
            element + weights[p.output_bias + column];`,
  });
}

export const ATTENTION_OUTPUT_SHADER = createAttentionOutputShader(false);

/** Same projection as ATTENTION_OUTPUT_SHADER, but commits directly into an existing residual tensor. */
export const ATTENTION_OUTPUT_RESIDUAL_SHADER = createAttentionOutputShader(true);

export class AttentionGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  readonly options: AttentionGpuOptions;

  constructor(device: GPUDevice, options: AttentionGpuOptions = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    this.options = options;
  }

  async run(input: AttentionInput): Promise<AttentionResult> {
    validate(input);
    const packed = packAttentionWeights(input);
    const flashKernel = selectAttentionFlashKernel(
      this.device, input.channels / input.heads, this.options.flashVariant ?? "auto",
    );
    const [normalize, project, pairProject, flash, outputProject] = await Promise.all([
      this.pipelines.get("attention:normalize", ATTENTION_NORMALIZE_SHADER),
      this.pipelines.get("attention:project", ATTENTION_PROJECT_SHADER),
      this.pipelines.get("attention:pair-bias", ATTENTION_PAIR_BIAS_SHADER),
      this.pipelines.get(flashKernel.cacheKey, flashKernel.shader),
      this.pipelines.get("attention:output", ATTENTION_OUTPUT_SHADER),
    ]);
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    const storage = GPUBufferUsage.STORAGE;
    const rows = input.batch * input.queryLength;
    const tensorBytes = rows * input.channels * 4;
    const linearGrid = (elements: number): readonly [number, number] => {
      const groups = ceilDivide(elements, 64);
      return [Math.min(groups, GRID_WIDTH), ceilDivide(groups, GRID_WIDTH)];
    };
    const rowGrid = (rowsValue: number): readonly [number, number] => [
      Math.min(rowsValue, GRID_WIDTH), ceilDivide(rowsValue, GRID_WIDTH),
    ];
    try {
      const source = keep(this.allocator.upload("attention.source", input.activations, storage));
      const mask = keep(this.allocator.upload("attention.mask", input.mask, storage));
      const weights = keep(this.allocator.upload("attention.weights", packed.data, storage));
      const params = keep(this.allocator.upload(
        "attention.parameters", createAttentionParameters(input, packed.offsets), GPUBufferUsage.UNIFORM,
      ));
      const queryNormParams = keep(this.allocator.upload("attention.query-norm-parameters", createAttentionNormParameters(
        rows, input.channels, packed.offsets[0]!, packed.offsets[1]!, input.transpose === true,
        input.batch, input.queryLength, input.epsilon ?? 1e-5,
      ), GPUBufferUsage.UNIFORM));
      const normalized = keep(this.allocator.allocate("attention.normalized", tensorBytes, storage));
      const query = keep(this.allocator.allocate("attention.query", tensorBytes, storage));
      const key = keep(this.allocator.allocate("attention.key", tensorBytes, storage));
      const value = keep(this.allocator.allocate("attention.value", tensorBytes, storage));
      const gate = keep(this.allocator.allocate("attention.gate", tensorBytes, storage));
      const weighted = keep(this.allocator.allocate("attention.weighted", tensorBytes, storage));
      const output = keep(this.allocator.allocate("attention.output", tensorBytes, storage | GPUBufferUsage.COPY_SRC));
      const readback = keep(this.allocator.allocate(
        "attention.readback", tensorBytes, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      let pairNormalized = normalized;
      let pairSource: AllocatedGpuBuffer | undefined;
      let pairNormParams: AllocatedGpuBuffer | undefined;
      if (input.pairBias?.source === "separate") {
        pairSource = keep(this.allocator.upload("attention.pair-source", input.pairBias.activations, storage));
        pairNormalized = keep(this.allocator.allocate(
          "attention.pair-normalized", input.queryLength * input.queryLength * input.pairBias.channels * 4, storage,
        ));
        pairNormParams = keep(this.allocator.upload("attention.pair-norm-parameters", createAttentionNormParameters(
          input.queryLength * input.queryLength, input.pairBias.channels,
          packed.offsets[9]!, packed.offsets[10]!, false, 1, input.queryLength * input.queryLength,
          input.epsilon ?? 1e-5,
        ), GPUBufferUsage.UNIFORM));
      }
      const pairBiasElements = input.pairBias === undefined
        ? 1 : input.heads * input.queryLength * input.queryLength;
      const pairBias = keep(this.allocator.allocate("attention.pair-bias", pairBiasElements * 4, storage));

      const encoder = this.device.createCommandEncoder({ label: "attention" });
      this.device.pushErrorScope("validation");
      const pass = (
        pipeline: GPUComputePipeline, buffers: readonly GPUBuffer[], x: number, y = 1, z = 1,
      ): void => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        compute.dispatchWorkgroups(x, y, z);
        compute.end();
      };
      let grid = rowGrid(rows);
      pass(normalize, [source.buffer, weights.buffer, queryNormParams.buffer, normalized.buffer], grid[0], grid[1]);
      if (pairSource !== undefined && pairNormParams !== undefined) {
        grid = rowGrid(input.queryLength * input.queryLength);
        pass(normalize, [pairSource.buffer, weights.buffer, pairNormParams.buffer, pairNormalized.buffer],
          grid[0], grid[1]);
      }
      if (input.pairBias !== undefined) {
        const pairGrid = linearGrid(input.heads * input.queryLength * input.queryLength);
        pass(pairProject, [pairNormalized.buffer, weights.buffer, params.buffer, pairBias.buffer],
          pairGrid[0], pairGrid[1]);
      }
      const projectGrid = gemmGrid(rows, 4 * input.channels);
      pass(project, [normalized.buffer, weights.buffer, params.buffer, query.buffer, key.buffer, value.buffer, gate.buffer],
        projectGrid[0], projectGrid[1]);
      pass(flash, [query.buffer, key.buffer, value.buffer, gate.buffer, mask.buffer, pairBias.buffer, params.buffer,
        weighted.buffer], ceilDivide(input.queryLength, flashKernel.queryTile),
        input.batch, input.heads);
      const outputGrid = gemmGrid(rows, input.channels);
      pass(outputProject, [weighted.buffer, weights.buffer, params.buffer, output.buffer],
        outputGrid[0], outputGrid[1]);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, tensorBytes);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return { output: result, elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot() };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index]!.release();
    }
  }
}
