import {
  planShards, shardBindings, shardLoader, shardStorer, shardWordLoader, type ShardLayout,
} from "../runtime/sharded.js";
import {
  createAttentionNormalizeShader,
  createAttentionOutputShader,
  createAttentionStatisticsShader,
  ATTENTION_OUTPUT_SHADER,
  ATTENTION_OUTPUT_RESIDUAL_SHADER,
  ATTENTION_PAIR_BIAS_SHADER,
  attentionKeyValueStorage,
  attentionQueriesPerThread,
  attentionProjectShader,
  createAttentionRegisterFlashShader,
  ATTENTION_WINDOW_TARGET_BYTES,
  attentionBatchWindow,
  createAttentionNormParameters,
  createAttentionParameters,
  packAttentionWeights,
  type AttentionPairBias,
  type AttentionWeights,
} from "./attention.js";
import { attentionFlashKernelForShape } from "./attention-calibration.js";
import { calibrateAttentionShape } from "../runtime/attention-queries.js";
import { createTiledGemmShader, GEMM_TILE_ROWS, gemmGrid } from "../runtime/gemm.js";
import { releaseScratch } from "./execution-scratch.js";
import {
  createOuterProductMeanNormalizeShader,
  createOuterProductMeanParameters,
  createOuterProductMeanContractShader,
  OUTER_PRODUCT_MEAN_PAIR_COUNT_SHADER,
  OUTER_PRODUCT_BLOCK_LIMIT_BYTES,
  OUTER_PRODUCT_NORMALIZE_WINDOW_BYTES,
  outerProductMeanNormalizeWindow,
  outerProductMeanRowBlock,
  OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER,
  OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_RESIDUAL_SHADER, createOuterProductMeanProjectOutputShader,
  OUTER_PRODUCT_MEAN_NORMALIZE_SHADER,
  OUTER_PRODUCT_MEAN_PROJECT_SHADER,
  packOuterProductMeanWeights,
  type OuterProductMeanWeights,
} from "./outer-product-mean.js";
import {
  createTransitionNormalizeParameters,
  createTransitionShaders,
  packTransitionWeights,
  transitionChunkRows,
  TRANSITION_TILE_COLUMNS,
  TRANSITION_TILE_ROWS,
  type TransitionWeights,
} from "./transition.js";
import { WebGpuExecution, type GpuTensor } from "../runtime/execution.js";
import { type TriangleWholeStorage, createTriangleShaders, type TriangleDirection } from "../triangle/shaders.js";
import {
  type ActivationStorage, packHalfWords, storageArray, storageWords, storedElement, unpackHalfWords,
} from "../runtime/storage.js";
import type { TriangleMultiplicationWeights } from "../triangle/types.js";
import { packWeights as packTriangleWeights } from "../triangle/weights.js";
import type { AllocationSnapshot } from "../runtime/allocator.js";

export interface AttentionModuleWeights {
  readonly heads: number;
  readonly attention: AttentionWeights;
}

export interface RowAttentionModuleWeights extends AttentionModuleWeights {
  readonly pairLayerNormScale: Float32Array;
  readonly pairLayerNormOffset: Float32Array;
  readonly pairProjectionWeight: Float32Array;
}

export interface TriangleAttentionModuleWeights extends AttentionModuleWeights {
  readonly pairProjectionWeight: Float32Array;
}

export interface EvoformerBlockWeights {
  readonly msaRowAttention: RowAttentionModuleWeights;
  readonly msaColumnAttention: AttentionModuleWeights;
  readonly msaTransition: TransitionWeights;
  readonly outerProductMean: OuterProductMeanWeights;
  readonly triangleMultiplicationOutgoing: TriangleMultiplicationWeights;
  readonly triangleMultiplicationIncoming: TriangleMultiplicationWeights;
  readonly triangleAttentionStarting: TriangleAttentionModuleWeights;
  readonly triangleAttentionEnding: TriangleAttentionModuleWeights;
  readonly pairTransition: TransitionWeights;
}

/**
 * Ends the command buffer under construction and returns its successor.
 *
 * Everything already encoded is submitted, so the queue keeps the ordering the
 * caller wrote, and the caller continues on the returned encoder.
 */
export type SubmissionFlush = (label: string) => Promise<GPUCommandEncoder>;

/**
 * Splits the command buffer once a loop has put enough dispatches in it.
 *
 * Returns the encoder to keep using, which is a new one after a split, and the
 * dispatch count the next split measures from.
 */
async function splitWhenLong(
  execution: WebGpuExecution, encoder: GPUCommandEncoder, since: number,
  flush: SubmissionFlush | undefined, label: string,
): Promise<readonly [GPUCommandEncoder, number]> {
  if (flush === undefined || execution.dispatchCount - since < execution.submissionDispatchLimit) {
    return [encoder, since];
  }
  return [await flush(label), execution.dispatchCount];
}

export interface EvoformerBlockInput {
  readonly msa: Float32Array;
  readonly pair: Float32Array;
  readonly msaMask: Float32Array;
  readonly pairMask: Float32Array;
  readonly sequences: number;
  readonly length: number;
  readonly cM: number;
  readonly cZ: number;
  readonly cOuter: number;
  readonly triangleHidden: number;
  /** Multimer-v3 applies outer-product mean before MSA attention. */
  readonly outerProductMeanFirst?: boolean;
  /** Overrides the scratch budget of every windowed operation, so tests can force windowing. */
  readonly scratchWindowBytes?: number;
  /** Storage of the triangle multiplication's whole projection; `f16` halves it inexactly. */
  readonly triangleWholeStorage?: TriangleWholeStorage;
  /** Storage of the MSA activations this block reads and updates; `f16` halves them inexactly. */
  readonly msaStorage?: ActivationStorage;
  /**
   * Storage of the pair this block reads and updates; `f16` halves it
   * inexactly. The pair is one of the three tensors that set the trunk's peak,
   * beside the MSA and the triangle multiplication's whole projection.
   */
  readonly pairStorage?: ActivationStorage;
  /**
   * Closes the command buffer being built and opens the next one.
   *
   * A long chain puts thousands of dispatches in one submission, and a driver
   * that waits seconds for a command buffer to finish is a driver that decides
   * the GPU has hung. The loops that grow with the shape call this so no
   * submission covers more than a bounded number of dispatches.
   */
  readonly flush?: SubmissionFlush;
  /**
   * Bytes one binding may cover of the pair, defaulting to the device's limit.
   *
   * Lowering it forces the sharded path on a device whose limit is generous,
   * which is the only way to exercise it here.
   */
  readonly pairBindingBytes?: number;
  /** The same, for the MSA activations. */
  readonly msaBindingBytes?: number;
  readonly weights: EvoformerBlockWeights;
}

export interface EvoformerBlockResult {
  readonly msa: Float32Array;
  readonly pair: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

export interface EvoformerPairBlockWeights {
  readonly outerProductMean: OuterProductMeanWeights;
  readonly triangleMultiplicationOutgoing: TriangleMultiplicationWeights;
  readonly triangleMultiplicationIncoming: TriangleMultiplicationWeights;
  readonly triangleAttentionStarting: TriangleAttentionModuleWeights;
  readonly triangleAttentionEnding: TriangleAttentionModuleWeights;
  readonly pairTransition: TransitionWeights;
}

export interface GlobalAttentionWeights {
  readonly queryNormScale: Float32Array;
  readonly queryNormOffset: Float32Array;
  readonly queryWeight: Float32Array;
  readonly keyWeight: Float32Array;
  readonly valueWeight: Float32Array;
  readonly gatingWeight: Float32Array;
  readonly gatingBias: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
  readonly heads: number;
}

export interface GlobalAttentionInput {
  readonly activations: Float32Array;
  readonly mask: Float32Array;
  readonly sequences: number;
  readonly length: number;
  readonly channels: number;
  readonly weights: GlobalAttentionWeights;
}

export interface GlobalAttentionResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

export interface ExtraMsaBlockWeights extends EvoformerPairBlockWeights {
  readonly msaRowAttention: RowAttentionModuleWeights;
  readonly msaColumnGlobalAttention: GlobalAttentionWeights;
  readonly msaTransition: TransitionWeights;
}

export type TemplatePairBlockWeights = Omit<EvoformerPairBlockWeights, "outerProductMean">;

type EvoformerShape = Pick<
  EvoformerBlockInput,
  "sequences" | "length" | "cM" | "cZ" | "cOuter" | "triangleHidden" | "outerProductMeanFirst"
  | "scratchWindowBytes" | "triangleWholeStorage" | "msaStorage" | "pairStorage" | "flush"
  | "pairBindingBytes" | "msaBindingBytes"
>;

const GLOBAL_ATTENTION_COMMON = `
struct Parameters {
  length: u32, sequences: u32, channels: u32, heads: u32, head_dim: u32,
  query_weight: u32, key_weight: u32, value_weight: u32, gating_weight: u32,
  gating_bias: u32, output_weight: u32, output_bias: u32,
  norm_scale: u32, norm_offset: u32, padding_0: u32, padding_1: u32,
};
const GRID_WIDTH: u32 = 32768u;
`;

/**
 * Reads one normalized element of the extra MSA.
 *
 * The three consumers of the normalization used to share a tensor as large as
 * the MSA itself, which packing the activations made the largest thing alive
 * in the extra stack. Each now normalizes while loading, from the row
 * statistics, so the row's mean and inverse standard deviation stand in for a
 * whole normalized copy.
 */
function globalAttentionLoader(binding = "source"): string {
  return `
fn normalized_element(source_row: u32, c: u32) -> f32 {
  return (${binding}_load(source_row * p.channels + c) - statistics[2u * source_row])
    * statistics[2u * source_row + 1u] * weights[p.norm_scale + c] + weights[p.norm_offset + c];
}`;
}

/** One binding covering the whole extra MSA, which is the common case. */
const GLOBAL_UNSHARDED: ShardLayout = {
  count: 1, shardElements: Number.MAX_SAFE_INTEGER, totalElements: 0,
};

function createGlobalAttentionKvShader(
  storage: ActivationStorage, shards: ShardLayout = GLOBAL_UNSHARDED,
): string {
  const n = shards.count;
  return `${GLOBAL_ATTENTION_COMMON}
${shardBindings(shards, "source", storage, 0, false)}
@group(0) @binding(${n}) var<storage, read> weights: array<f32>;
@group(0) @binding(${n + 1}) var<uniform> p: Parameters;
@group(0) @binding(${n + 2}) var<storage, read_write> keys: array<f32>;
@group(0) @binding(${n + 3}) var<storage, read_write> values: array<f32>;
@group(0) @binding(${n + 4}) var<storage, read> statistics: array<f32>;
${shardLoader(shards, "source", storage)}
${globalAttentionLoader()}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.sequences * p.head_dim) { return; }
  let d = index % p.head_dim; let row = index / p.head_dim;
  // Rows here run column-major over the MSA, which is stored sequence-major.
  let source_row = (row % p.sequences) * p.length + row / p.sequences;
  var key = 0.0; var value = 0.0;
  for (var c = 0u; c < p.channels; c += 1u) {
    let x = normalized_element(source_row, c);
    key += x * weights[p.key_weight + c * p.head_dim + d];
    value += x * weights[p.value_weight + c * p.head_dim + d];
  }
  keys[index] = key; values[index] = value;
}`;
}

/**
 * Mask-weighted mean over sequences, shared by every query channel.
 *
 * Global column attention derives its single query per column from the masked
 * mean of that column's sequences. The mean does not depend on the head or the
 * head channel, so it is computed once here rather than once per projected
 * channel inside the query projection.
 */
function createGlobalAttentionColumnMeanShader(
  storage: ActivationStorage, shards: ShardLayout = GLOBAL_UNSHARDED,
): string {
  const n = shards.count;
  return `${GLOBAL_ATTENTION_COMMON}
${shardBindings(shards, "source", storage, 0, false)}
@group(0) @binding(${n}) var<storage, read> mask: array<f32>;
@group(0) @binding(${n + 1}) var<uniform> p: Parameters;
@group(0) @binding(${n + 2}) var<storage, read_write> means: array<f32>;
@group(0) @binding(${n + 3}) var<storage, read> statistics: array<f32>;
@group(0) @binding(${n + 4}) var<storage, read> weights: array<f32>;
${shardLoader(shards, "source", storage)}
${globalAttentionLoader()}
var<workgroup> column_denominator: array<f32, 1>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let column = group.x + group.y * GRID_WIDTH;
  if (column >= p.length) { return; }
  if (local.x == 0u) {
    var denominator = 1e-10;
    for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
      denominator += mask[sequence * p.length + column];
    }
    column_denominator[0] = denominator;
  }
  workgroupBarrier();
  let inverse_denominator = 1.0 / column_denominator[0];
  for (var c = local.x; c < p.channels; c += 64u) {
    var total = 0.0;
    for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
      total += normalized_element(sequence * p.length + column, c) * mask[sequence * p.length + column];
    }
    means[column * p.channels + c] = total * inverse_denominator;
  }
}`;
}

/** Projects the per-column mean into every head's query. */
function globalAttentionQueryShader(): string {
  // Built on demand, not at module load: the projection variant is installed
  // while the device is created, and this module is imported before that
  // happens. A module-scope constant would freeze the f32 kernel in place and
  // quietly miss every selection made afterwards.
  return createTiledGemmShader({
    preamble: `${GLOBAL_ATTENTION_COMMON}
@group(0) @binding(0) var<storage, read> means: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> query: array<f32>;`,
    rows: "p.length",
    inner: "p.channels",
    columns: "p.heads * p.head_dim",
    sourceElement: "means[row * p.channels + k]",
    weightElement: "weights[p.query_weight + k * p.heads * p.head_dim + column]",
    store: `query[row * p.heads * p.head_dim + column] = element * inverseSqrt(f32(p.head_dim));`,
  });
}

/**
 * Global column attention over the sequence axis.
 *
 * One query attends to every sequence of a column, so the whole reduction used
 * to run on a single invocation per column and head: 472 invocations for the
 * extra-MSA stack, which left the device idle. Each workgroup now splits the
 * sequences across 64 invocations that each keep a partial online softmax, and
 * combines them pairwise with the standard rescaling.
 */
const GLOBAL_ATTENTION_FLASH_SHADER = `${GLOBAL_ATTENTION_COMMON}
const LANES: u32 = 64u;
const MAX_HEAD_DIM: u32 = 32u;
@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> keys: array<f32>;
@group(0) @binding(2) var<storage, read> values: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
var<workgroup> partial_accumulated: array<f32, 2048>;
var<workgroup> partial_maximum: array<f32, LANES>;
var<workgroup> partial_sum: array<f32, LANES>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let column = group.x;
  let head = group.y;
  if (column >= p.length || head >= p.heads) { return; }
  let lane = local.x;
  let accumulated_base = lane * MAX_HEAD_DIM;
  let query_base = (column * p.heads + head) * p.head_dim;
  var maximum = -1e30;
  var denominator = 0.0;
  for (var d = 0u; d < p.head_dim; d += 1u) { partial_accumulated[accumulated_base + d] = 0.0; }

  for (var sequence = lane; sequence < p.sequences; sequence += LANES) {
    let key_base = (column * p.sequences + sequence) * p.head_dim;
    var logit = 0.0;
    for (var d = 0u; d < p.head_dim; d += 1u) {
      logit += query[query_base + d] * keys[key_base + d];
    }
    if (mask[sequence * p.length + column] == 0.0) { logit = -1e9; }
    let next_maximum = max(maximum, logit);
    let previous_scale = exp(maximum - next_maximum);
    let weight = exp(logit - next_maximum);
    denominator = denominator * previous_scale + weight;
    for (var d = 0u; d < p.head_dim; d += 1u) {
      partial_accumulated[accumulated_base + d] = partial_accumulated[accumulated_base + d] * previous_scale
        + weight * values[key_base + d];
    }
    maximum = next_maximum;
  }
  partial_maximum[lane] = maximum;
  partial_sum[lane] = denominator;
  workgroupBarrier();

  for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
    if (lane < stride) {
      let other = lane + stride;
      let merged_maximum = max(partial_maximum[lane], partial_maximum[other]);
      let scale_self = exp(partial_maximum[lane] - merged_maximum);
      let scale_other = exp(partial_maximum[other] - merged_maximum);
      partial_sum[lane] = partial_sum[lane] * scale_self + partial_sum[other] * scale_other;
      partial_maximum[lane] = merged_maximum;
      let other_base = other * MAX_HEAD_DIM;
      for (var d = 0u; d < p.head_dim; d += 1u) {
        partial_accumulated[accumulated_base + d] = partial_accumulated[accumulated_base + d] * scale_self
          + partial_accumulated[other_base + d] * scale_other;
      }
    }
    workgroupBarrier();
  }

  for (var d = lane; d < p.head_dim; d += LANES) {
    output[query_base + d] = partial_accumulated[d] / partial_sum[0];
  }
}`;

/**
 * Gated output projection for global column attention.
 *
 * The gate for one row and one projected channel is a full contraction over
 * the input channels, and it does not depend on the output channel. Computing
 * it inside the output loop, as a straightforward transcription does, repeats
 * that contraction once per output channel and made this the most expensive
 * kernel in the extra-MSA block by a wide margin. Expressing the projection as
 * a tiled GEMM whose A element is the gated attention value evaluates each gate
 * exactly once, because the extra-MSA channel count is one column tile wide.
 */
/**
 * The residual form reads and writes one buffer.
 *
 * Its gate needs every channel of a row, which is now read from the MSA
 * itself rather than from a normalized copy, and WebGPU rejects a buffer bound
 * read-only and read-write in one dispatch. A single read-write binding is
 * exact here: the extra-MSA channel count is one column tile wide, so a
 * workgroup owns whole rows, and the contraction's last barrier separates
 * every gate read from every store.
 */
function createGlobalAttentionOutputShader(
  residual: boolean, storage: ActivationStorage = "f32", shards: ShardLayout = GLOBAL_UNSHARDED,
): string {
  const sourceBinding = residual ? "output" : "source";
  // Bindings are numbered from zero in the order the dispatch passes them, so
  // the residual form's list is one set shorter rather than one hole longer.
  let binding = 0;
  const next = (): number => binding++;
  const take = (count: number): number => { const first = binding; binding += count; return first; };
  const sourceDeclaration = residual ? "" : shardBindings(shards, "source", storage, take(shards.count), false);
  return createTiledGemmShader({
    preamble: `${GLOBAL_ATTENTION_COMMON}
${sourceDeclaration}
@group(0) @binding(${next()}) var<storage, read> attended: array<f32>;
@group(0) @binding(${next()}) var<storage, read> weights: array<f32>;
@group(0) @binding(${next()}) var<uniform> p: Parameters;
${shardBindings(shards, "output", storage, take(shards.count), true)}
@group(0) @binding(${next()}) var<storage, read> statistics: array<f32>;
${residual ? "" : shardLoader(shards, "source", storage)}
${shardStorer(shards, "output", storage)}
${storage === "f16" || residual ? shardLoader(shards, "output", storage) : ""}
${storage === "f16" ? shardWordLoader(shards, "output") : ""}
${globalAttentionLoader(sourceBinding)}

// Rows and the MSA are both sequence-major here.
fn gated_attention(row: u32, projected_channel: u32) -> f32 {
  let column = row % p.length;
  let projected = p.heads * p.head_dim;
  var gate = weights[p.gating_bias + projected_channel];
  for (var c = 0u; c < p.channels; c += 1u) {
    gate += normalized_element(row, c) * weights[p.gating_weight + c * projected + projected_channel];
  }
  return attended[column * projected + projected_channel] / (1.0 + exp(-gate));
}`,
    rows: "p.sequences * p.length",
    inner: "p.heads * p.head_dim",
    columns: "p.channels",
    sourceElement: "gated_attention(row, k)",
    weightElement: "weights[p.output_weight + k * p.channels + column]",
    store: `let index = row * p.channels + column;
          let written = element + weights[p.output_bias + column];
          output_store(index, ${residual ? "output_load(index) + written" : "written"});`,
    // Packed storage is written a word (two adjacent channels) at a time.
    ...(storage === "f16" ? { storeVector: `let base = row * p.channels + column;
${[0, 2].map((pair) => `          if (column + ${pair + 1}u < p.channels) {
            var stored = vec2<f32>(values[${pair}] + weights[p.output_bias + column + ${pair}u],
              values[${pair + 1}] + weights[p.output_bias + column + ${pair + 1}u]);
            ${residual ? `stored += unpack2x16float(output_load_word((base + ${pair}u) >> 1u));` : ""}
            output_store((base + ${pair}u) >> 1u, pack2x16float(stored));
          }`).join("\n")}` } : {}),
  });
}

const GLOBAL_ATTENTION_OUTPUT_SHADER = createGlobalAttentionOutputShader(false);
const GLOBAL_ATTENTION_OUTPUT_RESIDUAL_SHADER = createGlobalAttentionOutputShader(true);

function uniform(execution: WebGpuExecution, label: string, data: ArrayBufferView): GpuTensor {
  return execution.upload(label, data, GPUBufferUsage.UNIFORM);
}

async function encodeTransition(
  execution: WebGpuExecution,
  encoderValue: GPUCommandEncoder,
  source: GpuTensor,
  rows: number,
  channels: number,
  weightsValue: TransitionWeights,
  label: string,
  residualTarget?: GpuTensor,
  storage: ActivationStorage = "f32",
  flush?: SubmissionFlush,
): Promise<GpuTensor> {
  let encoder = encoderValue;
  const hiddenChannels = weightsValue.firstBias.length;
  const descriptor = {
    activations: new Float32Array(0), rows, channels, hiddenChannels, weights: weightsValue,
  };
  const packed = packTransitionWeights(descriptor);
  const chunkRows = transitionChunkRows(
    rows, channels, hiddenChannels, execution.transitionBufferLimit,
    execution.device.limits.minStorageBufferOffsetAlignment,
  );
  const shaders = createTransitionShaders(descriptor, packed.offsets, storage);
  const [normalize, linear, linearResidual] = await Promise.all([
    execution.pipelines.get(`block:transition:normalize:${storage}`, shaders[0]!),
    execution.pipelines.get("block:transition:linear", shaders[1]!),
    execution.pipelines.get(`block:transition:linear-residual:${storage}`, shaders[2]!),
  ]);
  const weights = execution.upload(`${label}.weights`, packed.data);
  const output = residualTarget ?? execution.allocate(`${label}.output`, rows * channels);
  if (chunkRows === rows) {
    const normalizeParams = uniform(execution, `${label}.normalize-parameters`,
      createTransitionNormalizeParameters(descriptor, packed.offsets));
    const firstParams = uniform(execution, `${label}.first-parameters`, new Uint32Array([
      rows, channels, hiddenChannels, packed.offsets[2]!, packed.offsets[3]!, 1, 0, 0,
    ]));
    const secondParams = uniform(execution, `${label}.second-parameters`, new Uint32Array([
      rows, hiddenChannels, channels, packed.offsets[4]!, packed.offsets[5]!, 0, 0, 0,
    ]));
    const normalized = execution.allocate(`${label}.normalized`, rows * channels);
    const hidden = execution.allocate(`${label}.hidden`, rows * hiddenChannels);
    const normalizeGrid = execution.linearGrid(rows, 1);
    execution.dispatch(encoder, normalize, [source, weights, normalizeParams, normalized],
      normalizeGrid[0], normalizeGrid[1], 1, `${label}.normalize`);
    execution.dispatch(encoder, linear, [normalized, weights, firstParams, hidden],
      Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(rows / TRANSITION_TILE_ROWS), 1,
      `${label}.first`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hidden, weights, secondParams, output],
      Math.ceil(channels / TRANSITION_TILE_COLUMNS), Math.ceil(rows / TRANSITION_TILE_ROWS), 1,
      `${label}.second`);
    releaseScratch([normalized, hidden], output);
    return output;
  }
  const normalized = execution.allocate(`${label}.normalized-chunk`, chunkRows * channels);
  const hidden = execution.allocate(`${label}.hidden-chunk`, chunkRows * hiddenChannels);
  let dispatchedAtSplit = execution.dispatchCount;
  for (let rowOffset = 0; rowOffset < rows; rowOffset += chunkRows) {
    [encoder, dispatchedAtSplit] = await splitWhenLong(execution, encoder, dispatchedAtSplit, flush,
      `${label}.flush-${rowOffset}`);
    const count = Math.min(chunkRows, rows - rowOffset);
    const chunkDescriptor = { ...descriptor, rows: count };
    const normalizeParams = uniform(execution, `${label}.normalize-parameters-${rowOffset}`,
      createTransitionNormalizeParameters(chunkDescriptor, packed.offsets));
    const firstParams = uniform(execution, `${label}.first-parameters-${rowOffset}`, new Uint32Array([
      count, channels, hiddenChannels, packed.offsets[2]!, packed.offsets[3]!, 1, 0, 0,
    ]));
    const secondParams = uniform(execution, `${label}.second-parameters-${rowOffset}`, new Uint32Array([
      count, hiddenChannels, channels, packed.offsets[4]!, packed.offsets[5]!, 0, 0, 0,
    ]));
    const sourceChunk = execution.view(source,
      storageWords(rowOffset * channels, storage), storageWords(count * channels, storage));
    const outputChunk = execution.view(output,
      storageWords(rowOffset * channels, storage), storageWords(count * channels, storage));
    const normalizedChunk = execution.view(normalized, 0, count * channels);
    const hiddenChunk = execution.view(hidden, 0, count * hiddenChannels);
    const normalizeGrid = execution.linearGrid(count, 1);
    execution.dispatch(encoder, normalize, [sourceChunk, weights, normalizeParams, normalizedChunk],
      normalizeGrid[0], normalizeGrid[1], 1, `${label}.normalize-${rowOffset}`);
    execution.dispatch(encoder, linear, [normalizedChunk, weights, firstParams, hiddenChunk],
      Math.ceil(hiddenChannels / TRANSITION_TILE_COLUMNS), Math.ceil(count / TRANSITION_TILE_ROWS), 1,
      `${label}.first-${rowOffset}`);
    execution.dispatch(encoder, residualTarget === undefined ? linear : linearResidual,
      [hiddenChunk, weights, secondParams, outputChunk],
      Math.ceil(channels / TRANSITION_TILE_COLUMNS), Math.ceil(count / TRANSITION_TILE_ROWS), 1,
      `${label}.second-${rowOffset}`);
  }
  releaseScratch([normalized, hidden], output);
  return output;
}

interface EncodeAttentionOptions {
  readonly source: GpuTensor;
  readonly mask: GpuTensor;
  readonly pairSource?: GpuTensor;
  readonly batch: number;
  readonly queries: number;
  readonly channels: number;
  readonly heads: number;
  readonly transpose: boolean;
  readonly weights: AttentionWeights;
  readonly pairBias?: AttentionPairBias;
  readonly label: string;
  readonly residualTarget?: GpuTensor;
  readonly windowBytes?: number | undefined;
  /** Storage of `source` (and of `residualTarget`, which is the same tensor when set). */
  readonly storage?: ActivationStorage | undefined;
  /** Storage of `pairSource`, which the bias projection normalizes window by window. */
  readonly pairStorage?: ActivationStorage | undefined;
  /** Splits the command buffer between windows; see `SubmissionFlush`. */
  readonly flush?: SubmissionFlush | undefined;
  /** Bytes one binding may cover of `source`; defaults to the device's limit. */
  readonly bindingBytes?: number | undefined;
}

async function encodeAttention(
  execution: WebGpuExecution,
  encoderValue: GPUCommandEncoder,
  options: EncodeAttentionOptions,
): Promise<GpuTensor> {
  let encoder = encoderValue;
  const descriptor = {
    activations: new Float32Array(0), mask: new Float32Array(0), batch: options.batch,
    queryLength: options.queries, channels: options.channels, heads: options.heads,
    transpose: options.transpose, weights: options.weights,
    ...(options.pairBias === undefined ? {} : { pairBias: options.pairBias }),
  };
  const packed = packAttentionWeights(descriptor);
  const flashKernel = await attentionFlashKernelForShape(
    execution.device, options.channels / options.heads, options.queries,
  );
  const storage = options.storage ?? "f32";
  // Keys and values are the operands this kernel spends its time reading, and
  // packing them as half words measured 1.29x on the shape triangle attention
  // runs. Only the register kernel reads them packed; every other flash
  // variant keeps the single-precision pair it was written against, so a
  // device that selects one of those is untouched. `pack2x16float` is core
  // WGSL, so this needs no device feature and costs no portability.
  // Every register kernel reads packed keys and values, not only the
  // one-query one: above 128 queries the shape picks the two-query variant,
  // and long chains are entirely above 128 queries, so gating on the exact
  // name left the packing switched off for precisely the predictions it was
  // built for.

  // The shape rule takes two queries per invocation above 128 of them, from a
  // threshold its own comment records as measured on an NVIDIA GB10. On Apple
  // that is 2.2x the wrong way and costs a 1,416-residue complex 40% of its
  // runtime, so the count is measured here instead.
  //
  // But only where the rule would have taken two. Below its threshold the rule
  // is right on both devices — the same comment records two queries at 0.89x
  // for 59 of them — and a probe run at 512 queries has nothing to say about a
  // shape with 59. So a small shape keeps one query whatever the probe found,
  // and the measurement decides only the case the rule was guessing at.
  const registerFamily = flashKernel.variant.startsWith("register");
  const byShape = registerFamily ? flashKernel.queryTile / 64 : 1;
  const choice = registerFamily
    ? await calibrateAttentionShape(execution.device, options.channels / options.heads)
    : undefined;
  const slots = attentionQueriesPerThread(
    byShape === 1 || choice === undefined ? byShape : choice.slots);
  // Packing halves what the key loop reads and costs an unpack for each, which
  // is 1.29x on a device waiting for memory and may be nothing on one waiting
  // for issue slots. Measured with the query count, for the same reason.
  const keyValueStorage = attentionKeyValueStorage(
    registerFamily && choice !== undefined ? choice.keyValue : "f32");
  const flashShader = flashKernel.variant.startsWith("register")
    ? createAttentionRegisterFlashShader(
      options.channels / options.heads, slots, keyValueStorage)
    : flashKernel.shader;
  const flashQueryTile = flashKernel.variant.startsWith("register")
    ? 64 * slots : flashKernel.queryTile;
  // The source and the residual target are the same tensor, and at long chain
  // lengths it outgrows one binding, so both are bound as windows of it.
  const sourceShards = planShards(options.batch * options.queries * options.channels, options.channels,
    options.bindingBytes ?? execution.bindingLimitBytes, storage === "f16" ? 2 : 4);
  // The output projection binds the windows beside the weighted values, the
  // weights and the parameters; the normalize binds them beside three more.
  const attentionSlots = sourceShards.count + 3;
  if (attentionSlots > execution.device.limits.maxStorageBuffersPerShaderStage) {
    throw new RangeError(`${options.label} needs ${attentionSlots} storage bindings for ${sourceShards.count} `
      + `windows of its input, past this device's limit of `
      + `${execution.device.limits.maxStorageBuffersPerShaderStage}. Its `
      + `${(execution.bindingLimitBytes / 1024 ** 2).toFixed(0)} MiB binding limit is what forces the windows.`);
  }
  const shardKey = `${storage}:${sourceShards.count}`;
  const [normalize, project, pairProject, flash, outputProject, pairNormalize] = await Promise.all([
    execution.pipelines.get(`block:attention:normalize:${shardKey}`,
      createAttentionNormalizeShader(storage, sourceShards)),
    execution.pipelines.get(`block:attention:project:${keyValueStorage}`,
      attentionProjectShader(keyValueStorage)),
    execution.pipelines.get("block:attention:pair-bias", ATTENTION_PAIR_BIAS_SHADER),
    execution.pipelines.get(
      `block:${flashKernel.cacheKey}:kv-${keyValueStorage}:q${slots}`, flashShader),
    execution.pipelines.get(
      `block:attention:output${options.residualTarget === undefined ? "" : "-residual"}:${shardKey}`,
      createAttentionOutputShader(options.residualTarget !== undefined, storage, sourceShards),
    ),
    // The pair bias source has its own storage, which need not match the
    // attention source's: MSA row attention reads a pair, not an MSA.
    execution.pipelines.get(`block:attention:normalize:${options.pairStorage ?? "f32"}`,
      createAttentionNormalizeShader(options.pairStorage ?? "f32")),
  ]);
  const wholeRows = options.batch * options.queries;
  const weights = execution.upload(`${options.label}.weights`, packed.data);
  const output = options.residualTarget ?? execution.allocate(
    `${options.label}.output`, wholeRows * options.channels);

  // Attention is independent across batch entries, so the per-row tensors only
  // ever have to hold one window of them.
  const windowBatch = attentionBatchWindow(options.batch, options.queries, options.channels,
    Math.min(options.windowBytes ?? ATTENTION_WINDOW_TARGET_BYTES, execution.bindingLimitBytes));
  const windowElements = windowBatch * options.queries * options.channels;

  const normalized = execution.allocate(`${options.label}.normalized`, windowElements);
  const windows: { readonly offset: number; readonly count: number }[] = [];
  for (let offset = 0; offset < options.batch; offset += windowBatch) {
    windows.push({ offset, count: Math.min(windowBatch, options.batch - offset) });
  }
  const windowParameters = windows.map(({ offset, count }) => ({
    offset, count,
    attention: uniform(execution, `${options.label}.parameters-${offset}`,
      createAttentionParameters(descriptor, packed.offsets, { offset, count })),
    norm: uniform(execution, `${options.label}.norm-parameters-${offset}`,
      createAttentionNormParameters(
        count * options.queries, options.channels, packed.offsets[0]!, packed.offsets[1]!, options.transpose,
        count, options.queries, 1e-5, offset, options.batch,
      )),
  }));
  const shardsOf = (tensor: GpuTensor): readonly GpuTensor[] => {
    if (sourceShards.count === 1) return [tensor];
    const total = options.batch * options.queries * options.channels;
    return Array.from({ length: sourceShards.count }, (_, index) => {
      const offset = index * sourceShards.shardElements;
      const count = Math.min(sourceShards.shardElements, total - offset);
      return execution.view(tensor, storageWords(offset, storage), storageWords(count, storage));
    });
  };
  const sourceViews = shardsOf(options.source);
  const normalizeWindow = (window: (typeof windowParameters)[number]): GpuTensor => {
    const rows = window.count * options.queries;
    const target = execution.view(normalized, 0, rows * options.channels);
    const grid = execution.linearGrid(rows, 1);
    execution.dispatch(encoder, normalize, [...sourceViews, weights, window.norm, target],
      grid[0], grid[1], 1, `${options.label}.normalize-${window.offset}`);
    return target;
  };

  let normalizedPair: GpuTensor | undefined;
  const pairBiasElements = options.pairBias === undefined ? 1 : options.heads * options.queries * options.queries;
  const pairBias = execution.allocate(`${options.label}.pair-bias`, pairBiasElements);
  if (options.pairBias !== undefined) {
    if (options.pairBias.source === "separate") {
      if (options.pairSource === undefined) throw new Error("separate attention pair bias requires a GPU source");
      // The separate source is the pair itself. Its normalization is only an
      // input to the bias projection, so it is produced one window of pair
      // rows at a time rather than as a whole pair-shaped tensor.
      const channels = options.pairBias.channels;
      const rowElements = options.queries * channels;
      const pairWindowRows = Math.max(1, Math.min(options.queries, Math.floor(
        Math.min(options.windowBytes ?? ATTENTION_WINDOW_TARGET_BYTES, execution.bindingLimitBytes)
          / (rowElements * Float32Array.BYTES_PER_ELEMENT),
      )));
      normalizedPair = execution.allocate(`${options.label}.pair-normalized`, pairWindowRows * rowElements);
      for (let offset = 0; offset < options.queries; offset += pairWindowRows) {
        const count = Math.min(pairWindowRows, options.queries - offset);
        const rows = count * options.queries;
        const pairNormParams = uniform(execution, `${options.label}.pair-norm-parameters-${offset}`,
          createAttentionNormParameters(
            rows, channels, packed.offsets[9]!, packed.offsets[10]!, false, 1, rows, 1e-5,
          ));
        const target = execution.view(normalizedPair, 0, rows * channels);
        const pairGrid = execution.linearGrid(rows, 1);
        const pairStorage = options.pairStorage ?? "f32";
        execution.dispatch(encoder, pairNormalize, [
          execution.view(options.pairSource, storageWords(offset * rowElements, pairStorage),
            storageWords(rows * channels, pairStorage)),
          weights, pairNormParams, target,
        ], pairGrid[0], pairGrid[1], 1, `${options.label}.pair-normalize-${offset}`);
        const params = uniform(execution, `${options.label}.pair-parameters-${offset}`,
          createAttentionParameters(descriptor, packed.offsets, { offset, count }));
        const grid = execution.linearGrid(options.heads * rows);
        execution.dispatch(encoder, pairProject, [target, weights, params, pairBias],
          grid[0], grid[1], 1, `${options.label}.pair-bias-${offset}`);
      }
      releaseScratch([normalizedPair], output);
    } else {
      // The bias is read by every window's attention but is derived from the
      // normalized input one window at a time, so it is built in a pass of its
      // own first. That costs a second normalization rather than a whole
      // pair-shaped tensor held across the operation.
      for (const window of windowParameters) {
        const source = normalizeWindow(window);
        const grid = execution.linearGrid(options.heads * window.count * options.queries);
        execution.dispatch(encoder, pairProject, [source, weights, window.attention, pairBias],
          grid[0], grid[1], 1, `${options.label}.pair-bias-${window.offset}`);
      }
    }
  }

  const outputViews = output === options.source ? sourceViews : shardsOf(output);
  const query = execution.allocate(`${options.label}.query`, windowElements);
  // The projection writes each in the width the flash kernel will read it.
  const keyWords = keyValueStorage === "f16" || keyValueStorage === "f16-key" ? "f16" : "f32";
  const valueWords = keyValueStorage === "f16" || keyValueStorage === "f16-value" ? "f16" : "f32";
  const key = execution.allocate(`${options.label}.key`, storageWords(windowElements, keyWords));
  const value = execution.allocate(`${options.label}.value`, storageWords(windowElements, valueWords));
  const gate = execution.allocate(`${options.label}.gate`, windowElements);
  // Within a window the normalized input dies at the projection and the
  // attention result is born at the flash, so one windowed tensor serves both,
  // and no dispatch binds it in both roles. The next window renormalizes over
  // it only after this window's output projection has read it.
  const weighted = normalized;

  let dispatchedAtSplit = execution.dispatchCount;
  for (const window of windowParameters) {
    [encoder, dispatchedAtSplit] = await splitWhenLong(execution, encoder, dispatchedAtSplit, options.flush,
      `${options.label}.flush-${window.offset}`);
    const { offset, count } = window;
    const rows = count * options.queries;
    const params = window.attention;
    const windowNormalized = normalizeWindow(window);
    const projectGrid = gemmGrid(rows, 4 * options.channels);
    execution.dispatch(encoder, project, [windowNormalized, weights, params, query, key, value, gate],
      projectGrid[0], projectGrid[1], 1, `${options.label}.project-${offset}`);
    execution.dispatch(encoder, flash, [query, key, value, gate, options.mask, pairBias, params, weighted],
      Math.ceil(options.queries / flashQueryTile), count, options.heads,
      `${options.label}.flash-${offset}`);
    const outputGrid = gemmGrid(rows, options.channels);
    execution.dispatch(encoder, outputProject, [weighted, weights, params, ...outputViews],
      outputGrid[0], outputGrid[1], 1, `${options.label}.output-${offset}`);
  }
  releaseScratch([pairBias, normalized, query, key, value, gate, weighted], output);
  return output;
}

async function encodeGlobalAttention(
  execution: WebGpuExecution,
  encoder: GPUCommandEncoder,
  source: GpuTensor,
  mask: GpuTensor,
  shape: EvoformerShape,
  weightsValue: GlobalAttentionWeights,
  label: string,
  residualTarget?: GpuTensor,
): Promise<GpuTensor> {
  const w = weightsValue;
  const tensors = [w.queryNormScale, w.queryNormOffset, w.queryWeight, w.keyWeight, w.valueWeight,
    w.gatingWeight, w.gatingBias, w.outputWeight, w.outputBias] as const;
  const offsets: number[] = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const packed = new Float32Array(size);
  tensors.forEach((tensor, index) => packed.set(tensor, offsets[index]));
  const headDim = w.gatingBias.length / w.heads;
  const params = new Uint32Array([
    shape.length, shape.sequences, shape.cM, w.heads, headDim,
    offsets[2]!, offsets[3]!, offsets[4]!, offsets[5]!, offsets[6]!, offsets[7]!, offsets[8]!,
    // The normalization the consumers apply while loading.
    offsets[0]!, offsets[1]!, 0, 0,
  ]);
  const storage = shape.msaStorage ?? "f32";
  // Every kernel here reads the extra MSA down its columns, so none of them
  // can take a window of rows; past the binding limit it arrives as several.
  const shards = planShards(shape.sequences * shape.length * shape.cM, shape.cM,
    shape.msaBindingBytes ?? execution.bindingLimitBytes, storage === "f16" ? 2 : 4);
  const key = `${storage}:${shards.count}`;
  const slots = shards.count * (residualTarget === undefined ? 2 : 1) + 3;
  if (slots > execution.device.limits.maxStorageBuffersPerShaderStage) {
    throw new RangeError(`${label} needs ${slots} storage bindings for ${shards.count} windows of the extra `
      + `alignment, past this device's limit of `
      + `${execution.device.limits.maxStorageBuffersPerShaderStage}. Fewer extra MSA rows will run.`);
  }
  const [statisticsPipeline, kvPipeline, columnMeanPipeline, queryPipeline, flashPipeline, outputPipeline]
    = await Promise.all([
    execution.pipelines.get(`block:attention:statistics:${key}`,
      createAttentionStatisticsShader(storage, shards)),
    execution.pipelines.get(`block:global-attention:kv:${key}`, createGlobalAttentionKvShader(storage, shards)),
    execution.pipelines.get(`block:global-attention:column-mean:${key}`,
      createGlobalAttentionColumnMeanShader(storage, shards)),
    execution.pipelines.get("block:global-attention:query", globalAttentionQueryShader()),
    execution.pipelines.get("block:global-attention:flash", GLOBAL_ATTENTION_FLASH_SHADER),
    execution.pipelines.get(
      `block:global-attention:output${residualTarget === undefined ? "" : "-residual"}:${key}`,
      createGlobalAttentionOutputShader(residualTarget !== undefined, storage, shards),
    ),
  ]);
  const shardsOf = (tensor: GpuTensor): readonly GpuTensor[] => {
    if (shards.count === 1) return [tensor];
    return Array.from({ length: shards.count }, (_, index) => {
      const offset = index * shards.shardElements;
      const count = Math.min(shards.shardElements, shards.totalElements - offset);
      return execution.view(tensor, storageWords(offset, storage), storageWords(count, storage));
    });
  };
  const sourceViews = shardsOf(source);
  const weights = execution.upload(`${label}.weights`, packed);
  const parameters = uniform(execution, `${label}.parameters`, params);
  const normParameters = uniform(execution, `${label}.norm-parameters`, createAttentionNormParameters(
    shape.length * shape.sequences, shape.cM, offsets[0]!, offsets[1]!, true,
    shape.length, shape.sequences, 1e-5,
  ));
  const statistics = execution.allocate(`${label}.statistics`, shape.length * shape.sequences * 2);
  const keys = execution.allocate(`${label}.keys`, shape.length * shape.sequences * headDim);
  const values = execution.allocate(`${label}.values`, shape.length * shape.sequences * headDim);
  const query = execution.allocate(`${label}.query`, shape.length * w.heads * headDim);
  const attended = execution.allocate(`${label}.attended`, shape.length * w.heads * headDim);
  const output = residualTarget ?? execution.allocate(`${label}.output`, shape.sequences * shape.length * shape.cM);
  let grid = execution.linearGrid(shape.length * shape.sequences, 1);
  execution.dispatch(encoder, statisticsPipeline, [...sourceViews, normParameters, statistics],
    grid[0], grid[1], 1, `${label}.statistics`);
  grid = execution.linearGrid(shape.length * shape.sequences * headDim);
  execution.dispatch(encoder, kvPipeline, [...sourceViews, weights, parameters, keys, values, statistics],
    grid[0], grid[1], 1, `${label}.kv`);
  const means = execution.allocate(`${label}.column-means`, shape.length * shape.cM);
  grid = execution.linearGrid(shape.length, 1);
  execution.dispatch(encoder, columnMeanPipeline,
    [...sourceViews, mask, parameters, means, statistics, weights],
    grid[0], grid[1], 1, `${label}.column-mean`);
  const queryGrid = gemmGrid(shape.length, w.heads * headDim);
  execution.dispatch(encoder, queryPipeline, [means, weights, parameters, query],
    queryGrid[0], queryGrid[1], 1, `${label}.query`);
  execution.dispatch(encoder, flashPipeline, [query, keys, values, mask, parameters, attended],
    shape.length, w.heads, 1, `${label}.flash`);
  const outputGrid = gemmGrid(shape.sequences * shape.length, shape.cM);
  // The residual form reads the output binding, so the source is not bound twice.
  const outputViews = output === source ? sourceViews : shardsOf(output);
  execution.dispatch(encoder, outputPipeline,
    residualTarget === undefined
      ? [...sourceViews, attended, weights, parameters, ...outputViews, statistics]
      : [attended, weights, parameters, ...outputViews, statistics],
    outputGrid[0], outputGrid[1], 1, `${label}.output`);
  releaseScratch([statistics, means, keys, values, query, attended], output);
  return output;
}

async function encodeOuterProductMean(
  execution: WebGpuExecution,
  encoderValue: GPUCommandEncoder,
  msa: GpuTensor,
  msaMask: GpuTensor,
  input: EvoformerShape,
  weightsValue: OuterProductMeanWeights,
  residualTarget?: GpuTensor,
): Promise<GpuTensor> {
  const descriptor = {
    activations: new Float32Array(0), mask: new Float32Array(0), sequences: input.sequences,
    length: input.length, cM: input.cM, cOuter: input.cOuter, cZ: input.cZ,
    weights: weightsValue,
  };
  const packed = packOuterProductMeanWeights(descriptor);
  const storage = input.msaStorage ?? "f32";
  const rows = input.sequences * input.length;
  // Both projections carry every sequence, so a deep alignment at a long
  // length puts them past one binding; the contraction reads them as windows.
  const projectionShards = planShards(rows * input.cOuter, input.cOuter, execution.bindingLimitBytes);
  const [normalize, project, contractPipeline, pairCountPipeline, projectOutputPipeline] = await Promise.all([
    execution.pipelines.get(`block:opm:normalize:${storage}`, createOuterProductMeanNormalizeShader(storage)),
    execution.pipelines.get("block:opm:project", OUTER_PRODUCT_MEAN_PROJECT_SHADER),
    execution.pipelines.get(`block:opm:contract:${projectionShards.count}`,
      createOuterProductMeanContractShader(projectionShards)),
    execution.pipelines.get("block:opm:pair-count", OUTER_PRODUCT_MEAN_PAIR_COUNT_SHADER),
    execution.pipelines.get(
      `block:opm:project-output${residualTarget === undefined ? "" : "-residual"}:${input.pairStorage ?? "f32"}`,
      createOuterProductMeanProjectOutputShader(residualTarget !== undefined, input.pairStorage ?? "f32"),
    ),
  ]);
  const weights = execution.upload("opm.weights", packed.data);
  const params = uniform(execution, "opm.parameters", createOuterProductMeanParameters(descriptor, packed.offsets));
  const normalizeRows = outerProductMeanNormalizeWindow(rows, input.cM,
    Math.min(input.scratchWindowBytes ?? OUTER_PRODUCT_NORMALIZE_WINDOW_BYTES, execution.bindingLimitBytes));
  const normalized = execution.allocate("opm.normalized", normalizeRows * input.cM);
  const projectionViews = (tensor: GpuTensor): readonly GpuTensor[] => {
    if (projectionShards.count === 1) return [tensor];
    return Array.from({ length: projectionShards.count }, (_, index) => {
      const offset = index * projectionShards.shardElements;
      return execution.view(tensor, offset,
        Math.min(projectionShards.shardElements, projectionShards.totalElements - offset));
    });
  };
  const left = execution.allocate("opm.left", rows * input.cOuter);
  const right = execution.allocate("opm.right", rows * input.cOuter);
  const rowBlock = outerProductMeanRowBlock(input.length, input.cOuter,
    Math.min(OUTER_PRODUCT_BLOCK_LIMIT_BYTES, execution.bindingLimitBytes));
  const outer = execution.allocate("opm.outer", rowBlock * input.length * input.cOuter * input.cOuter);
  const pairCount = execution.allocate("opm.pair-count", input.length * input.length);
  const output = residualTarget ?? execution.allocate("opm.output",
    storageWords(input.length * input.length * input.cZ, input.pairStorage ?? "f32"));
  // Both the normalization and the projection are row-wise, so a window of rows
  // needs only views of the whole-operation tensors and a shorter dispatch. The
  // shaders' own bounds checks are against the full row count, which a shorter
  // dispatch never reaches.
  let encoder = encoderValue;
  let dispatchedAtSplit = execution.dispatchCount;
  for (let offset = 0; offset < rows; offset += normalizeRows) {
    [encoder, dispatchedAtSplit] = await splitWhenLong(execution, encoder, dispatchedAtSplit, input.flush,
      `opm.normalize-flush-${offset}`);
    const count = Math.min(normalizeRows, rows - offset);
    const msaWindow = execution.view(msa,
      storageWords(offset * input.cM, storage), storageWords(count * input.cM, storage));
    const maskWindow = execution.view(msaMask, offset, count);
    const normalizedWindow = execution.view(normalized, 0, count * input.cM);
    const leftWindow = execution.view(left, offset * input.cOuter, count * input.cOuter);
    const rightWindow = execution.view(right, offset * input.cOuter, count * input.cOuter);
    let windowGrid = execution.linearGrid(count, 1);
    execution.dispatch(encoder, normalize, [msaWindow, weights, params, normalizedWindow],
      windowGrid[0], windowGrid[1], 1, `opm.normalize-${offset}`);
    windowGrid = execution.linearGrid(count * input.cOuter);
    execution.dispatch(encoder, project,
      [normalizedWindow, maskWindow, weights, params, leftWindow, rightWindow],
      windowGrid[0], windowGrid[1], 1, `opm.project-${offset}`);
  }
  let grid = execution.linearGrid(input.length * input.length);
  execution.dispatch(encoder, pairCountPipeline, [msaMask, params, pairCount],
    grid[0], grid[1], 1, "opm.pair-count");
  for (let offset = 0; offset < input.length; offset += rowBlock) {
    [encoder, dispatchedAtSplit] = await splitWhenLong(execution, encoder, dispatchedAtSplit, input.flush,
      `opm.contract-flush-${offset}`);
    const count = Math.min(rowBlock, input.length - offset);
    const tile = uniform(execution, `opm.block-${offset}`, new Uint32Array([offset, count, 0, 0]));
    const contractGrid = gemmGrid(count * input.cOuter, input.length * input.cOuter);
    execution.dispatch(encoder, contractPipeline,
      [...projectionViews(left), ...projectionViews(right), params, tile, outer],
      contractGrid[0], contractGrid[1], 1, "opm.contract");
    const projectOutputGrid = gemmGrid(count * input.length, input.cZ);
    // The block writes its own pair rows, so it binds only those.
    const pairStorage = input.pairStorage ?? "f32";
    const outputWindow = execution.view(output,
      storageWords(offset * input.length * input.cZ, pairStorage),
      storageWords(count * input.length * input.cZ, pairStorage));
    execution.dispatch(encoder, projectOutputPipeline, [outer, pairCount, weights, params, tile, outputWindow],
      projectOutputGrid[0], projectOutputGrid[1], 1, `opm.project-output-${offset}`);
  }
  releaseScratch([normalized, left, right, outer, pairCount], output);
  return output;
}

/**
 * Rows of the blocked residue axis one triangle step covers.
 *
 * The step holds a block of the normalized pair, a block of one projection and
 * a block of the contraction, so the budget is shared between the widest of
 * them. The other projection, or the accumulated contraction, stays whole.
 *
 * Every block streams that whole tensor again, so smaller blocks trade
 * bandwidth for memory: at 384 residues, 8 MiB blocks cost 6% of the recycle
 * against 16 MiB blocks and lower the working set by 33 MiB.
 */
export const TRIANGLE_BLOCK_TARGET_BYTES = 8 * 1024 * 1024;

/**
 * The storage binding size every WebGPU implementation guarantees.
 *
 * A block tensor is bound whole, so the block cannot be grown past this
 * however much it would help.
 */
const WEBGPU_GUARANTEED_BINDING_BYTES = 128 * 1024 * 1024;

export function triangleBlockRows(
  length: number, cZ: number, triangleHidden: number,
  budgetBytes: number = TRIANGLE_BLOCK_TARGET_BYTES,
): number {
  if (![length, cZ, triangleHidden, budgetBytes].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("triangle block dimensions must be positive safe integers");
  }
  const bytesPerRow = length * Math.max(cZ, triangleHidden) * Float32Array.BYTES_PER_ELEMENT;
  // A block shorter than the contraction's tile pays for the rows it does not
  // have. The GEMM dispatches whole 64-row tiles and masks what falls outside
  // the block, so an eleven-row block does a sixth of the work in the same
  // time: 428 GFLOP/s against 2,106 for a full tile, measured. An 8 MiB budget
  // gives exactly eleven rows at 1,416 residues, which is where a long
  // prediction spends its time.
  //
  // So the budget holds at least one whole tile. At 1,416 residues that is
  // 46 MiB rather than 8, it raises the estimated peak from 2,440 MiB to
  // 2,573 against a 5,734 MiB safety budget, and it takes the prediction from
  // 407 seconds to 310. The raise is capped at the binding size every WebGPU
  // implementation guarantees, since a block tensor is bound whole and a
  // longer chain would otherwise ask for more than one binding can hold.
  const perTile = GEMM_TILE_ROWS * bytesPerRow;
  const effective = Math.max(budgetBytes, Math.min(perTile, WEBGPU_GUARANTEED_BINDING_BYTES));
  const rows = Math.max(1, Math.min(length, Math.floor(effective / bytesPerRow)));
  // Whole tiles, so the last one is not mostly masked either — but never at
  // the cost of splitting a block that already covered the whole length, which
  // rounding down would do to any chain a little over one tile.
  const tiled = rows >= length ? length
    : rows >= GEMM_TILE_ROWS ? Math.floor(rows / GEMM_TILE_ROWS) * GEMM_TILE_ROWS
      : rows;
  // Blocks start on even pair rows so packed half-precision pairs never span
  // two words; a single block covering every row starts at zero anyway.
  return tiled < length && (length % 2 === 1) ? Math.max(2, tiled - (tiled % 2)) : tiled;
}

async function encodeTriangleMultiplication(
  execution: WebGpuExecution,
  encoderValue: GPUCommandEncoder,
  pair: GpuTensor,
  pairMask: GpuTensor,
  input: EvoformerShape,
  weightsValue: TriangleMultiplicationWeights,
  direction: TriangleDirection,
  residualTarget?: GpuTensor,
): Promise<GpuTensor> {
  const shape = { length: input.length, cZ: input.cZ, cHidden: input.triangleHidden };
  const packed = packTriangleWeights(weightsValue, "f32");
  const blockRows = triangleBlockRows(input.length, input.cZ, input.triangleHidden,
    Math.min(input.scratchWindowBytes ?? TRIANGLE_BLOCK_TARGET_BYTES, execution.bindingLimitBytes));
  const wholeStorage = input.triangleWholeStorage ?? "f32";
  const pairStorage = input.pairStorage ?? "f32";
  // A pair past the device's binding limit is bound as several windows of the
  // same buffer, which the shaders read through a generated accessor.
  const pairShards = planShards(input.length * input.length * input.cZ, input.cZ,
    input.pairBindingBytes ?? execution.bindingLimitBytes,
    pairStorage === "f16" ? 2 : 4);
  // The whole operand is the pair's size in hidden channels, so it passes the
  // binding limit before the pair does and is windowed the same way.
  const wholeStride = input.length * input.length + (input.length * input.length) % 2;
  const wholeShards = planShards(wholeStride * input.triangleHidden, 2,
    input.pairBindingBytes ?? execution.bindingLimitBytes,
    wholeStorage === "f16" ? 2 : 4);
  // The projection reads the pair and writes the whole operand in one
  // dispatch, so their windows share the stage's storage slots with three
  // more for the mask, the weights and the statistics.
  const slots = pairShards.count + wholeShards.count + 3;
  if (slots > execution.device.limits.maxStorageBuffersPerShaderStage) {
    throw new RangeError(`A ${input.length}-residue pair needs ${slots} storage bindings in the triangle `
      + `multiplication (${pairShards.count} windows of the pair and ${wholeShards.count} of its projection), `
      + `past this device's limit of ${execution.device.limits.maxStorageBuffersPerShaderStage}. `
      + `Its ${(execution.bindingLimitBytes / 1024 ** 2).toFixed(0)} MiB binding limit is what forces the `
      + "windows: a shorter sequence, or a device that binds more of a buffer at once, will run.");
  }
  const shaders = createTriangleShaders(shape, "f32", packed.offsets, 1e-5, direction, blockRows, wholeStorage,
    pairStorage, residualTarget !== undefined, pairShards, wholeShards);
  const pipelineKey = `block:triangle:${direction}:${input.length}:${input.cZ}`
    + `:${input.triangleHidden}:${blockRows}:${wholeStorage}:${pairStorage}:${pairShards.count}`
    + `:${wholeShards.count}`;
  const [inputStatistics, projectGate, projectBlockOperand, projectWholeOperand, contract, hiddenStatistics, projectOutput]
    = await Promise.all([
    execution.pipelines.get(`${pipelineKey}:input-statistics`, shaders.inputStatistics),
    execution.pipelines.get(`${pipelineKey}:project-gate`, shaders.projectGate),
    execution.pipelines.get(`${pipelineKey}:project-block-operand`, shaders.projectBlockOperand),
    execution.pipelines.get(`${pipelineKey}:project-whole-operand`, shaders.projectWholeOperand),
    execution.pipelines.get(`${pipelineKey}:contract`, shaders.contract),
    execution.pipelines.get(`${pipelineKey}:hidden-statistics`, shaders.hiddenStatistics),
    execution.pipelines.get(
      `${pipelineKey}:project-output${residualTarget === undefined ? "" : "-residual"}`,
      shaders.projectOutput,
    ),
  ]);
  const pairs = input.length * input.length;
  const views = (tensor: GpuTensor, layout: ShardLayout, storage: ActivationStorage): readonly GpuTensor[] => {
    if (layout.count === 1) return [tensor];
    return Array.from({ length: layout.count }, (_, index) => {
      const offset = index * layout.shardElements;
      const count = Math.min(layout.shardElements, layout.totalElements - offset);
      return execution.view(tensor, storageWords(offset, storage), storageWords(count, storage));
    });
  };
  const shardViews = (tensor: GpuTensor): readonly GpuTensor[] => views(tensor, pairShards, pairStorage);
  const blockPairs = blockRows * input.length;
  const weights = execution.upload(`triangle.${direction}.weights`, packed.data);
  const output = residualTarget
    ?? execution.allocate(`triangle.${direction}.output`, storageWords(pairs * input.cZ, pairStorage));
  const blocks: { readonly offset: number; readonly count: number; readonly uniform: GpuTensor }[] = [];
  for (let offset = 0; offset < input.length; offset += blockRows) {
    const count = Math.min(blockRows, input.length - offset);
    blocks.push({ offset, count,
      uniform: uniform(execution, `triangle.${direction}.block-${offset}`,
        new Uint32Array([offset * input.length, count * input.length, offset === 0 ? 1 : 0, count])) });
  }

  // Every consumer of the normalized pair normalizes the raw pair while
  // loading it, from per-row statistics computed once.
  const statistics = execution.allocate(`triangle.${direction}.statistics`, pairs * 2);
  const statisticsGrid = execution.linearGrid(pairs, 1);
  let encoder = encoderValue;
  const pairViews = shardViews(pair);
  execution.dispatch(encoder, inputStatistics, [...pairViews, statistics],
    statisticsGrid[0], statisticsGrid[1], 1, `triangle.${direction}.input-statistics`);
  const gate = execution.allocate(`triangle.${direction}.gate`, blockPairs * input.cZ);
  const project = (pipeline: GPUComputePipeline, target: readonly GpuTensor[],
    block: (typeof blocks)[number], label: string): void => {
    const grid = gemmGrid(block.count * input.length, 2 * input.triangleHidden);
    execution.dispatch(encoder, pipeline, [...pairViews, pairMask, weights, statistics, ...target, block.uniform],
      grid[0], grid[1], 1, `triangle.${direction}.${label}-${block.offset}`);
  };

  // Both directions block the output rows. The operand indexed by the output's
  // first residue is projected per block; the other has to be complete before
  // any block contracts and is filled block by block, in half precision when
  // requested.
  const wholeProjection = execution.allocate(`triangle.${direction}.whole`,
    storageWords(wholeStride * input.triangleHidden, wholeStorage === "f16" ? "f16" : "f32"));
  const wholeViews = views(wholeProjection, wholeShards, wholeStorage === "f16" ? "f16" : "f32");
  for (const block of blocks) project(projectWholeOperand, wholeViews, block, "project-whole");
  const blockedProjection = execution.allocate(`triangle.${direction}.blocked`, blockPairs * input.triangleHidden);
  const contracted = execution.allocate(`triangle.${direction}.contracted`, blockPairs * input.triangleHidden);
  const hiddenStats = execution.allocate(`triangle.${direction}.hidden-statistics`, blockPairs * 2);
  const outputViews = output === pair ? pairViews : shardViews(output);
  let dispatchedAtSplit = execution.dispatchCount;
  for (const block of blocks) {
    [encoder, dispatchedAtSplit] = await splitWhenLong(execution, encoder, dispatchedAtSplit, input.flush,
      `triangle.${direction}.flush-${block.offset}`);
    const rows = block.count * input.length;
    project(projectBlockOperand, [blockedProjection], block, "project-block");
    const contractGrid = gemmGrid(block.count, input.length);
    execution.dispatch(encoder, contract, [blockedProjection, ...wholeViews, contracted, block.uniform],
      contractGrid[0], contractGrid[1], input.triangleHidden, `triangle.${direction}.contract-${block.offset}`);
    const stats = execution.view(hiddenStats, 0, rows * 2);
    execution.dispatch(encoder, hiddenStatistics, [contracted, stats, block.uniform],
      Math.ceil(rows / 64), 1, 1, `triangle.${direction}.hidden-statistics-${block.offset}`);
    // The pair tensor is still the value that was normalized: the output
    // projection is the first thing to write it.
    const gateGrid = gemmGrid(rows, input.cZ);
    const gateBlock = execution.view(gate, 0, rows * input.cZ);
    execution.dispatch(encoder, projectGate, [...pairViews, weights, statistics, gateBlock, block.uniform],
      gateGrid[0], gateGrid[1], 1, `triangle.${direction}.project-gate-${block.offset}`);
    execution.dispatch(encoder, projectOutput,
      [gateBlock, contracted, weights, stats, ...outputViews, block.uniform],
      gateGrid[0], gateGrid[1], 1, `triangle.${direction}.project-output-${block.offset}`);
  }
  releaseScratch([statistics, gate, blockedProjection, wholeProjection, contracted, hiddenStats], output);
  return output;
}

export async function encodeEvoformerBlock(
  execution: WebGpuExecution,
  encoder: GPUCommandEncoder,
  input: EvoformerBlockInput,
  msa: GpuTensor,
  pair: GpuTensor,
  msaMask: GpuTensor,
  pairMask: GpuTensor,
): Promise<void> {
  const shapeWindowBytes = input.scratchWindowBytes;
  const applyOuterProductMean = async (): Promise<void> => {
    const update = await encodeOuterProductMean(
      execution, encoder, msa, msaMask, input, input.weights.outerProductMean, pair,
    );
    if (update !== pair) await execution.addInPlace(encoder, pair, update, "outer-product-mean.residual");
  };
  if (input.outerProductMeanFirst === true) await applyOuterProductMean();
  const row = input.weights.msaRowAttention;
  await encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, pairSource: pair, batch: input.sequences, queries: input.length,
    channels: input.cM, heads: row.heads, transpose: false, weights: row.attention,
    pairBias: {
      source: "separate", activations: new Float32Array(0), channels: input.cZ,
      layerNormScale: row.pairLayerNormScale, layerNormOffset: row.pairLayerNormOffset,
      projectionWeight: row.pairProjectionWeight,
    },
    label: "msa-row-attention",
    windowBytes: shapeWindowBytes, residualTarget: msa, storage: input.msaStorage,
    pairStorage: input.pairStorage, flush: input.flush, bindingBytes: input.msaBindingBytes,
  });

  const column = input.weights.msaColumnAttention;
  await encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, batch: input.length, queries: input.sequences,
    channels: input.cM, heads: column.heads, transpose: true, weights: column.attention,
    label: "msa-column-attention",
    windowBytes: shapeWindowBytes, residualTarget: msa, storage: input.msaStorage, flush: input.flush,
    bindingBytes: input.msaBindingBytes,
  });

  await encodeTransition(
    execution, encoder, msa, input.sequences * input.length, input.cM,
    input.weights.msaTransition, "msa-transition", msa, input.msaStorage ?? "f32", input.flush,
  );

  if (input.outerProductMeanFirst !== true) await applyOuterProductMean();

  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, input, input.weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, input, input.weights.triangleMultiplicationIncoming, "incoming", pair,
  );

  const starting = input.weights.triangleAttentionStarting;
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: input.length, queries: input.length,
    channels: input.cZ, heads: starting.heads, transpose: false, weights: starting.attention,
    pairBias: { source: "normalized-input", projectionWeight: starting.pairProjectionWeight },
    label: "triangle-attention-starting",
    windowBytes: shapeWindowBytes, residualTarget: pair, storage: input.pairStorage, flush: input.flush,
    bindingBytes: input.pairBindingBytes,
  });

  const ending = input.weights.triangleAttentionEnding;
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: input.length, queries: input.length,
    channels: input.cZ, heads: ending.heads, transpose: true, weights: ending.attention,
    pairBias: { source: "normalized-input", projectionWeight: ending.pairProjectionWeight },
    label: "triangle-attention-ending",
    windowBytes: shapeWindowBytes, residualTarget: pair, storage: input.pairStorage, flush: input.flush,
    bindingBytes: input.pairBindingBytes,
  });

  await encodeTransition(
    execution, encoder, pair, input.length * input.length, input.cZ,
    input.weights.pairTransition, "pair-transition", pair, input.pairStorage ?? "f32", input.flush,
  );
}

export async function encodeEvoformerPairBlock(
  execution: WebGpuExecution,
  encoder: GPUCommandEncoder,
  shape: EvoformerShape,
  weights: EvoformerPairBlockWeights,
  msa: GpuTensor,
  pair: GpuTensor,
  msaMask: GpuTensor,
  pairMask: GpuTensor,
  includeOuterProductMean = true,
): Promise<void> {
  const shapeWindowBytes = shape.scratchWindowBytes;
  if (includeOuterProductMean) {
    const update = await encodeOuterProductMean(
      execution, encoder, msa, msaMask, shape, weights.outerProductMean, pair,
    );
    if (update !== pair) await execution.addInPlace(encoder, pair, update, "extra.outer-product-mean.residual");
  }
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationIncoming, "incoming", pair,
  );
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionStarting.heads, transpose: false,
    weights: weights.triangleAttentionStarting.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionStarting.pairProjectionWeight,
    },
    label: "extra.triangle-attention-starting",
    windowBytes: shapeWindowBytes, residualTarget: pair, storage: shape.pairStorage, flush: shape.flush,
  });
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionEnding.heads, transpose: true,
    weights: weights.triangleAttentionEnding.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionEnding.pairProjectionWeight,
    },
    label: "extra.triangle-attention-ending",
    windowBytes: shapeWindowBytes, residualTarget: pair, storage: shape.pairStorage, flush: shape.flush,
  });
  await encodeTransition(
    execution, encoder, pair, shape.length * shape.length, shape.cZ,
    weights.pairTransition, "extra.pair-transition", pair, shape.pairStorage ?? "f32", shape.flush,
  );
}

export async function encodeExtraMsaBlock(
  execution: WebGpuExecution,
  encoder: GPUCommandEncoder,
  shape: EvoformerShape,
  weights: ExtraMsaBlockWeights,
  msa: GpuTensor,
  pair: GpuTensor,
  msaMask: GpuTensor,
  pairMask: GpuTensor,
): Promise<void> {
  const shapeWindowBytes = shape.scratchWindowBytes;
  if (shape.outerProductMeanFirst === true) {
    const update = await encodeOuterProductMean(
      execution, encoder, msa, msaMask, shape, weights.outerProductMean, pair,
    );
    if (update !== pair) await execution.addInPlace(encoder, pair, update, "extra.outer-product-mean.residual");
  }
  const row = weights.msaRowAttention;
  await encodeAttention(execution, encoder, {
    source: msa, mask: msaMask, pairSource: pair, batch: shape.sequences, queries: shape.length,
    channels: shape.cM, heads: row.heads, transpose: false, weights: row.attention,
    pairBias: {
      source: "separate", activations: new Float32Array(0), channels: shape.cZ,
      layerNormScale: row.pairLayerNormScale, layerNormOffset: row.pairLayerNormOffset,
      projectionWeight: row.pairProjectionWeight,
    },
    label: "extra.msa-row-attention",
    windowBytes: shapeWindowBytes, residualTarget: msa, storage: shape.msaStorage,
    pairStorage: shape.pairStorage, flush: shape.flush,
  });
  await encodeGlobalAttention(
    execution, encoder, msa, msaMask, shape, weights.msaColumnGlobalAttention,
    "extra.msa-column-global-attention", msa,
  );
  await encodeTransition(
    execution, encoder, msa, shape.sequences * shape.length, shape.cM, weights.msaTransition,
    "extra.msa-transition", msa, shape.msaStorage ?? "f32", shape.flush,
  );
  await encodeEvoformerPairBlock(
    execution, encoder, shape, weights, msa, pair, msaMask, pairMask,
    shape.outerProductMeanFirst !== true,
  );
}

export async function encodeTemplatePairBlock(
  execution: WebGpuExecution,
  encoder: GPUCommandEncoder,
  shape: EvoformerShape,
  weights: TemplatePairBlockWeights,
  pair: GpuTensor,
  pairMask: GpuTensor,
): Promise<void> {
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionStarting.heads, transpose: false,
    weights: weights.triangleAttentionStarting.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionStarting.pairProjectionWeight,
    },
    label: "template.triangle-attention-starting", residualTarget: pair,
  });
  await encodeAttention(execution, encoder, {
    source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
    channels: shape.cZ, heads: weights.triangleAttentionEnding.heads, transpose: true,
    weights: weights.triangleAttentionEnding.attention,
    pairBias: {
      source: "normalized-input", projectionWeight: weights.triangleAttentionEnding.pairProjectionWeight,
    },
    label: "template.triangle-attention-ending", residualTarget: pair,
  });
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationIncoming, "incoming", pair,
  );
  await encodeTransition(
    execution, encoder, pair, shape.length * shape.length, shape.cZ,
    weights.pairTransition, "template.pair-transition", pair,
  );
}

/** Multimer-v3 template pair stack order (Algorithms 16 and 17). */
export async function encodeMultimerTemplatePairBlock(
  execution: WebGpuExecution,
  encoder: GPUCommandEncoder,
  shape: EvoformerShape,
  weights: TemplatePairBlockWeights,
  pair: GpuTensor,
  pairMask: GpuTensor,
): Promise<void> {
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationOutgoing, "outgoing", pair,
  );
  await encodeTriangleMultiplication(
    execution, encoder, pair, pairMask, shape, weights.triangleMultiplicationIncoming, "incoming", pair,
  );
  for (const [module, transpose, label] of [
    [weights.triangleAttentionStarting, false, "multimer-template.triangle-attention-starting"],
    [weights.triangleAttentionEnding, true, "multimer-template.triangle-attention-ending"],
  ] as const) {
    await encodeAttention(execution, encoder, {
      source: pair, mask: pairMask, batch: shape.length, queries: shape.length,
      channels: shape.cZ, heads: module.heads, transpose, weights: module.attention,
      pairBias: { source: "normalized-input", projectionWeight: module.pairProjectionWeight },
      label, residualTarget: pair,
    });
  }
  await encodeTransition(
    execution, encoder, pair, shape.length * shape.length, shape.cZ,
    weights.pairTransition, "multimer-template.pair-transition", pair,
  );
}

export class EvoformerBlockGpu {
  readonly device: GPUDevice;

  constructor(device: GPUDevice) { this.device = device; }

  async run(input: EvoformerBlockInput): Promise<EvoformerBlockResult> {
    const execution = new WebGpuExecution(this.device);
    try {
      const msaElements = input.sequences * input.length * input.cM;
      const pairElements = input.length * input.length * input.cZ;
      if (input.msa.length !== msaElements || input.pair.length !== pairElements) {
        throw new RangeError("Evoformer block activation shape mismatch");
      }
      // Packed storage is uploaded packed and unpacked after readback, so the
      // wrapper compares against references in f32 either way.
      const msaStorage = input.msaStorage ?? "f32";
      const msa = execution.upload("block.msa", msaStorage === "f16" ? packHalfWords(input.msa) : input.msa,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pairStorage = input.pairStorage ?? "f32";
      const pair = execution.upload("block.pair",
        pairStorage === "f16" ? packHalfWords(input.pair) : input.pair,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaMask = execution.upload("block.msa-mask", input.msaMask);
      const pairMask = execution.upload("block.pair-mask", input.pairMask);
      const encoder = this.device.createCommandEncoder({ label: "evoformer-block" });
      this.device.pushErrorScope("validation");

      await encodeEvoformerBlock(execution, encoder, input, msa, pair, msaMask, pairMask);

      const msaReadback = execution.createReadback("block.msa-readback", msa, encoder);
      const pairReadback = execution.createReadback("block.pair-readback", pair, encoder);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      const [msaOutput, pairOutput] = await Promise.all([
        execution.mapFloat32(msaReadback).then((words) => msaStorage === "f16"
          ? unpackHalfWords(new Uint32Array(words.buffer, words.byteOffset, words.length), msaElements) : words),
        execution.mapFloat32(pairReadback).then((words) => pairStorage === "f16"
          ? unpackHalfWords(new Uint32Array(words.buffer, words.byteOffset, words.length), pairElements) : words),
      ]);
      return {
        msa: msaOutput,
        pair: pairOutput,
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      execution.release();
    }
  }
}

/** Standalone runner used to differentially qualify the extra-MSA global-attention kernel. */
export class GlobalAttentionGpu {
  readonly device: GPUDevice;

  constructor(device: GPUDevice) { this.device = device; }

  async run(input: GlobalAttentionInput): Promise<GlobalAttentionResult> {
    const { sequences, length, channels, weights } = input;
    if (![sequences, length, channels].every((value) => Number.isSafeInteger(value) && value > 0)) {
      throw new RangeError("global-attention dimensions must be positive safe integers");
    }
    if (input.activations.length !== sequences * length * channels
      || input.mask.length !== sequences * length) {
      throw new RangeError("global-attention activation or mask shape mismatch");
    }
    if (!Number.isSafeInteger(weights.heads) || weights.heads <= 0
      || weights.gatingBias.length % weights.heads !== 0) {
      throw new RangeError("global-attention head dimensions are invalid");
    }
    const headDim = weights.gatingBias.length / weights.heads;
    const expectedWeights: ReadonlyArray<readonly [string, Float32Array, number]> = [
      ["query norm scale", weights.queryNormScale, channels],
      ["query norm offset", weights.queryNormOffset, channels],
      ["query weight", weights.queryWeight, channels * weights.heads * headDim],
      ["key weight", weights.keyWeight, channels * headDim],
      ["value weight", weights.valueWeight, channels * headDim],
      ["gating weight", weights.gatingWeight, channels * weights.heads * headDim],
      ["output weight", weights.outputWeight, weights.heads * headDim * channels],
      ["output bias", weights.outputBias, channels],
    ];
    for (const [name, value, expected] of expectedWeights) {
      if (value.length !== expected) throw new RangeError(`${name} has ${value.length} values; expected ${expected}`);
    }
    const execution = new WebGpuExecution(this.device);
    try {
      const source = execution.upload("global-attention.source", input.activations);
      const mask = execution.upload("global-attention.mask", input.mask);
      const encoder = this.device.createCommandEncoder({ label: "global-attention" });
      this.device.pushErrorScope("validation");
      const output = await encodeGlobalAttention(execution, encoder, source, mask, {
        sequences, length, cM: channels, cZ: 1, cOuter: 1, triangleHidden: 1,
      }, weights, "global-attention");
      const readback = execution.createReadback("global-attention.readback", output, encoder);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      return {
        output: await execution.mapFloat32(readback),
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      execution.release();
    }
  }
}
