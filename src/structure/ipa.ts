import { type ActivationStorage, storageArray, storageWords, storedElement } from "../runtime/storage.js";
import { rowWindows } from "../runtime/sharded.js";
import {
  createTransitionShaders, TRANSITION_TILE_COLUMNS, TRANSITION_TILE_ROWS, type TransitionInput,
} from "../evoformer/transition.js";
import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";

export interface InvariantPointAttentionWeights {
  readonly pairNormScale: Float32Array;
  readonly pairNormOffset: Float32Array;
  readonly queryScalarWeight: Float32Array;
  readonly queryScalarBias: Float32Array;
  readonly keyValueScalarWeight: Float32Array;
  readonly keyValueScalarBias: Float32Array;
  readonly queryPointWeight: Float32Array;
  readonly queryPointBias: Float32Array;
  readonly keyValuePointWeight: Float32Array;
  readonly keyValuePointBias: Float32Array;
  readonly trainablePointWeights: Float32Array;
  readonly attention2dWeight: Float32Array;
  readonly attention2dBias: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
}

/** Native AlphaFold-Multimer IPA parameters before conversion to the fused WebGPU layout. */
export interface MultimerInvariantPointAttentionWeights {
  readonly pairNormScale: Float32Array;
  readonly pairNormOffset: Float32Array;
  readonly queryScalarWeight: Float32Array;
  readonly keyScalarWeight: Float32Array;
  readonly valueScalarWeight: Float32Array;
  readonly queryPointWeight: Float32Array;
  readonly queryPointBias: Float32Array;
  readonly keyPointWeight: Float32Array;
  readonly keyPointBias: Float32Array;
  readonly valuePointWeight: Float32Array;
  readonly valuePointBias: Float32Array;
  readonly trainablePointWeights: Float32Array;
  readonly attention2dWeight: Float32Array;
  readonly attention2dBias: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
}

function checkedLength(name: string, value: Float32Array, expected: number): void {
  if (value.length !== expected || value.byteLength !== expected * 4) {
    throw new RangeError(`${name} must contain exactly ${expected} float32 values`);
  }
}

/**
 * Convert Multimer-v3's separate, head-major Q/K/V tensors to the fused,
 * coordinate-major layout consumed by the WebGPU IPA kernels. This is a
 * lossless permutation/concatenation; no parameter values are approximated.
 */
export function adaptMultimerInvariantPointAttentionWeights(
  weights: MultimerInvariantPointAttentionWeights,
  channels: number,
  pairChannels: number,
  heads: number,
  scalarQk: number,
  scalarV: number,
  pointQk: number,
  pointV: number,
): InvariantPointAttentionWeights {
  const dimensions = [channels, pairChannels, heads, scalarQk, scalarV, pointQk, pointV];
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("multimer IPA dimensions must be positive safe integers");
  }
  checkedLength("queryScalarWeight", weights.queryScalarWeight, channels * heads * scalarQk);
  checkedLength("keyScalarWeight", weights.keyScalarWeight, channels * heads * scalarQk);
  checkedLength("valueScalarWeight", weights.valueScalarWeight, channels * heads * scalarV);
  checkedLength("queryPointWeight", weights.queryPointWeight, channels * heads * 3 * pointQk);
  checkedLength("queryPointBias", weights.queryPointBias, heads * 3 * pointQk);
  checkedLength("keyPointWeight", weights.keyPointWeight, channels * heads * 3 * pointQk);
  checkedLength("keyPointBias", weights.keyPointBias, heads * 3 * pointQk);
  checkedLength("valuePointWeight", weights.valuePointWeight, channels * heads * 3 * pointV);
  checkedLength("valuePointBias", weights.valuePointBias, heads * 3 * pointV);
  checkedLength("pairNormScale", weights.pairNormScale, pairChannels);
  checkedLength("pairNormOffset", weights.pairNormOffset, pairChannels);
  checkedLength("trainablePointWeights", weights.trainablePointWeights, heads);
  checkedLength("attention2dWeight", weights.attention2dWeight, pairChannels * heads);
  checkedLength("attention2dBias", weights.attention2dBias, heads);
  const featureChannels = heads * (scalarV + 4 * pointV + pairChannels);
  checkedLength("outputWeight", weights.outputWeight, featureChannels * channels);
  checkedLength("outputBias", weights.outputBias, channels);

  const queryScalarBias = new Float32Array(heads * scalarQk);
  const keyValueScalarWeight = new Float32Array(channels * heads * (scalarQk + scalarV));
  const keyValueScalarBias = new Float32Array(heads * (scalarQk + scalarV));
  for (let input = 0; input < channels; input += 1) {
    for (let head = 0; head < heads; head += 1) {
      const outputBase = (input * heads + head) * (scalarQk + scalarV);
      const keyBase = (input * heads + head) * scalarQk;
      const valueBase = (input * heads + head) * scalarV;
      keyValueScalarWeight.set(weights.keyScalarWeight.subarray(keyBase, keyBase + scalarQk), outputBase);
      keyValueScalarWeight.set(
        weights.valueScalarWeight.subarray(valueBase, valueBase + scalarV), outputBase + scalarQk,
      );
    }
  }

  const reorderPointProjection = (
    source: Float32Array, inputChannels: number, points: number,
  ): Float32Array => {
    const output = new Float32Array(inputChannels * heads * 3 * points);
    for (let input = 0; input < inputChannels; input += 1) {
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        for (let head = 0; head < heads; head += 1) {
          for (let point = 0; point < points; point += 1) {
            const sourceIndex = ((input * heads + head) * 3 + coordinate) * points + point;
            const outputIndex = ((input * 3 + coordinate) * heads + head) * points + point;
            output[outputIndex] = source[sourceIndex]!;
          }
        }
      }
    }
    return output;
  };
  const queryPointWeight = reorderPointProjection(weights.queryPointWeight, channels, pointQk);
  const queryPointBias = reorderPointProjection(weights.queryPointBias, 1, pointQk);
  const keyPoint = reorderPointProjection(weights.keyPointWeight, channels, pointQk);
  const valuePoint = reorderPointProjection(weights.valuePointWeight, channels, pointV);
  const keyPointBias = reorderPointProjection(weights.keyPointBias, 1, pointQk);
  const valuePointBias = reorderPointProjection(weights.valuePointBias, 1, pointV);
  const keyValuePointWeight = new Float32Array(channels * 3 * heads * (pointQk + pointV));
  const keyValuePointBias = new Float32Array(3 * heads * (pointQk + pointV));
  for (let input = 0; input < channels; input += 1) {
    for (let coordinate = 0; coordinate < 3; coordinate += 1) {
      for (let head = 0; head < heads; head += 1) {
        const outputBase = ((input * 3 + coordinate) * heads + head) * (pointQk + pointV);
        const keyBase = ((input * 3 + coordinate) * heads + head) * pointQk;
        const valueBase = ((input * 3 + coordinate) * heads + head) * pointV;
        keyValuePointWeight.set(keyPoint.subarray(keyBase, keyBase + pointQk), outputBase);
        keyValuePointWeight.set(valuePoint.subarray(valueBase, valueBase + pointV), outputBase + pointQk);
      }
    }
  }
  for (let coordinate = 0; coordinate < 3; coordinate += 1) {
    for (let head = 0; head < heads; head += 1) {
      const outputBase = (coordinate * heads + head) * (pointQk + pointV);
      const keyBase = (coordinate * heads + head) * pointQk;
      const valueBase = (coordinate * heads + head) * pointV;
      keyValuePointBias.set(keyPointBias.subarray(keyBase, keyBase + pointQk), outputBase);
      keyValuePointBias.set(valuePointBias.subarray(valueBase, valueBase + pointV), outputBase + pointQk);
    }
  }
  return {
    pairNormScale: weights.pairNormScale,
    pairNormOffset: weights.pairNormOffset,
    queryScalarWeight: weights.queryScalarWeight,
    queryScalarBias,
    keyValueScalarWeight,
    keyValueScalarBias,
    queryPointWeight,
    queryPointBias,
    keyValuePointWeight,
    keyValuePointBias,
    trainablePointWeights: weights.trainablePointWeights,
    attention2dWeight: weights.attention2dWeight,
    attention2dBias: weights.attention2dBias,
    outputWeight: weights.outputWeight,
    outputBias: weights.outputBias,
  };
}

export interface InvariantPointAttentionInput {
  readonly activations: Float32Array;
  readonly pair: Float32Array;
  /** Optional device-resident pair tensor, avoiding a second pair-sized upload. */
  readonly pairBuffer?: GPUBuffer;
  /** Storage of that tensor; `f16` means packed half-precision words. */
  readonly pairStorage?: ActivationStorage;
  /**
   * Bytes one binding may cover of the pair, at most the device's own limit.
   *
   * Only tests set it: it makes a short prediction take the windowed path a
   * long one takes on a device with a small binding limit.
   */
  readonly bindingLimitBytes?: number;
  readonly mask: Float32Array;
  readonly affine: Float32Array;
  readonly length: number;
  readonly channels: number;
  readonly pairChannels: number;
  readonly heads: number;
  readonly scalarQk: number;
  readonly scalarV: number;
  readonly pointQk: number;
  readonly pointV: number;
  readonly weights: InvariantPointAttentionWeights;
  /** Apply the Multimer-v3 convention that scales the mask with all three logit terms. */
  readonly multimer?: boolean;
  readonly prepared?: PreparedInvariantPointAttention;
}

export interface InvariantPointAttentionResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

const LINEAR_SHADER = createTransitionShaders({} as TransitionInput, [])[1]!;

function validateInput(input: InvariantPointAttentionInput): void {
  const dimensions = [input.length, input.channels, input.pairChannels, input.heads,
    input.scalarQk, input.scalarV, input.pointQk, input.pointV];
  if (!dimensions.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("IPA dimensions must be positive safe integers");
  }
  const featureChannels = input.heads * (input.scalarV + 4 * input.pointV + input.pairChannels);
  const expected = [
    ["activations", input.activations, input.length * input.channels],
    ["mask", input.mask, input.length],
    ["affine", input.affine, input.length * 7],
    ["pairNormScale", input.weights.pairNormScale, input.pairChannels],
    ["pairNormOffset", input.weights.pairNormOffset, input.pairChannels],
    ["queryScalarWeight", input.weights.queryScalarWeight, input.channels * input.heads * input.scalarQk],
    ["queryScalarBias", input.weights.queryScalarBias, input.heads * input.scalarQk],
    ["keyValueScalarWeight", input.weights.keyValueScalarWeight,
      input.channels * input.heads * (input.scalarQk + input.scalarV)],
    ["keyValueScalarBias", input.weights.keyValueScalarBias, input.heads * (input.scalarQk + input.scalarV)],
    ["queryPointWeight", input.weights.queryPointWeight, input.channels * input.heads * 3 * input.pointQk],
    ["queryPointBias", input.weights.queryPointBias, input.heads * 3 * input.pointQk],
    ["keyValuePointWeight", input.weights.keyValuePointWeight,
      input.channels * input.heads * 3 * (input.pointQk + input.pointV)],
    ["keyValuePointBias", input.weights.keyValuePointBias, input.heads * 3 * (input.pointQk + input.pointV)],
    ["trainablePointWeights", input.weights.trainablePointWeights, input.heads],
    ["attention2dWeight", input.weights.attention2dWeight, input.pairChannels * input.heads],
    ["attention2dBias", input.weights.attention2dBias, input.heads],
    ["outputWeight", input.weights.outputWeight, featureChannels * input.channels],
    ["outputBias", input.weights.outputBias, input.channels],
  ] as const;
  for (const [name, value, elements] of expected) {
    if (!(value instanceof Float32Array) || value.length !== elements || value.byteLength !== elements * 4) {
      throw new RangeError(`${name} must contain exactly ${elements} float32 values`);
    }
  }
  if (input.pairBuffer === undefined) {
    if (input.pairStorage === "f16") {
      throw new RangeError("a packed pair must be passed as a device-resident buffer");
    }
    checkedLength("pair", input.pair, input.length * input.length * input.pairChannels);
  } else if (input.pair.length !== 0
    && input.pair.length !== input.length * input.length * input.pairChannels) {
    throw new RangeError("pair must be empty or match the device-resident pair tensor");
  }
}

function packWeights(input: InvariantPointAttentionInput): { data: Float32Array; offsets: readonly number[] } {
  const w = input.weights;
  const values = [
    w.pairNormScale, w.pairNormOffset,
    w.queryScalarWeight, w.queryScalarBias,
    w.keyValueScalarWeight, w.keyValueScalarBias,
    w.queryPointWeight, w.queryPointBias,
    w.keyValuePointWeight, w.keyValuePointBias,
    w.trainablePointWeights,
    w.attention2dWeight, w.attention2dBias,
    w.outputWeight, w.outputBias,
  ] as const;
  const offsets: number[] = [];
  let size = 0;
  for (const value of values) { offsets.push(size); size += value.length; }
  const data = new Float32Array(size);
  values.forEach((value, index) => data.set(value, offsets[index]));
  return { data, offsets };
}

function parameters(input: InvariantPointAttentionInput, offsets: readonly number[]): Uint8Array {
  const buffer = new ArrayBuffer(128);
  const view = new DataView(buffer);
  const featureChannels = input.heads * input.scalarV
    + 4 * input.heads * input.pointV + input.heads * input.pairChannels;
  const integers = [
    input.length, input.channels, input.pairChannels, input.heads,
    input.scalarQk, input.scalarV, input.pointQk, input.pointV, featureChannels,
    ...offsets,
  ];
  integers.forEach((value, index) => view.setUint32(index * 4, value!, true));
  view.setFloat32(96, Math.sqrt(1 / (3 * input.scalarQk)), true);
  view.setFloat32(100, Math.sqrt(1 / (3 * input.pointQk * 4.5)), true);
  view.setFloat32(104, Math.sqrt(1 / 3), true);
  view.setFloat32(108, input.multimer === true ? Math.sqrt(1 / 3) : 1, true);
  return new Uint8Array(buffer);
}

const COMMON = `
struct Parameters {
  length: u32, channels: u32, pair_channels: u32, heads: u32,
  scalar_qk: u32, scalar_v: u32, point_qk: u32, point_v: u32, feature_channels: u32,
  pair_norm_scale: u32, pair_norm_offset: u32,
  query_scalar_weight: u32, query_scalar_bias: u32,
  kv_scalar_weight: u32, kv_scalar_bias: u32,
  query_point_weight: u32, query_point_bias: u32,
  kv_point_weight: u32, kv_point_bias: u32,
  trainable_point_weights: u32,
  attention_2d_weight: u32, attention_2d_bias: u32,
  output_weight: u32, output_bias: u32,
  scalar_factor: f32, point_factor: f32, attention_2d_factor: f32, mask_factor: f32,
  padding_0: u32, padding_1: u32, padding_2: u32, padding_3: u32,
};
`;

const POINT_SHADER = `
struct PointParameters { length: u32, heads: u32, points: u32, padding: u32 };
@group(0) @binding(0) var<storage, read> local_points: array<f32>;
@group(0) @binding(1) var<storage, read> affine: array<f32>;
@group(0) @binding(2) var<uniform> p: PointParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let u = q.yzw;
  return 2.0 * dot(u, v) * u + (q.x * q.x - dot(u, u)) * v + 2.0 * q.x * cross(u, v);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= p.length * p.heads * p.points) { return; }
  let point = index % p.points;
  let head = (index / p.points) % p.heads;
  let residue = index / (p.points * p.heads);
  let plane = p.heads * p.points;
  let base = residue * 3u * plane + head * p.points + point;
  let local = vec3<f32>(local_points[base], local_points[base + plane], local_points[base + 2u * plane]);
  let affine_base = residue * 7u;
  let q = vec4<f32>(affine[affine_base], affine[affine_base + 1u], affine[affine_base + 2u], affine[affine_base + 3u]);
  let translation = vec3<f32>(affine[affine_base + 4u], affine[affine_base + 5u], affine[affine_base + 6u]);
  let global = rotate(q, local) + translation;
  let output_base = index * 3u;
  output[output_base] = global.x;
  output[output_base + 1u] = global.y;
  output[output_base + 2u] = global.z;
}`;

/**
 * Mean and inverse standard deviation of every pair row.
 *
 * The IPA's two pair consumers used to read a normalized copy of the pair,
 * which is another pair-sized tensor and at 512 residues the whole structure
 * phase's peak. From these statistics both can normalize while loading, the
 * way the triangle multiplication does, for two floats a row instead of 128.
 * The reduction is the one the LayerNorm kernel ran, so the values it feeds
 * consumers are the ones they read before.
 */
function createPairStatisticsShader(storage: ActivationStorage): string {
  return `${COMMON}
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> pair: array<${storageArray(storage)}>;
@group(0) @binding(1) var<uniform> p: Parameters;
@group(0) @binding(2) var<storage, read_write> statistics: array<f32>;
// The pair rows this dispatch covers: x is the first and y how many. The pair
// is bound at that row, since a whole pair is past what one binding reaches;
// the statistics are two floats a row and stay whole.
@group(0) @binding(3) var<uniform> w: vec4<u32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= w.y) { return; }
  let base = row * p.pair_channels;
  var sum = 0.0;
  for (var c = local.x; c < p.pair_channels; c += 64u) { sum += ${storedElement(storage, "pair", "base + c")}; }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(p.pair_channels); }
  workgroupBarrier();
  var squared = 0.0;
  for (var c = local.x; c < p.pair_channels; c += 64u) {
    let centered = ${storedElement(storage, "pair", "base + c")} - row_mean[0];
    squared += centered * centered;
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    statistics[2u * (w.x + row)] = row_mean[0];
    statistics[2u * (w.x + row) + 1u] = inverseSqrt(partial[0] / f32(p.pair_channels) + 1e-5);
  }
}`;
}

/**
 * The attention bias each head takes from the pair, projected once.
 *
 * It depends only on the pair, so the eight structure iterations share it, and
 * the logits kernel reads one number per residue pair instead of walking 128
 * pair channels every iteration.
 */
function createPairBiasShader(storage: ActivationStorage): string {
  return `${COMMON}
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> pair: array<${storageArray(storage)}>;
@group(0) @binding(1) var<storage, read> statistics: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> bias: array<f32>;
// x is the first pair row this dispatch covers and y how many; the pair is
// bound at that row, the statistics and the bias whole.
@group(0) @binding(5) var<uniform> w: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.heads * w.y) { return; }
  let row = index % w.y;
  let head = index / w.y;
  let mean = statistics[2u * (w.x + row)];
  let inverse_std = statistics[2u * (w.x + row) + 1u];
  let base = row * p.pair_channels;
  var result = weights[p.attention_2d_bias + head];
  for (var c = 0u; c < p.pair_channels; c += 1u) {
    let normalized = (${storedElement(storage, "pair", "base + c")} - mean) * inverse_std
      * weights[p.pair_norm_scale + c] + weights[p.pair_norm_offset + c];
    result += normalized * weights[p.attention_2d_weight + c * p.heads + head];
  }
  bias[head * p.length * p.length + w.x + row] = result;
}`;
}

const LOGITS_SHADER = `${COMMON}
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> query_scalar: array<f32>;
@group(0) @binding(1) var<storage, read> kv_scalar: array<f32>;
@group(0) @binding(2) var<storage, read> query_point: array<f32>;
@group(0) @binding(3) var<storage, read> kv_point: array<f32>;
@group(0) @binding(4) var<storage, read> pair_bias: array<f32>;
@group(0) @binding(5) var<storage, read> mask: array<f32>;
@group(0) @binding(6) var<storage, read> weights: array<f32>;
@group(0) @binding(7) var<uniform> p: Parameters;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.heads * p.length * p.length) { return; }
  let key_index = index % p.length;
  let query = (index / p.length) % p.length;
  let head = index / (p.length * p.length);
  var result = 0.0;
  let q_scalar_base = (query * p.heads + head) * p.scalar_qk;
  let kv_scalar_stride = p.scalar_qk + p.scalar_v;
  let k_scalar_base = (key_index * p.heads + head) * kv_scalar_stride;
  for (var c = 0u; c < p.scalar_qk; c += 1u) {
    result += p.scalar_factor * query_scalar[q_scalar_base + c] * kv_scalar[k_scalar_base + c];
  }
  var distance = 0.0;
  let q_point_base = (query * p.heads + head) * p.point_qk * 3u;
  let kv_points = p.point_qk + p.point_v;
  let k_point_base = (key_index * p.heads + head) * kv_points * 3u;
  for (var point = 0u; point < p.point_qk; point += 1u) {
    for (var coordinate = 0u; coordinate < 3u; coordinate += 1u) {
      let delta = query_point[q_point_base + point * 3u + coordinate]
        - kv_point[k_point_base + point * 3u + coordinate];
      distance += delta * delta;
    }
  }
  let point_weight = p.point_factor * log(1.0 + exp(weights[p.trainable_point_weights + head]));
  result -= 0.5 * point_weight * distance;
  result += p.attention_2d_factor * pair_bias[index];
  result -= p.mask_factor * 1e5 * (1.0 - mask[query] * mask[key_index]);
  output[index] = result;
}`;

const SOFTMAX_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<uniform> p: Parameters;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= p.heads * p.length) { return; }
  let base = row * p.length;
  var maximum = -1e30;
  for (var k = 0u; k < p.length; k += 1u) { maximum = max(maximum, logits[base + k]); }
  var sum = 0.0;
  for (var k = 0u; k < p.length; k += 1u) { sum += exp(logits[base + k] - maximum); }
  for (var k = 0u; k < p.length; k += 1u) { logits[base + k] = exp(logits[base + k] - maximum) / sum; }
}`;

const SCALAR_FEATURE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> kv_scalar: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> features: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= p.length * p.heads * p.scalar_v) { return; }
  let value_channel = index % p.scalar_v;
  let head = (index / p.scalar_v) % p.heads;
  let query = index / (p.scalar_v * p.heads);
  var result = 0.0;
  for (var key_index = 0u; key_index < p.length; key_index += 1u) {
    let a = attention[(head * p.length + query) * p.length + key_index];
    let kv_base = (key_index * p.heads + head) * (p.scalar_qk + p.scalar_v);
    result += a * kv_scalar[kv_base + p.scalar_qk + value_channel];
  }
  features[query * p.feature_channels + head * p.scalar_v + value_channel] = result;
}`;

const POINT_FEATURE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> kv_point: array<f32>;
@group(0) @binding(2) var<storage, read> affine: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> features: array<f32>;

fn inverse_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  return 2.0 * dot(q.yzw, v) * q.yzw + (q.x * q.x - dot(q.yzw, q.yzw)) * v - 2.0 * q.x * cross(q.yzw, v);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= p.length * p.heads * p.point_v) { return; }
  let point = index % p.point_v;
  let head = (index / p.point_v) % p.heads;
  let query = index / (p.point_v * p.heads);
  var global = vec3<f32>(0.0);
  let kv_points = p.point_qk + p.point_v;
  for (var key_index = 0u; key_index < p.length; key_index += 1u) {
    let a = attention[(head * p.length + query) * p.length + key_index];
    let point_base = ((key_index * p.heads + head) * kv_points + p.point_qk + point) * 3u;
    global += a * vec3<f32>(kv_point[point_base], kv_point[point_base + 1u], kv_point[point_base + 2u]);
  }
  let affine_base = query * 7u;
  let q = vec4<f32>(affine[affine_base], affine[affine_base + 1u], affine[affine_base + 2u], affine[affine_base + 3u]);
  let translation = vec3<f32>(affine[affine_base + 4u], affine[affine_base + 5u], affine[affine_base + 6u]);
  let local = inverse_rotate(q, global - translation);
  let scalar_size = p.heads * p.scalar_v;
  let point_size = p.heads * p.point_v;
  let point_index = head * p.point_v + point;
  let base = query * p.feature_channels;
  features[base + scalar_size + point_index] = local.x;
  features[base + scalar_size + point_size + point_index] = local.y;
  features[base + scalar_size + 2u * point_size + point_index] = local.z;
  features[base + scalar_size + 3u * point_size + point_index] = sqrt(1e-8 + dot(local, local));
}`;

function createPairFeatureShader(storage: ActivationStorage): string {
  return `${COMMON}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> pair: array<${storageArray(storage)}>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> features: array<f32>;
@group(0) @binding(4) var<storage, read> statistics: array<f32>;
@group(0) @binding(5) var<storage, read> weights: array<f32>;
// x is the first query this dispatch covers and y how many; every pair row it
// reads belongs to those queries, so the pair is bound at their first row and
// the statistics, which are small, stay whole.
@group(0) @binding(6) var<uniform> w: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= w.y * p.heads * p.pair_channels) { return; }
  let channel = index % p.pair_channels;
  let head = (index / p.pair_channels) % p.heads;
  let local_query = index / (p.pair_channels * p.heads);
  let query = w.x + local_query;
  var result = 0.0;
  let norm_scale = weights[p.pair_norm_scale + channel];
  let norm_offset = weights[p.pair_norm_offset + channel];
  for (var key_index = 0u; key_index < p.length; key_index += 1u) {
    let row = local_query * p.length + key_index;
    let global_row = w.x * p.length + row;
    let normalized = (${storedElement(storage, "pair", "row * p.pair_channels + channel")}
      - statistics[2u * global_row]) * statistics[2u * global_row + 1u] * norm_scale + norm_offset;
    result += attention[(head * p.length + query) * p.length + key_index] * normalized;
  }
  let offset = p.heads * p.scalar_v + 4u * p.heads * p.point_v;
  features[query * p.feature_channels + offset + head * p.pair_channels + channel] = result;
}`;
}

const GRID_WIDTH = 32_768;

/** A run of queries whose pair rows one binding can cover. */
export interface IpaQueryWindow {
  readonly offset: number;
  readonly count: number;
  /** The offset and count as a uniform, for the shader. */
  readonly bounds: GPUBuffer;
}

interface PreparedFields {
  readonly pipelines: readonly GPUComputePipeline[];
  readonly weights: AllocatedGpuBuffer;
  readonly params: AllocatedGpuBuffer;
  readonly mask: AllocatedGpuBuffer;
  /** Mean and inverse standard deviation of each pair row. */
  readonly statistics: AllocatedGpuBuffer;
  /** The pair's contribution to the attention logits, one value a head and pair. */
  readonly pairBias: AllocatedGpuBuffer;
  /** The raw pair every iteration normalizes while loading. */
  readonly pair: GPUBuffer;
  /** Query windows the pair feature walks, each inside one binding. */
  readonly queryWindows: readonly IpaQueryWindow[];
  /** How the pair is stored, which sizes every window of it. */
  readonly pairStorage: ActivationStorage;
  readonly queryScalarColumns: number;
  readonly kvScalarColumns: number;
  readonly queryPointColumns: number;
  readonly kvPointColumns: number;
  readonly featureChannels: number;
  readonly qScalarParams: AllocatedGpuBuffer;
  readonly kvScalarParams: AllocatedGpuBuffer;
  readonly qPointParams: AllocatedGpuBuffer;
  readonly kvPointParams: AllocatedGpuBuffer;
  readonly qPointTransformParams: AllocatedGpuBuffer;
  readonly kvPointTransformParams: AllocatedGpuBuffer;
  readonly outputParams: AllocatedGpuBuffer;
}

/** Device-resident IPA state shared by the eight structure iterations. */
export class PreparedInvariantPointAttention implements PreparedFields {
  readonly pipelines!: readonly GPUComputePipeline[];
  readonly weights!: AllocatedGpuBuffer;
  readonly params!: AllocatedGpuBuffer;
  readonly mask!: AllocatedGpuBuffer;
  readonly statistics!: AllocatedGpuBuffer;
  readonly pairBias!: AllocatedGpuBuffer;
  readonly pair!: GPUBuffer;
  readonly queryWindows!: readonly IpaQueryWindow[];
  readonly pairStorage!: ActivationStorage;
  readonly queryScalarColumns!: number;
  readonly kvScalarColumns!: number;
  readonly queryPointColumns!: number;
  readonly kvPointColumns!: number;
  readonly featureChannels!: number;
  readonly qScalarParams!: AllocatedGpuBuffer;
  readonly kvScalarParams!: AllocatedGpuBuffer;
  readonly qPointParams!: AllocatedGpuBuffer;
  readonly kvPointParams!: AllocatedGpuBuffer;
  readonly qPointTransformParams!: AllocatedGpuBuffer;
  readonly kvPointTransformParams!: AllocatedGpuBuffer;
  readonly outputParams!: AllocatedGpuBuffer;
  readonly #device: GPUDevice;
  readonly #shape: readonly number[];
  readonly #pairSource: Float32Array;
  readonly #pairBufferSource: GPUBuffer | undefined;
  readonly #maskSource: Float32Array;
  readonly #weightSource: InvariantPointAttentionWeights;
  readonly #allocations: AllocatedGpuBuffer[];
  #released = false;

  constructor(device: GPUDevice, input: InvariantPointAttentionInput, fields: PreparedFields,
    allocations: AllocatedGpuBuffer[]) {
    this.#device = device;
    this.#shape = [input.length, input.channels, input.pairChannels, input.heads,
      input.scalarQk, input.scalarV, input.pointQk, input.pointV, input.multimer === true ? 1 : 0];
    this.#pairSource = input.pair;
    this.#pairBufferSource = input.pairBuffer;
    this.#maskSource = input.mask;
    this.#weightSource = input.weights;
    this.#allocations = allocations;
    Object.assign(this, fields);
  }

  assertCompatible(device: GPUDevice, input: InvariantPointAttentionInput): void {
    if (this.#released) throw new Error("prepared invariant point attention state has been released");
    const shape = [input.length, input.channels, input.pairChannels, input.heads,
      input.scalarQk, input.scalarV, input.pointQk, input.pointV, input.multimer === true ? 1 : 0];
    if (device !== this.#device || shape.some((value, index) => value !== this.#shape[index])
      || input.pair !== this.#pairSource || input.pairBuffer !== this.#pairBufferSource
      || input.mask !== this.#maskSource || input.weights !== this.#weightSource) {
      throw new Error("prepared invariant point attention state does not match this input");
    }
  }

  release(): void {
    if (this.#released) return;
    this.#released = true;
    for (let index = this.#allocations.length - 1; index >= 0; index -= 1) {
      this.#allocations[index]!.release();
    }
    this.#allocations.length = 0;
  }
}

export class InvariantPointAttentionGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  constructor(device: GPUDevice, allocator = new GpuBufferAllocator(device)) {
    this.device = device;
    this.allocator = allocator;
    this.pipelines = pipelineCacheForDevice(device);
  }

  async prepare(input: InvariantPointAttentionInput): Promise<PreparedInvariantPointAttention> {
    validateInput(input);
    const pairStorage = input.pairStorage ?? "f32";
    const bindingLimit = Math.min(this.device.limits.maxStorageBufferBindingSize,
      input.bindingLimitBytes ?? Number.MAX_SAFE_INTEGER);
    const pipelines = await Promise.all([
      this.pipelines.get(`ipa:pair-statistics:${pairStorage}`, createPairStatisticsShader(pairStorage)),
      this.pipelines.get("ipa:linear", LINEAR_SHADER),
      this.pipelines.get("ipa:point", POINT_SHADER),
      this.pipelines.get("ipa:logits", LOGITS_SHADER),
      this.pipelines.get("ipa:softmax", SOFTMAX_SHADER),
      this.pipelines.get("ipa:scalar-feature", SCALAR_FEATURE_SHADER),
      this.pipelines.get("ipa:point-feature", POINT_FEATURE_SHADER),
      this.pipelines.get(`ipa:pair-feature:${pairStorage}`, createPairFeatureShader(pairStorage)),
      this.pipelines.get(`ipa:pair-bias:${pairStorage}`, createPairBiasShader(pairStorage)),
    ]);
    const packed = packWeights(input);
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    const upload = (label: string, value: ArrayBufferView, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label: string, elements: number, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    const queryScalarColumns = input.heads * input.scalarQk;
    const kvScalarColumns = input.heads * (input.scalarQk + input.scalarV);
    const queryPointColumns = input.heads * 3 * input.pointQk;
    const kvPointColumns = input.heads * 3 * (input.pointQk + input.pointV);
    const featureChannels = input.weights.outputWeight.length / input.channels;
    let prepared: PreparedInvariantPointAttention | undefined;
    try {
      const weights = upload("ipa.weights", packed.data);
      const params = upload("ipa.parameters", parameters(input, packed.offsets), GPUBufferUsage.UNIFORM);
      const mask = upload("ipa.mask", input.mask);
      // Two floats a pair row, and one bias a head and residue pair, in place
      // of a whole normalized pair.
      const statistics = allocate("ipa.pair-statistics", input.length * input.length * 2);
      const pairBias = allocate("ipa.pair-bias", input.heads * input.length * input.length);
      const linearParams = (label: string, columns: number, weight: number, bias: number): AllocatedGpuBuffer =>
        upload(label, new Uint32Array([input.length, input.channels, columns, weight, bias, 0, 0, 0]),
          GPUBufferUsage.UNIFORM);
      const uploadedPair = input.pairBuffer === undefined
        ? keep(this.allocator.upload("ipa.pair", input.pair, GPUBufferUsage.STORAGE)) : undefined;
      // One window of queries at a time keeps the pair rows a feature pass
      // reads inside a single binding.
      const queryWindows = rowWindows(input.length, bindingLimit,
        [storageWords(input.length * input.pairChannels, pairStorage) * 4]).map((window) => ({
        offset: window.offset, count: window.count,
        bounds: upload(`ipa.query-window-${window.offset}`,
          new Uint32Array([window.offset, window.count, 0, 0]), GPUBufferUsage.UNIFORM).buffer,
      }));
      prepared = new PreparedInvariantPointAttention(this.device, input, {
        pipelines, weights, params, mask, statistics, pairBias, queryWindows, pairStorage,
        pair: input.pairBuffer ?? uploadedPair!.buffer,
        queryScalarColumns, kvScalarColumns, queryPointColumns, kvPointColumns, featureChannels,
        qScalarParams: linearParams("ipa.q-scalar-params", queryScalarColumns, packed.offsets[2]!, packed.offsets[3]!),
        kvScalarParams: linearParams("ipa.kv-scalar-params", kvScalarColumns, packed.offsets[4]!, packed.offsets[5]!),
        qPointParams: linearParams("ipa.q-point-params", queryPointColumns, packed.offsets[6]!, packed.offsets[7]!),
        kvPointParams: linearParams("ipa.kv-point-params", kvPointColumns, packed.offsets[8]!, packed.offsets[9]!),
        qPointTransformParams: upload("ipa.q-point-transform-params", new Uint32Array([
          input.length, input.heads, input.pointQk, 0,
        ]), GPUBufferUsage.UNIFORM),
        kvPointTransformParams: upload("ipa.kv-point-transform-params", new Uint32Array([
          input.length, input.heads, input.pointQk + input.pointV, 0,
        ]), GPUBufferUsage.UNIFORM),
        outputParams: upload("ipa.output-params", new Uint32Array([
          input.length, featureChannels, input.channels, packed.offsets[13]!, packed.offsets[14]!, 0, 0, 0,
        ]), GPUBufferUsage.UNIFORM),
      }, allocations);
      {
        const pairBuffer = prepared.pair;
        const encoder = this.device.createCommandEncoder({ label: "ipa.prepare" });
        const compute = encoder.beginComputePass();
        const rows = input.length * input.length;
        const bind = (pipeline: GPUComputePipeline, buffers: readonly GPUBindingResource[],
          x: number, y = 1): void => {
          compute.setPipeline(pipeline);
          compute.setBindGroup(0, this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((resource, binding) => ({ binding, resource })),
          }));
          compute.dispatchWorkgroups(x, y);
        };
        // Both passes read the pair by row, and a whole pair is past what one
        // binding may cover, so they walk it a window of rows at a time.
        for (const window of rowWindows(rows, bindingLimit,
          [storageWords(input.pairChannels, pairStorage) * 4])) {
          const bounds = upload(`ipa.pair-window-${window.offset}`,
            new Uint32Array([window.offset, window.count, 0, 0]), GPUBufferUsage.UNIFORM);
          const pairWindow = {
            buffer: pairBuffer,
            offset: storageWords(window.offset * input.pairChannels, pairStorage) * 4,
            size: storageWords(window.count * input.pairChannels, pairStorage) * 4,
          };
          const statisticsWindow = { buffer: statistics.buffer };
          bind(pipelines[0]!, [pairWindow, { buffer: params.buffer }, statisticsWindow, { buffer: bounds.buffer }],
            Math.min(window.count, GRID_WIDTH), Math.ceil(window.count / GRID_WIDTH));
          // The bias holds for every structure iteration, so it is projected once.
          const biasGroups = Math.ceil(input.heads * window.count / 64);
          bind(pipelines[8]!, [pairWindow, statisticsWindow, { buffer: weights.buffer },
            { buffer: params.buffer }, { buffer: pairBias.buffer }, { buffer: bounds.buffer }],
            Math.min(biasGroups, GRID_WIDTH), Math.ceil(biasGroups / GRID_WIDTH));
        }
        compute.end();
        this.device.pushErrorScope("validation");
        this.device.queue.submit([encoder.finish()]);
        // Deliberately not waiting for completion. The statistics and the
        // bias are only read by dispatches submitted after this one, which the
        // queue orders behind it, and WebGPU keeps a destroyed buffer's memory
        // alive until the work already submitted against it has finished.
        const error = await this.device.popErrorScope();
        if (error !== null) throw new Error(`WebGPU IPA preparation failed: ${error.message}`);
      }
      return prepared;
    } catch (error) {
      if (prepared === undefined) {
        for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index]!.release();
      } else {
        prepared.release();
      }
      throw error;
    }
  }

  /**
   * Encodes one attention pass over GPU-resident activations.
   *
   * The eight structure iterations differ only in their activations and frame,
   * so keeping them on the device lets the whole loop share one command buffer
   * instead of a submission, a fence and a readback per iteration. Scratch is
   * returned so the caller can retire it after its final encoded use. A pooled
   * allocator may then safely alias it in later dispatches in the same pass;
   * command ordering preserves the earlier reads and writes.
   */
  encode(
    pass: GPUComputePassEncoder,
    input: InvariantPointAttentionInput,
    shared: PreparedInvariantPointAttention,
    source: AllocatedGpuBuffer,
    affine: AllocatedGpuBuffer,
  ): { readonly output: AllocatedGpuBuffer; readonly scratch: readonly AllocatedGpuBuffer[] } {
    const {
      pipelines, weights, params, mask, pair, featureChannels,
      queryScalarColumns, kvScalarColumns, queryPointColumns, kvPointColumns,
    } = shared;
    const scratch: AllocatedGpuBuffer[] = [];
    const allocate = (label: string, elements: number, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer => {
      const buffer = this.allocator.allocate(label, elements * 4, usage);
      scratch.push(buffer);
      return buffer;
    };
    const queryScalar = allocate("ipa.query-scalar", input.length * queryScalarColumns);
    const kvScalar = allocate("ipa.kv-scalar", input.length * kvScalarColumns);
    const queryPointLocal = allocate("ipa.query-point-local", input.length * queryPointColumns);
    const kvPointLocal = allocate("ipa.kv-point-local", input.length * kvPointColumns);
    const queryPoint = allocate("ipa.query-point", input.length * input.heads * input.pointQk * 3);
    const kvPoint = allocate("ipa.kv-point", input.length * input.heads * (input.pointQk + input.pointV) * 3);
    const attentionElements = input.heads * input.length * input.length;
    const logits = allocate("ipa.logits", attentionElements);
    const features = allocate("ipa.features", input.length * featureChannels);
    const output = this.allocator.allocate("ipa.output", input.length * input.channels * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    // One pass for the whole structure loop: a pass boundary per dispatch cost
    // more than the dispatches themselves at short chain lengths.
    const dispatch = (pipeline: GPUComputePipeline,
      buffers: readonly (AllocatedGpuBuffer | GPUBuffer | GPUBufferBinding)[], x: number, y = 1): void => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: buffers.map((value, binding) => ({
          binding,
          resource: value instanceof GPUBuffer
            ? { buffer: value }
            : ("buffer" in value && value.buffer instanceof GPUBuffer
              ? value as GPUBufferBinding : { buffer: (value as AllocatedGpuBuffer).buffer }),
        })),
      }));
      pass.dispatchWorkgroups(x, y);
    };
    const gridFor = (elements: number, workgroupSize = 64): readonly [number, number] => {
      const groups = Math.ceil(elements / workgroupSize);
      return [Math.min(groups, 32_768), Math.ceil(groups / 32_768)];
    };
    const linear = (paramsValue: AllocatedGpuBuffer, result: AllocatedGpuBuffer, columns: number): void =>
      dispatch(pipelines[1]!, [source, weights, paramsValue, result],
        Math.ceil(columns / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
    linear(shared.qScalarParams, queryScalar, queryScalarColumns);
    linear(shared.kvScalarParams, kvScalar, kvScalarColumns);
    linear(shared.qPointParams, queryPointLocal, queryPointColumns);
    linear(shared.kvPointParams, kvPointLocal, kvPointColumns);
    dispatch(pipelines[2]!, [queryPointLocal, affine, shared.qPointTransformParams, queryPoint],
      Math.ceil(queryPoint.byteLength / 4 / 3 / 64));
    dispatch(pipelines[2]!, [kvPointLocal, affine, shared.kvPointTransformParams, kvPoint],
      Math.ceil(kvPoint.byteLength / 4 / 3 / 64));
    const attentionGrid = gridFor(attentionElements);
    dispatch(pipelines[3]!,
      [queryScalar, kvScalar, queryPoint, kvPoint, shared.pairBias, mask, weights, params, logits],
      attentionGrid[0], attentionGrid[1]);
    dispatch(pipelines[4]!, [logits, params], input.heads * input.length);
    dispatch(pipelines[5]!, [logits, kvScalar, params, features],
      Math.ceil(input.length * input.heads * input.scalarV / 64));
    dispatch(pipelines[6]!, [logits, kvPoint, affine, params, features],
      Math.ceil(input.length * input.heads * input.pointV / 64));
    // Every pair row this reads belongs to the query it is accumulating, so a
    // window of queries needs only their rows: a whole pair is past what one
    // binding may cover at long chain lengths.
    for (const window of shared.queryWindows) {
      dispatch(pipelines[7]!, [logits, {
        buffer: pair,
        offset: storageWords(window.offset * input.length * input.pairChannels, shared.pairStorage) * 4,
        size: storageWords(window.count * input.length * input.pairChannels, shared.pairStorage) * 4,
      }, params, features, shared.statistics, weights, window.bounds],
        Math.ceil(window.count * input.heads * input.pairChannels / 64));
    }
    dispatch(pipelines[1]!, [features, weights, shared.outputParams, output],
      Math.ceil(input.channels / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
    return { output, scratch };
  }

  async run(input: InvariantPointAttentionInput): Promise<InvariantPointAttentionResult> {
    validateInput(input);
    const shared = input.prepared ?? await this.prepare(input);
    const ownsShared = input.prepared === undefined;
    shared.assertCompatible(this.device, input);
    const allocations: AllocatedGpuBuffer[] = [];
    try {
      const source = this.allocator.upload("ipa.source", input.activations, GPUBufferUsage.STORAGE);
      allocations.push(source);
      const affine = this.allocator.upload("ipa.affine", input.affine, GPUBufferUsage.STORAGE);
      allocations.push(affine);
      const encoder = this.device.createCommandEncoder({ label: "invariant-point-attention" });
      this.device.pushErrorScope("validation");
      const pass = encoder.beginComputePass({ label: "invariant-point-attention" });
      const encoded = this.encode(pass, input, shared, source, affine);
      pass.end();
      allocations.push(...encoded.scratch, encoded.output);
      const outputElements = encoded.output.byteLength / 4;
      const readback = this.allocator.allocate("ipa.readback", outputElements * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      allocations.push(readback);
      encoder.copyBufferToBuffer(encoded.output.buffer, 0, readback.buffer, 0, encoded.output.byteLength);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU validation failed: ${error.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return { output: result, elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot() };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index]!.release();
      if (ownsShared) shared.release();
    }
  }
}
