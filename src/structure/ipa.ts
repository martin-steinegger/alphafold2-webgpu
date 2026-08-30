import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "../evoformer/attention.js";
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

export interface InvariantPointAttentionInput {
  readonly activations: Float32Array;
  readonly pair: Float32Array;
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
}

export interface InvariantPointAttentionResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

const LINEAR_SHADER = createTransitionShaders({} as TransitionInput, [])[1]!;

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
  scalar_factor: f32, point_factor: f32, attention_2d_factor: f32,
  padding_0: u32, padding_1: u32, padding_2: u32, padding_3: u32, padding_4: u32,
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

const LOGITS_SHADER = `${COMMON}
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> query_scalar: array<f32>;
@group(0) @binding(1) var<storage, read> kv_scalar: array<f32>;
@group(0) @binding(2) var<storage, read> query_point: array<f32>;
@group(0) @binding(3) var<storage, read> kv_point: array<f32>;
@group(0) @binding(4) var<storage, read> pair: array<f32>;
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
  var pair_bias = weights[p.attention_2d_bias + head];
  let pair_base = (query * p.length + key_index) * p.pair_channels;
  for (var c = 0u; c < p.pair_channels; c += 1u) {
    pair_bias += pair[pair_base + c] * weights[p.attention_2d_weight + c * p.heads + head];
  }
  result += p.attention_2d_factor * pair_bias;
  result -= 1e5 * (1.0 - mask[query] * mask[key_index]);
  output[index] = result;
}`;

const SOFTMAX_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<uniform> p: Parameters;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= p.heads * p.length) { return; }
  let base = row * p.length;
  var maximum = -1e30;
  for (var k = 0u; k < p.length; k += 1u) { maximum = max(maximum, logits[base + k]); }
  var sum = 0.0;
  for (var k = 0u; k < p.length; k += 1u) { sum += exp(logits[base + k] - maximum); }
  for (var k = 0u; k < p.length; k += 1u) { output[base + k] = exp(logits[base + k] - maximum) / sum; }
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

const PAIR_FEATURE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> attention: array<f32>;
@group(0) @binding(1) var<storage, read> pair: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> features: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x;
  if (index >= p.length * p.heads * p.pair_channels) { return; }
  let channel = index % p.pair_channels;
  let head = (index / p.pair_channels) % p.heads;
  let query = index / (p.pair_channels * p.heads);
  var result = 0.0;
  for (var key_index = 0u; key_index < p.length; key_index += 1u) {
    result += attention[(head * p.length + query) * p.length + key_index]
      * pair[(query * p.length + key_index) * p.pair_channels + channel];
  }
  let offset = p.heads * p.scalar_v + 4u * p.heads * p.point_v;
  features[query * p.feature_channels + offset + head * p.pair_channels + channel] = result;
}`;

export class InvariantPointAttentionGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  constructor(device: GPUDevice) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input: InvariantPointAttentionInput): Promise<InvariantPointAttentionResult> {
    const packed = packWeights(input);
    const pipelines = await Promise.all([
      this.pipelines.get("ipa:normalize", ATTENTION_NORMALIZE_SHADER),
      this.pipelines.get("ipa:linear", LINEAR_SHADER),
      this.pipelines.get("ipa:point", POINT_SHADER),
      this.pipelines.get("ipa:logits", LOGITS_SHADER),
      this.pipelines.get("ipa:softmax", SOFTMAX_SHADER),
      this.pipelines.get("ipa:scalar-feature", SCALAR_FEATURE_SHADER),
      this.pipelines.get("ipa:point-feature", POINT_FEATURE_SHADER),
      this.pipelines.get("ipa:pair-feature", PAIR_FEATURE_SHADER),
    ]);
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    const upload = (label: string, value: ArrayBufferView, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label: string, elements: number, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    try {
      const source = upload("ipa.source", input.activations);
      const pairSource = upload("ipa.pair", input.pair);
      const mask = upload("ipa.mask", input.mask);
      const affine = upload("ipa.affine", input.affine);
      const weights = upload("ipa.weights", packed.data);
      const params = upload("ipa.parameters", parameters(input, packed.offsets), GPUBufferUsage.UNIFORM);
      const pairNormParams = upload("ipa.pair-norm-parameters", createAttentionNormParameters(
        input.length * input.length, input.pairChannels, packed.offsets[0]!, packed.offsets[1]!,
        false, 1, input.length * input.length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const pair = allocate("ipa.pair-normalized", input.length * input.length * input.pairChannels);
      const queryScalarColumns = input.heads * input.scalarQk;
      const kvScalarColumns = input.heads * (input.scalarQk + input.scalarV);
      const queryPointColumns = input.heads * 3 * input.pointQk;
      const kvPointColumns = input.heads * 3 * (input.pointQk + input.pointV);
      const queryScalar = allocate("ipa.query-scalar", input.length * queryScalarColumns);
      const kvScalar = allocate("ipa.kv-scalar", input.length * kvScalarColumns);
      const queryPointLocal = allocate("ipa.query-point-local", input.length * queryPointColumns);
      const kvPointLocal = allocate("ipa.kv-point-local", input.length * kvPointColumns);
      const queryPoint = allocate("ipa.query-point", input.length * input.heads * input.pointQk * 3);
      const kvPoint = allocate(
        "ipa.kv-point", input.length * input.heads * (input.pointQk + input.pointV) * 3,
      );
      const attentionElements = input.heads * input.length * input.length;
      const logits = allocate("ipa.logits", attentionElements);
      const attention = allocate("ipa.attention", attentionElements);
      const featureChannels = input.weights.outputWeight.length / input.channels;
      const features = allocate("ipa.features", input.length * featureChannels);
      const output = allocate("ipa.output", input.length * input.channels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const linearParams = (label: string, columns: number, weight: number, bias: number): AllocatedGpuBuffer =>
        upload(label, new Uint32Array([input.length, input.channels, columns, weight, bias, 0, 0, 0]),
          GPUBufferUsage.UNIFORM);
      const qScalarParams = linearParams("ipa.q-scalar-params", queryScalarColumns, packed.offsets[2]!, packed.offsets[3]!);
      const kvScalarParams = linearParams("ipa.kv-scalar-params", kvScalarColumns, packed.offsets[4]!, packed.offsets[5]!);
      const qPointParams = linearParams("ipa.q-point-params", queryPointColumns, packed.offsets[6]!, packed.offsets[7]!);
      const kvPointParams = linearParams("ipa.kv-point-params", kvPointColumns, packed.offsets[8]!, packed.offsets[9]!);
      const qPointTransformParams = upload("ipa.q-point-transform-params", new Uint32Array([
        input.length, input.heads, input.pointQk, 0,
      ]), GPUBufferUsage.UNIFORM);
      const kvPointTransformParams = upload("ipa.kv-point-transform-params", new Uint32Array([
        input.length, input.heads, input.pointQk + input.pointV, 0,
      ]), GPUBufferUsage.UNIFORM);
      const outputParams = upload("ipa.output-params", new Uint32Array([
        input.length, featureChannels, input.channels, packed.offsets[13]!, packed.offsets[14]!, 0, 0, 0,
      ]), GPUBufferUsage.UNIFORM);
      const encoder = this.device.createCommandEncoder({ label: "invariant-point-attention" });
      this.device.pushErrorScope("validation");
      const pass = (pipeline: GPUComputePipeline, buffers: readonly AllocatedGpuBuffer[], x: number, y = 1): void => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer: buffer.buffer } })),
        }));
        compute.dispatchWorkgroups(x, y);
        compute.end();
      };
      const grid = (elements: number, workgroupSize = 64): readonly [number, number] => {
        const groups = Math.ceil(elements / workgroupSize);
        return [Math.min(groups, 32_768), Math.ceil(groups / 32_768)];
      };
      let dispatch = grid(input.length * input.length, 1);
      pass(pipelines[0]!, [pairSource, weights, pairNormParams, pair], dispatch[0], dispatch[1]);
      const linear = (paramsValue: AllocatedGpuBuffer, result: AllocatedGpuBuffer, columns: number): void =>
        pass(pipelines[1]!, [source, weights, paramsValue, result],
          Math.ceil(columns / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      linear(qScalarParams, queryScalar, queryScalarColumns);
      linear(kvScalarParams, kvScalar, kvScalarColumns);
      linear(qPointParams, queryPointLocal, queryPointColumns);
      linear(kvPointParams, kvPointLocal, kvPointColumns);
      pass(pipelines[2]!, [queryPointLocal, affine, qPointTransformParams, queryPoint],
        Math.ceil(queryPoint.byteLength / 4 / 3 / 64));
      pass(pipelines[2]!, [kvPointLocal, affine, kvPointTransformParams, kvPoint],
        Math.ceil(kvPoint.byteLength / 4 / 3 / 64));
      dispatch = grid(attentionElements);
      pass(pipelines[3]!, [queryScalar, kvScalar, queryPoint, kvPoint, pair, mask, weights, params, logits],
        dispatch[0], dispatch[1]);
      pass(pipelines[4]!, [logits, params, attention], input.heads * input.length);
      pass(pipelines[5]!, [attention, kvScalar, params, features],
        Math.ceil(input.length * input.heads * input.scalarV / 64));
      pass(pipelines[6]!, [attention, kvPoint, affine, params, features],
        Math.ceil(input.length * input.heads * input.pointV / 64));
      pass(pipelines[7]!, [attention, pair, params, features],
        Math.ceil(input.length * input.heads * input.pairChannels / 64));
      pass(pipelines[1]!, [features, weights, outputParams, output],
        Math.ceil(input.channels / TRANSITION_TILE_COLUMNS), Math.ceil(input.length / TRANSITION_TILE_ROWS));
      const outputElements = output.byteLength / 4;
      const readback = allocate("ipa.readback", outputElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, output.byteLength);
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
    }
  }
}
