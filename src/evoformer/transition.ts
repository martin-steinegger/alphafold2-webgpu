import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";

export interface TransitionWeights {
  readonly layerNormScale: Float32Array;
  readonly layerNormOffset: Float32Array;
  readonly firstWeight: Float32Array;
  readonly firstBias: Float32Array;
  readonly secondWeight: Float32Array;
  readonly secondBias: Float32Array;
}

export interface TransitionInput {
  readonly activations: Float32Array;
  readonly rows: number;
  readonly channels: number;
  readonly hiddenChannels: number;
  readonly weights: TransitionWeights;
  readonly epsilon?: number;
}

export interface TransitionResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

export interface TransitionGpuOptions {
  /** Test/diagnostic override; production sizing is derived from WebGPU limits. */
  readonly maxChunkRows?: number;
}

const ceilDivide = (value: number, divisor: number): number => Math.ceil(value / divisor);
const LINEAR_THREADS = 256;
const LINEAR_TILE_INNER = 8;
/** Output tile one workgroup of the shared projection covers. */
export const TRANSITION_TILE_COLUMNS = 128;
export const TRANSITION_TILE_ROWS = 64;
const LINEAR_TILE_COLUMNS = TRANSITION_TILE_COLUMNS;
const LINEAR_TILE_ROWS = TRANSITION_TILE_ROWS;
export const TRANSITION_CHUNK_TARGET_BYTES = 96 * 1024 * 1024;

const gcd = (left: number, right: number): number => {
  let a = left; let b = right;
  while (b !== 0) { const remainder = a % b; a = b; b = remainder; }
  return a;
};

/** Selects an aligned row window whose largest transition binding stays bounded. */
export function transitionChunkRows(
  rows: number,
  channels: number,
  hiddenChannels: number,
  maxStorageBufferBindingSize: number,
  minStorageBufferOffsetAlignment = 256,
): number {
  if (![rows, channels, hiddenChannels, maxStorageBufferBindingSize, minStorageBufferOffsetAlignment]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("transition chunk dimensions and limits must be positive safe integers");
  }
  const rowBytes = Math.max(channels, hiddenChannels) * Float32Array.BYTES_PER_ELEMENT;
  if (rows * rowBytes <= maxStorageBufferBindingSize) return rows;
  const capacity = Math.floor(Math.min(maxStorageBufferBindingSize, TRANSITION_CHUNK_TARGET_BYTES) / rowBytes);
  if (capacity < 1) throw new RangeError("WebGPU storage binding is too small for one transition row");
  const sourceRowBytes = channels * Float32Array.BYTES_PER_ELEMENT;
  const offsetRowAlignment = minStorageBufferOffsetAlignment / gcd(sourceRowBytes, minStorageBufferOffsetAlignment);
  const rowAlignment = TRANSITION_TILE_ROWS * offsetRowAlignment
    / gcd(TRANSITION_TILE_ROWS, offsetRowAlignment);
  if (rows <= capacity) return rows;
  if (capacity < rowAlignment) {
    throw new RangeError("WebGPU storage binding cannot hold one aligned transition chunk");
  }
  return Math.min(rows, Math.floor(capacity / rowAlignment) * rowAlignment);
}

/**
 * The dense projection shared by every AlphaFold module.
 *
 * Row-major A[rows, inner] x row-major W[inner, columns] + bias, with an
 * optional ReLU and an optional accumulate-into-output form. One workgroup of
 * 256 invocations covers a TILE_ROWS x TILE_COLUMNS output tile; each
 * invocation keeps eight contiguous rows by four columns in registers, so a
 * single k step costs three vector reads from workgroup memory and 32 fused
 * multiply-adds. A is staged k-major, which makes both its global loads and
 * its shared loads contiguous.
 *
 * Measured against the previous 16x64 tiling on GB10: 1.6x at the Evoformer
 * projection and transition shapes, at 6 KiB of workgroup storage, well inside
 * the 16 KiB every WebGPU device guarantees.
 */
export function createLinearShader(residual: boolean): string {
  const rowsPerThread = LINEAR_TILE_ROWS / (LINEAR_THREADS / (LINEAR_TILE_COLUMNS / 4));
  const columnThreads = LINEAR_TILE_COLUMNS / 4;
  const sourceVectors = (LINEAR_TILE_ROWS * LINEAR_TILE_INNER) / 4;
  const weightVectors = (LINEAR_TILE_INNER * LINEAR_TILE_COLUMNS) / 4;
  const vectorsPerThread = rowsPerThread / 4;
  const lines = (count: number, body: (index: number) => string): string =>
    Array.from({ length: count }, (_, index) => body(index)).join("\n");
  const store = residual
    ? (target: string, value: string) => `output[${target}] += ${value};`
    : (target: string, value: string) => `output[${target}] = ${value};`;
  return `
struct MatmulParameters {
  rows: u32,
  inner: u32,
  columns: u32,
  weight_offset: u32,
  bias_offset: u32,
  activation: u32,
  padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

// Staged k-major as tile_source[k][row] so one k step reads this invocation's
// rows as ${vectorsPerThread} aligned vectors.
var<workgroup> tile_source: array<vec4<f32>, ${sourceVectors}>;
var<workgroup> tile_weight: array<vec4<f32>, ${weightVectors}>;

@compute @workgroup_size(${LINEAR_THREADS}, 1, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let thread = local.x;
  let column_thread = thread % ${columnThreads}u;
  let row_thread = thread / ${columnThreads}u;
  let row_origin = group.y * ${LINEAR_TILE_ROWS}u + row_thread * ${rowsPerThread}u;
  let column_origin = group.x * ${LINEAR_TILE_COLUMNS}u;
  let column = column_origin + column_thread * 4u;
${lines(rowsPerThread, (row) => `  var acc${row} = vec4<f32>(0.0);`)}

  for (var k0 = 0u; k0 < parameters.inner; k0 += ${LINEAR_TILE_INNER}u) {
    for (var item = thread; item < ${LINEAR_TILE_ROWS * LINEAR_TILE_INNER}u; item += ${LINEAR_THREADS}u) {
      let load_row = item / ${LINEAR_TILE_INNER}u;
      let load_k = item % ${LINEAR_TILE_INNER}u;
      let source_row = group.y * ${LINEAR_TILE_ROWS}u + load_row;
      let source_k = k0 + load_k;
      var value = 0.0;
      if (source_row < parameters.rows && source_k < parameters.inner) {
        value = source[source_row * parameters.inner + source_k];
      }
      let slot = load_k * ${LINEAR_TILE_ROWS}u + load_row;
      tile_source[slot / 4u][slot % 4u] = value;
    }
    for (var item = thread; item < ${weightVectors}u; item += ${LINEAR_THREADS}u) {
      let load_k = item / ${columnThreads}u;
      let load_column = column_origin + (item % ${columnThreads}u) * 4u;
      let weight_k = k0 + load_k;
      var value = vec4<f32>(0.0);
      if (weight_k < parameters.inner) {
        let base = parameters.weight_offset + weight_k * parameters.columns + load_column;
        if (load_column + 3u < parameters.columns) {
          value = vec4<f32>(weights[base], weights[base + 1u], weights[base + 2u], weights[base + 3u]);
        } else {
          for (var lane = 0u; lane < 4u; lane += 1u) {
            if (load_column + lane < parameters.columns) { value[lane] = weights[base + lane]; }
          }
        }
      }
      tile_weight[item] = value;
    }
    workgroupBarrier();
    for (var k = 0u; k < ${LINEAR_TILE_INNER}u; k += 1u) {
      let w = tile_weight[k * ${columnThreads}u + column_thread];
      let a_base = (k * ${LINEAR_TILE_ROWS}u + row_thread * ${rowsPerThread}u) / 4u;
${lines(vectorsPerThread, (vector) => `      let a${vector} = tile_source[a_base + ${vector}u];`)}
${lines(rowsPerThread, (row) => `      acc${row} += a${Math.floor(row / 4)}[${row % 4}u] * w;`)}
    }
    workgroupBarrier();
  }

  var bias = vec4<f32>(0.0);
  if (column + 3u < parameters.columns) {
    let base = parameters.bias_offset + column;
    bias = vec4<f32>(weights[base], weights[base + 1u], weights[base + 2u], weights[base + 3u]);
  } else {
    for (var lane = 0u; lane < 4u; lane += 1u) {
      if (column + lane < parameters.columns) { bias[lane] = weights[parameters.bias_offset + column + lane]; }
    }
  }
${lines(rowsPerThread, (row) => `
  {
    let row = row_origin + ${row}u;
    if (row < parameters.rows) {
      var value = acc${row} + bias;
      if (parameters.activation == 1u) { value = max(value, vec4<f32>(0.0)); }
      let base = row * parameters.columns + column;
      if (column + 3u < parameters.columns) {
        ${store("base", "value.x")} ${store("base + 1u", "value.y")}
        ${store("base + 2u", "value.z")} ${store("base + 3u", "value.w")}
      } else {
        for (var lane = 0u; lane < 4u; lane += 1u) {
          if (column + lane < parameters.columns) { ${store("base + lane", "value[lane]")} }
        }
      }
    }
  }`)}
}`;
}

function validate(input: TransitionInput): void {
  const { rows, channels, hiddenChannels, activations, weights } = input;
  if (![rows, channels, hiddenChannels].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("transition dimensions must be positive safe integers");
  }
  const lengths: ReadonlyArray<readonly [string, Float32Array, number]> = [
    ["activations", activations, rows * channels],
    ["layerNormScale", weights.layerNormScale, channels],
    ["layerNormOffset", weights.layerNormOffset, channels],
    ["firstWeight", weights.firstWeight, channels * hiddenChannels],
    ["firstBias", weights.firstBias, hiddenChannels],
    ["secondWeight", weights.secondWeight, hiddenChannels * channels],
    ["secondBias", weights.secondBias, channels],
  ];
  for (const [name, value, expected] of lengths) {
    if (value.length !== expected) throw new RangeError(`${name} has ${value.length} values; expected ${expected}`);
  }
}

export function packTransitionWeights(input: TransitionInput): { data: Float32Array; offsets: readonly number[] } {
  const values = [
    input.weights.layerNormScale,
    input.weights.layerNormOffset,
    input.weights.firstWeight,
    input.weights.firstBias,
    input.weights.secondWeight,
    input.weights.secondBias,
  ] as const;
  const offsets: number[] = [];
  let length = 0;
  for (const value of values) {
    offsets.push(length);
    length += value.length;
  }
  const data = new Float32Array(length);
  for (let index = 0; index < values.length; index += 1) data.set(values[index]!, offsets[index]);
  return { data, offsets };
}

export function createTransitionShaders(input: TransitionInput, offsets: readonly number[]): readonly string[] {
  void input;
  void offsets;
  const normalize = `
struct NormalizeParameters {
  rows: u32,
  channels: u32,
  scale_offset: u32,
  offset_offset: u32,
  epsilon: f32,
  padding_0: u32,
  padding_1: u32,
  padding_2: u32,
};
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: NormalizeParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

@compute @workgroup_size(64)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= parameters.rows) { return; }
  let base = row * parameters.channels;
  var sum = 0.0;
  for (var c = local.x; c < parameters.channels; c += 64u) {
    sum += source[base + c];
  }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(parameters.channels); }
  workgroupBarrier();

  var sum_squared = 0.0;
  for (var c = local.x; c < parameters.channels; c += 64u) {
    let centered = source[base + c] - row_mean[0];
    sum_squared += centered * centered;
  }
  partial[local.x] = sum_squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(parameters.channels) + parameters.epsilon);
  for (var c = local.x; c < parameters.channels; c += 64u) {
    output[base + c] = (source[base + c] - row_mean[0]) * inverse_std
      * weights[parameters.scale_offset + c] + weights[parameters.offset_offset + c];
  }
}`;
  return [normalize, createLinearShader(false), createLinearShader(true)];
}

export function createTransitionNormalizeParameters(input: TransitionInput, offsets: readonly number[]): Uint8Array {
  const buffer = new ArrayBuffer(32);
  const view = new DataView(buffer);
  view.setUint32(0, input.rows, true);
  view.setUint32(4, input.channels, true);
  view.setUint32(8, offsets[0]!, true);
  view.setUint32(12, offsets[1]!, true);
  view.setFloat32(16, input.epsilon ?? 1e-5, true);
  return new Uint8Array(buffer);
}

export class TransitionGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  readonly maxChunkRows: number | undefined;

  constructor(device: GPUDevice, options: TransitionGpuOptions = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
    this.maxChunkRows = options.maxChunkRows;
    if (this.maxChunkRows !== undefined
      && (!Number.isSafeInteger(this.maxChunkRows) || this.maxChunkRows <= 0)) {
      throw new RangeError("maxChunkRows must be a positive safe integer");
    }
  }

  async run(input: TransitionInput): Promise<TransitionResult> {
    validate(input);
    const packed = packTransitionWeights(input);
    const chunkRows = Math.min(this.maxChunkRows ?? input.rows, transitionChunkRows(
      input.rows, input.channels, input.hiddenChannels, this.device.limits.maxStorageBufferBindingSize,
      this.device.limits.minStorageBufferOffsetAlignment,
    ));
    const code = createTransitionShaders(input, packed.offsets);
    const key = `transition:${input.rows}:${input.channels}:${input.hiddenChannels}:${input.epsilon ?? 1e-5}`;
    const pipelines: GPUComputePipeline[] = [];
    for (let index = 0; index < code.length; index += 1) {
      pipelines.push(await this.pipelines.get(`${key}:${index}`, code[index]!));
    }
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    const storage = GPUBufferUsage.STORAGE;
    try {
      const source = keep(this.allocator.upload("transition.source", input.activations, storage));
      const weights = keep(this.allocator.upload("transition.weights", packed.data, storage));
      const normalized = keep(this.allocator.allocate(
        "transition.normalized-chunk", chunkRows * input.channels * 4, storage,
      ));
      const hidden = keep(this.allocator.allocate(
        "transition.hidden-chunk", chunkRows * input.hiddenChannels * 4, storage,
      ));
      const output = keep(this.allocator.allocate(
        "transition.output", input.rows * input.channels * 4, storage | GPUBufferUsage.COPY_SRC,
      ));
      const readback = keep(this.allocator.allocate(
        "transition.readback", input.rows * input.channels * 4,
        GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      const encoder = this.device.createCommandEncoder({ label: "transition" });
      this.device.pushErrorScope("validation");
      type Binding = { readonly buffer: GPUBuffer; readonly offset?: number; readonly size?: number };
      const pass = (pipeline: GPUComputePipeline, buffers: readonly Binding[], x: number, y = 1): void => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((resource, binding) => ({ binding, resource })),
        }));
        compute.dispatchWorkgroups(x, y);
        compute.end();
      };
      const binding = (buffer: AllocatedGpuBuffer, offset = 0, size = buffer.byteLength): Binding =>
        ({ buffer: buffer.buffer, offset, size });
      for (let rowOffset = 0; rowOffset < input.rows; rowOffset += chunkRows) {
        const count = Math.min(chunkRows, input.rows - rowOffset);
        const chunkInput = { ...input, rows: count, activations: new Float32Array(0) };
        const layerNormParameters = keep(this.allocator.upload(
          `transition.normalize.parameters-${rowOffset}`,
          createTransitionNormalizeParameters(chunkInput, packed.offsets), GPUBufferUsage.UNIFORM,
        ));
        const firstParameters = keep(this.allocator.upload(
          `transition.first.parameters-${rowOffset}`, new Uint32Array([
            count, input.channels, input.hiddenChannels, packed.offsets[2]!, packed.offsets[3]!, 1, 0, 0,
          ]), GPUBufferUsage.UNIFORM,
        ));
        const secondParameters = keep(this.allocator.upload(
          `transition.second.parameters-${rowOffset}`, new Uint32Array([
            count, input.hiddenChannels, input.channels, packed.offsets[4]!, packed.offsets[5]!, 0, 0, 0,
          ]), GPUBufferUsage.UNIFORM,
        ));
        const sourceBytes = count * input.channels * 4;
        const hiddenBytes = count * input.hiddenChannels * 4;
        const sourceOffset = rowOffset * input.channels * 4;
        const normalizeGrid = [Math.min(count, 32_768), ceilDivide(count, 32_768)] as const;
        pass(pipelines[0]!, [
          binding(source, sourceOffset, sourceBytes), binding(weights), binding(layerNormParameters),
          binding(normalized, 0, sourceBytes),
        ], normalizeGrid[0], normalizeGrid[1]);
        pass(pipelines[1]!, [
          binding(normalized, 0, sourceBytes), binding(weights), binding(firstParameters),
          binding(hidden, 0, hiddenBytes),
        ], ceilDivide(input.hiddenChannels, TRANSITION_TILE_COLUMNS), ceilDivide(count, TRANSITION_TILE_ROWS));
        pass(pipelines[1]!, [
          binding(hidden, 0, hiddenBytes), binding(weights), binding(secondParameters),
          binding(output, sourceOffset, sourceBytes),
        ], ceilDivide(input.channels, TRANSITION_TILE_COLUMNS), ceilDivide(count, TRANSITION_TILE_ROWS));
      }
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, input.rows * input.channels * 4);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      await readback.buffer.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.buffer.getMappedRange().slice(0));
      readback.buffer.unmap();
      return {
        output: result,
        elapsedMilliseconds: performance.now() - start,
        memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index]!.release();
    }
  }
}
