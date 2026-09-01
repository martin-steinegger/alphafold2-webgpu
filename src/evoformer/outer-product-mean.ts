import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";
import { createTiledGemmShader, gemmGrid } from "../runtime/gemm.js";

export interface OuterProductMeanWeights {
  readonly layerNormScale: Float32Array;
  readonly layerNormOffset: Float32Array;
  readonly leftWeight: Float32Array;
  readonly leftBias: Float32Array;
  readonly rightWeight: Float32Array;
  readonly rightBias: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
}

export interface OuterProductMeanInput {
  readonly activations: Float32Array;
  readonly mask: Float32Array;
  readonly sequences: number;
  readonly length: number;
  readonly cM: number;
  readonly cOuter: number;
  readonly cZ: number;
  readonly weights: OuterProductMeanWeights;
  readonly layerNormEpsilon?: number;
  readonly normalizationEpsilon?: number;
  /** Overrides the residue block the contraction is chunked over, for tests. */
  readonly rowBlockResidues?: number;
}

export interface OuterProductMeanResult {
  readonly output: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

const GRID_WIDTH = 32_768;
const ceilDivide = (value: number, divisor: number): number => Math.ceil(value / divisor);

function validate(input: OuterProductMeanInput): void {
  const { sequences, length, cM, cOuter, cZ, activations, mask, weights } = input;
  if (![sequences, length, cM, cOuter, cZ].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("outer product mean dimensions must be positive safe integers");
  }
  const expected: ReadonlyArray<readonly [string, Float32Array, number]> = [
    ["activations", activations, sequences * length * cM],
    ["mask", mask, sequences * length],
    ["layerNormScale", weights.layerNormScale, cM],
    ["layerNormOffset", weights.layerNormOffset, cM],
    ["leftWeight", weights.leftWeight, cM * cOuter],
    ["leftBias", weights.leftBias, cOuter],
    ["rightWeight", weights.rightWeight, cM * cOuter],
    ["rightBias", weights.rightBias, cOuter],
    ["outputWeight", weights.outputWeight, cOuter * cOuter * cZ],
    ["outputBias", weights.outputBias, cZ],
  ];
  for (const [name, value, size] of expected) {
    if (value.length !== size) throw new RangeError(`${name} has ${value.length} values; expected ${size}`);
  }
}

export function packOuterProductMeanWeights(input: OuterProductMeanInput): { data: Float32Array; offsets: readonly number[] } {
  const tensors = [
    input.weights.layerNormScale, input.weights.layerNormOffset,
    input.weights.leftWeight, input.weights.leftBias,
    input.weights.rightWeight, input.weights.rightBias,
    input.weights.outputWeight, input.weights.outputBias,
  ] as const;
  const offsets: number[] = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const data = new Float32Array(size);
  tensors.forEach((tensor, index) => data.set(tensor, offsets[index]));
  return { data, offsets };
}

export function createOuterProductMeanParameters(input: OuterProductMeanInput, offsets: readonly number[]): Uint8Array {
  const buffer = new ArrayBuffer(64);
  const view = new DataView(buffer);
  const integers = [input.sequences, input.length, input.cM, input.cOuter, input.cZ, ...offsets];
  integers.forEach((value, index) => view.setUint32(index * 4, value!, true));
  view.setFloat32(52, input.layerNormEpsilon ?? 1e-5, true);
  view.setFloat32(56, input.normalizationEpsilon ?? 1e-3, true);
  return new Uint8Array(buffer);
}

const COMMON = `
struct Parameters {
  sequences: u32,
  length: u32,
  c_m: u32,
  c_outer: u32,
  c_z: u32,
  layer_norm_scale: u32,
  layer_norm_offset: u32,
  left_weight: u32,
  left_bias: u32,
  right_weight: u32,
  right_bias: u32,
  output_weight: u32,
  output_bias: u32,
  layer_norm_epsilon: f32,
  normalization_epsilon: f32,
  padding: u32,
};
const GRID_WIDTH: u32 = 32768u;
`;

export const OUTER_PRODUCT_MEAN_NORMALIZE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
var<workgroup> partial: array<f32, 64>;
var<workgroup> row_mean: array<f32, 1>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let rows = p.sequences * p.length;
  let row = group.x + group.y * GRID_WIDTH;
  if (row >= rows) { return; }
  let base = row * p.c_m;
  var sum = 0.0;
  for (var c = local.x; c < p.c_m; c += 64u) { sum += source[base + c]; }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) { row_mean[0] = partial[0] / f32(p.c_m); }
  workgroupBarrier();
  var squared = 0.0;
  for (var c = local.x; c < p.c_m; c += 64u) {
    let centered = source[base + c] - row_mean[0];
    squared += centered * centered;
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride /= 2u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let inverse_std = inverseSqrt(partial[0] / f32(p.c_m) + p.layer_norm_epsilon);
  for (var c = local.x; c < p.c_m; c += 64u) {
    output[base + c] = (source[base + c] - row_mean[0]) * inverse_std
      * weights[p.layer_norm_scale + c] + weights[p.layer_norm_offset + c];
  }
}`;

export const OUTER_PRODUCT_MEAN_PROJECT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> left: array<f32>;
@group(0) @binding(5) var<storage, read_write> right: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  let rows = p.sequences * p.length;
  if (index >= rows * p.c_outer) { return; }
  let row = index / p.c_outer;
  let outer = index % p.c_outer;
  var left_value = weights[p.left_bias + outer];
  var right_value = weights[p.right_bias + outer];
  for (var c = 0u; c < p.c_m; c += 1u) {
    let value = source[row * p.c_m + c];
    left_value += value * weights[p.left_weight + c * p.c_outer + outer];
    right_value += value * weights[p.right_weight + c * p.c_outer + outer];
  }
  left[index] = mask[row] * left_value;
  right[index] = mask[row] * right_value;
}`;

export const OUTER_PRODUCT_MEAN_INTERMEDIATE_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  let elements = p.sequences * p.length * p.c_outer * p.c_z;
  if (index >= elements) { return; }
  let z = index % p.c_z;
  let outer_right = (index / p.c_z) % p.c_outer;
  let residue = (index / (p.c_z * p.c_outer)) % p.length;
  let sequence = index / (p.c_z * p.c_outer * p.length);
  var value = 0.0;
  for (var outer_left = 0u; outer_left < p.c_outer; outer_left += 1u) {
    let left_index = (sequence * p.length + residue) * p.c_outer + outer_left;
    let weight_index = ((outer_left * p.c_outer + outer_right) * p.c_z) + z;
    value += left[left_index] * weights[p.output_weight + weight_index];
  }
  output[index] = value;
}`;

export const OUTER_PRODUCT_MEAN_OUTPUT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> right: array<f32>;
@group(0) @binding(1) var<storage, read> intermediate: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.c_z) { return; }
  let z = index % p.c_z;
  let pair = index / p.c_z;
  let i = pair / p.length;
  let j = pair % p.length;
  var value = weights[p.output_bias + z];
  var count = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    count += mask[sequence * p.length + i] * mask[sequence * p.length + j];
    for (var outer = 0u; outer < p.c_outer; outer += 1u) {
      let temp_index = (((sequence * p.length + i) * p.c_outer + outer) * p.c_z) + z;
      let right_index = (sequence * p.length + j) * p.c_outer + outer;
      value += intermediate[temp_index] * right[right_index];
    }
  }
  output[index] = value / (p.normalization_epsilon + count);
}`;

const OPM_TILE_COMMON = `${COMMON}
struct TileParameters { offset: u32, count: u32, padding0: u32, padding1: u32 };
`;

export const OUTER_PRODUCT_MEAN_PAIR_COUNT_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> mask: array<f32>;
@group(0) @binding(1) var<uniform> p: Parameters;
@group(0) @binding(2) var<storage, read_write> pair_count: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length) { return; }
  let i = index / p.length;
  let j = index % p.length;
  var count = 0.0;
  for (var sequence = 0u; sequence < p.sequences; sequence += 1u) {
    count += mask[sequence * p.length + i] * mask[sequence * p.length + j];
  }
  pair_count[index] = count;
}`;

/**
 * The outer-product contraction, as one GEMM.
 *
 * outer[i][j][a][b] = sum_s left[s][i][a] * right[s][j][b] is left-transpose
 * times right once the residue and outer-channel axes are folded into a single
 * index: both operands are already stored sequence-major, so the contraction
 * reads them contiguously. Only the store has to unfold (i, a) and (j, b) back
 * into the layout the output projection expects.
 */
export const OUTER_PRODUCT_MEAN_CONTRACT_SHADER = createTiledGemmShader({
  preamble: `${OPM_TILE_COMMON}
@group(0) @binding(0) var<storage, read> left: array<f32>;
@group(0) @binding(1) var<storage, read> right: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<uniform> tile: TileParameters;
@group(0) @binding(4) var<storage, read_write> outer: array<f32>;`,
  rows: "tile.count * p.c_outer",
  inner: "p.sequences",
  columns: "p.length * p.c_outer",
  sourceElement: "left[k * p.length * p.c_outer + tile.offset * p.c_outer + row]",
  weightElement: "right[k * p.length * p.c_outer + column]",
  store: `let block_i = row / p.c_outer;
          let outer_left = row % p.c_outer;
          let j = column / p.c_outer;
          let outer_right = column % p.c_outer;
          outer[((block_i * p.length + j) * p.c_outer + outer_left) * p.c_outer + outer_right] = element;`,
});

/**
 * Output projection of the contracted outer product.
 *
 * The contraction axis is the flattened outer-channel pair, which the layout
 * above makes contiguous per residue pair, so this is a plain GEMM followed by
 * the shared pair normalizer.
 */
function createOuterProductMeanProjectOutputShader(residual: boolean): string {
  return createTiledGemmShader({
    preamble: `${OPM_TILE_COMMON}
@group(0) @binding(0) var<storage, read> outer: array<f32>;
@group(0) @binding(1) var<storage, read> pair_count: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<uniform> tile: TileParameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;`,
    rows: "tile.count * p.length",
    inner: "p.c_outer * p.c_outer",
    columns: "p.c_z",
    sourceElement: "outer[row * p.c_outer * p.c_outer + k]",
    weightElement: "weights[p.output_weight + k * p.c_z + column]",
    store: `let pair = tile.offset * p.length + row;
          let projected = element + weights[p.output_bias + column];
          output[pair * p.c_z + column] ${residual ? "+=" : "="}
            projected / (p.normalization_epsilon + pair_count[pair]);`,
  });
}

export const OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER = createOuterProductMeanProjectOutputShader(false);

export const OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_RESIDUAL_SHADER =
  createOuterProductMeanProjectOutputShader(true);

/**
 * Residue block whose contracted outer product stays inside the scratch budget.
 *
 * Contracting first and projecting second costs 2.7x fewer multiply-adds at 256
 * residues than folding the projection into a per-sequence accumulation, and
 * both halves are dense GEMMs rather than a strided reduction. What it costs is
 * a length^2 * c_outer^2 intermediate, 268 MB at that length, so the
 * contraction is blocked over the first residue axis: every block keeps the
 * cheaper arithmetic while the intermediate stays bounded.
 */
/**
 * Rows of the normalized MSA one window covers.
 *
 * The normalization and the left/right projection are both row-wise, and the
 * projections they feed are a small fraction of the width, so the full-width
 * normalized copy only has to exist one window at a time. Windows are a
 * multiple of 64 rows so the mask view stays on a valid binding offset.
 */
export const OUTER_PRODUCT_NORMALIZE_WINDOW_BYTES = 16 * 1024 * 1024;

export function outerProductMeanNormalizeWindow(
  rows: number, cM: number, budgetBytes: number = OUTER_PRODUCT_NORMALIZE_WINDOW_BYTES,
): number {
  if (![rows, cM, budgetBytes].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("outer-product normalize window dimensions must be positive safe integers");
  }
  const alignment = 64;
  const capacity = Math.floor(budgetBytes / (cM * Float32Array.BYTES_PER_ELEMENT));
  const aligned = Math.floor(capacity / alignment) * alignment;
  return Math.max(alignment, Math.min(rows, aligned));
}

export const OUTER_PRODUCT_BLOCK_LIMIT_BYTES = 32 * 1024 * 1024;

export function outerProductMeanRowBlock(
  length: number, cOuter: number, budgetBytes: number = OUTER_PRODUCT_BLOCK_LIMIT_BYTES,
): number {
  if (![length, cOuter, budgetBytes].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("outer-product block dimensions and budget must be positive safe integers");
  }
  const bytesPerResidue = length * cOuter * cOuter * Float32Array.BYTES_PER_ELEMENT;
  return Math.max(1, Math.min(length, Math.floor(budgetBytes / bytesPerResidue)));
}

export class OuterProductMeanGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;

  constructor(device: GPUDevice) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input: OuterProductMeanInput): Promise<OuterProductMeanResult> {
    validate(input);
    const packed = packOuterProductMeanWeights(input);
    const [normalize, project, contractPipeline, pairCountPipeline, projectOutputPipeline] = await Promise.all([
      this.pipelines.get("opm:normalize", OUTER_PRODUCT_MEAN_NORMALIZE_SHADER),
      this.pipelines.get("opm:project", OUTER_PRODUCT_MEAN_PROJECT_SHADER),
      this.pipelines.get("opm:contract", OUTER_PRODUCT_MEAN_CONTRACT_SHADER),
      this.pipelines.get("opm:pair-count", OUTER_PRODUCT_MEAN_PAIR_COUNT_SHADER),
      this.pipelines.get("opm:project-output", OUTER_PRODUCT_MEAN_PROJECT_OUTPUT_SHADER),
    ]);
    const storage = GPUBufferUsage.STORAGE;
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    const rows = input.sequences * input.length;
    const pairElements = input.length * input.length * input.cZ;
    const linearGrid = (elements: number): readonly [number, number] => {
      const groups = ceilDivide(elements, 64);
      return [Math.min(groups, GRID_WIDTH), ceilDivide(groups, GRID_WIDTH)];
    };
    const rowBlock = input.rowBlockResidues ?? outerProductMeanRowBlock(input.length, input.cOuter);
    try {
      const source = keep(this.allocator.upload("opm.source", input.activations, storage));
      const mask = keep(this.allocator.upload("opm.mask", input.mask, storage));
      const weights = keep(this.allocator.upload("opm.weights", packed.data, storage));
      const params = keep(this.allocator.upload(
        "opm.parameters", createOuterProductMeanParameters(input, packed.offsets), GPUBufferUsage.UNIFORM,
      ));
      const normalized = keep(this.allocator.allocate("opm.normalized", rows * input.cM * 4, storage));
      const left = keep(this.allocator.allocate("opm.left", rows * input.cOuter * 4, storage));
      const right = keep(this.allocator.allocate("opm.right", rows * input.cOuter * 4, storage));
      const outer = keep(this.allocator.allocate(
        "opm.outer", rowBlock * input.length * input.cOuter * input.cOuter * 4, storage,
      ));
      const pairCount = keep(this.allocator.allocate(
        "opm.pair-count", input.length * input.length * 4, storage,
      ));
      const output = keep(this.allocator.allocate(
        "opm.output", pairElements * 4, storage | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      ));
      const readback = keep(this.allocator.allocate(
        "opm.readback", pairElements * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      ));
      const encoder = this.device.createCommandEncoder({ label: "outer-product-mean" });
      this.device.pushErrorScope("validation");
      const pass = (pipeline: GPUComputePipeline, buffers: readonly GPUBuffer[], x: number, y = 1): void => {
        const compute = encoder.beginComputePass();
        compute.setPipeline(pipeline);
        compute.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        compute.dispatchWorkgroups(x, y);
        compute.end();
      };
      const normalizeGrid = [Math.min(rows, GRID_WIDTH), ceilDivide(rows, GRID_WIDTH)] as const;
      pass(normalize, [source.buffer, weights.buffer, params.buffer, normalized.buffer],
        normalizeGrid[0], normalizeGrid[1]);
      const projectGrid = linearGrid(rows * input.cOuter);
      pass(project, [normalized.buffer, mask.buffer, weights.buffer, params.buffer, left.buffer, right.buffer],
        projectGrid[0], projectGrid[1]);
      const pairCountGrid = linearGrid(input.length * input.length);
      pass(pairCountPipeline, [mask.buffer, params.buffer, pairCount.buffer],
        pairCountGrid[0], pairCountGrid[1]);
      for (let offset = 0; offset < input.length; offset += rowBlock) {
        const count = Math.min(rowBlock, input.length - offset);
        const tile = keep(this.allocator.upload(
          `opm.block-${offset}`, new Uint32Array([offset, count, 0, 0]), GPUBufferUsage.UNIFORM,
        ));
        const contractGrid = gemmGrid(count * input.cOuter, input.length * input.cOuter);
        pass(contractPipeline, [left.buffer, right.buffer, params.buffer, tile.buffer, outer.buffer],
          contractGrid[0], contractGrid[1]);
        const outputGrid = gemmGrid(count * input.length, input.cZ);
        pass(projectOutputPipeline,
          [outer.buffer, pairCount.buffer, weights.buffer, params.buffer, tile.buffer, output.buffer],
          outputGrid[0], outputGrid[1]);
      }
      encoder.copyBufferToBuffer(output.buffer, 0, readback.buffer, 0, pairElements * 4);
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
