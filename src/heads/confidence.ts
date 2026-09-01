import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "../evoformer/attention.js";
import {
  createTransitionShaders, TRANSITION_TILE_COLUMNS, TRANSITION_TILE_ROWS, type TransitionInput,
} from "../evoformer/transition.js";
import {
  GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot,
} from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";

export interface PredictedLddtWeights {
  readonly normScale: Float32Array;
  readonly normOffset: Float32Array;
  readonly act0Weight: Float32Array;
  readonly act0Bias: Float32Array;
  readonly act1Weight: Float32Array;
  readonly act1Bias: Float32Array;
  readonly logitsWeight: Float32Array;
  readonly logitsBias: Float32Array;
}

export interface PredictedAlignedErrorWeights {
  readonly logitsWeight: Float32Array;
  readonly logitsBias: Float32Array;
}

export interface ConfidenceSummaryResult {
  readonly plddt: Float32Array;
  readonly meanPlddt: number;
  readonly predictedAlignedError: Float32Array;
  readonly maxPredictedAlignedError: number;
  readonly ptm: number;
  readonly memory?: AllocationSnapshot;
}

/** Diagnostic result retaining the categorical logits used by differential tests. */
export interface ConfidenceResult extends ConfidenceSummaryResult {
  readonly lddtLogits: Float32Array;
  readonly paeLogits: Float32Array;
}

/**
 * Memory-bounded production result. TM expectation terms are retained only long
 * enough for the multimer wrapper to derive ipTM; they are not user-facing.
 */
export interface ReducedConfidenceResult extends ConfidenceSummaryResult {
  readonly tmScoreTerms: Float32Array;
}

export interface ReducedConfidenceOptions {
  readonly pairBuffer?: GPUBuffer;
  /** Test/diagnostic override; production uses the fixed bounded target. */
  readonly maxPaeLogitsBytes?: number;
}

const LINEAR_SHADER = createTransitionShaders({} as TransitionInput, [])[1]!;
const RELU_SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index < arrayLength(&source)) { output[index] = max(source[index], 0.0); }
}`;

/** Largest transient PAE-logit allocation in memory-bounded inference. */
export const PAE_LOGITS_WINDOW_BYTES = 16 * 1024 * 1024;

const PAE_EXPECTATION_SHADER = `
struct ExpectationParameters {
  rows: u32,
  bins: u32,
  output_offset: u32,
  center_offset: u32,
  tm_offset: u32,
  padding_0: u32,
  padding_1: u32,
  padding_2: u32,
};
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> constants: array<f32>;
@group(0) @binding(2) var<uniform> parameters: ExpectationParameters;
@group(0) @binding(3) var<storage, read_write> predicted_error: array<f32>;
@group(0) @binding(4) var<storage, read_write> tm_score_terms: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= parameters.rows) { return; }
  let base = row * parameters.bins;
  var maximum = -3.402823466e+38;
  for (var bin = 0u; bin < parameters.bins; bin += 1u) {
    maximum = max(maximum, logits[base + bin]);
  }
  var denominator = 0.0;
  var error_numerator = 0.0;
  var tm_numerator = 0.0;
  for (var bin = 0u; bin < parameters.bins; bin += 1u) {
    let probability = exp(logits[base + bin] - maximum);
    denominator += probability;
    error_numerator += probability * constants[parameters.center_offset + bin];
    tm_numerator += probability * constants[parameters.tm_offset + bin];
  }
  let output_row = parameters.output_offset + row;
  predicted_error[output_row] = error_numerator / denominator;
  tm_score_terms[output_row] = tm_numerator / denominator;
}`;

const gcd = (left: number, right: number): number => {
  let a = left; let b = right;
  while (b !== 0) { const remainder = a % b; a = b; b = remainder; }
  return a;
};

function paeWindowRows(
  rows: number,
  pairChannels: number,
  paeBins: number,
  maxStorageBufferBindingSize: number,
  minStorageBufferOffsetAlignment: number,
  targetBytes: number,
): number {
  if (![rows, pairChannels, paeBins, maxStorageBufferBindingSize,
    minStorageBufferOffsetAlignment, targetBytes]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("PAE window dimensions and limits must be positive safe integers");
  }
  const pairRowBytes = pairChannels * Float32Array.BYTES_PER_ELEMENT;
  const logitsRowBytes = paeBins * Float32Array.BYTES_PER_ELEMENT;
  const capacity = Math.min(
    rows,
    Math.floor(Math.min(targetBytes, maxStorageBufferBindingSize) / logitsRowBytes),
    Math.floor(maxStorageBufferBindingSize / pairRowBytes),
  );
  const offsetRowAlignment = minStorageBufferOffsetAlignment
    / gcd(pairRowBytes, minStorageBufferOffsetAlignment);
  const rowAlignment = TRANSITION_TILE_ROWS * offsetRowAlignment
    / gcd(TRANSITION_TILE_ROWS, offsetRowAlignment);
  if (capacity < rowAlignment && capacity < rows) {
    throw new RangeError("WebGPU storage binding is too small for one aligned PAE window");
  }
  return capacity >= rows ? rows : Math.floor(capacity / rowAlignment) * rowAlignment;
}

function softmaxExpected(logits: Float32Array, rows: number, bins: number, centers: Float32Array): Float32Array {
  const result = new Float32Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * bins;
    let maximum = -Infinity;
    for (let bin = 0; bin < bins; bin += 1) maximum = Math.max(maximum, logits[base + bin]!);
    let denominator = 0;
    let numerator = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const probability = Math.exp(logits[base + bin]! - maximum);
      denominator += probability;
      numerator += probability * centers[bin]!;
    }
    result[row] = numerator / denominator;
  }
  return result;
}

function paeCenters(breaks: Float32Array): Float32Array {
  const bins = breaks.length + 1;
  const centers = new Float32Array(bins);
  const step = breaks[1]! - breaks[0]!;
  for (let bin = 0; bin < bins - 1; bin += 1) centers[bin] = breaks[bin]! + step / 2;
  centers[bins - 1] = breaks[breaks.length - 1]! + step / 2;
  return centers;
}

export function predictedTmScore(logits: Float32Array, length: number, breaks: Float32Array): number {
  const bins = breaks.length + 1;
  if (logits.length !== length * length * bins) throw new RangeError("invalid PAE logits shape");
  const centers = paeCenters(breaks);
  const effectiveLength = Math.max(length, 19);
  const d0 = 1.24 * Math.cbrt(effectiveLength - 15) - 1.8;
  const tmPerBin = centers.map((center) => 1 / (1 + center * center / (d0 * d0)));
  let score = 0;
  for (let anchor = 0; anchor < length; anchor += 1) {
    let alignment = 0;
    for (let residue = 0; residue < length; residue += 1) {
      const base = (anchor * length + residue) * bins;
      let maximum = -Infinity;
      for (let bin = 0; bin < bins; bin += 1) maximum = Math.max(maximum, logits[base + bin]!);
      let denominator = 0;
      let numerator = 0;
      for (let bin = 0; bin < bins; bin += 1) {
        const probability = Math.exp(logits[base + bin]! - maximum);
        denominator += probability;
        numerator += probability * tmPerBin[bin]!;
      }
      alignment += numerator / denominator;
    }
    score = Math.max(score, alignment / length);
  }
  return score;
}

/** Reduces one precomputed expected TM term per ordered residue pair. */
export function predictedTmScoreFromExpected(tmScoreTerms: Float32Array, length: number): number {
  if (!Number.isSafeInteger(length) || length <= 0 || tmScoreTerms.length !== length * length) {
    throw new RangeError("invalid expected TM score shape");
  }
  let score = 0;
  for (let anchor = 0; anchor < length; anchor += 1) {
    let alignment = 0;
    const base = anchor * length;
    for (let residue = 0; residue < length; residue += 1) {
      alignment += tmScoreTerms[base + residue]!;
    }
    score = Math.max(score, alignment / length);
  }
  return score;
}

function validateAsymId(asymId: Float32Array, length: number): void {
  if (asymId.length !== length || asymId.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
    throw new RangeError("asymId must contain one positive integer chain identifier per residue");
  }
}

/** Official AlphaFold-Multimer interface TM score: only inter-chain residue pairs contribute. */
export function predictedInterfaceTmScore(
  logits: Float32Array,
  length: number,
  breaks: Float32Array,
  asymId: Float32Array,
): number {
  const bins = breaks.length + 1;
  if (logits.length !== length * length * bins) throw new RangeError("invalid PAE logits shape");
  validateAsymId(asymId, length);
  const centers = paeCenters(breaks);
  const effectiveLength = Math.max(length, 19);
  const d0 = 1.24 * Math.cbrt(effectiveLength - 15) - 1.8;
  const tmPerBin = centers.map((center) => 1 / (1 + center * center / (d0 * d0)));
  let score = 0;
  for (let anchor = 0; anchor < length; anchor += 1) {
    let alignment = 0;
    let partners = 0;
    for (let residue = 0; residue < length; residue += 1) {
      if (asymId[anchor] === asymId[residue]) continue;
      const base = (anchor * length + residue) * bins;
      let maximum = -Infinity;
      for (let bin = 0; bin < bins; bin += 1) maximum = Math.max(maximum, logits[base + bin]!);
      let denominator = 0;
      let numerator = 0;
      for (let bin = 0; bin < bins; bin += 1) {
        const probability = Math.exp(logits[base + bin]! - maximum);
        denominator += probability;
        numerator += probability * tmPerBin[bin]!;
      }
      alignment += numerator / denominator;
      partners += 1;
    }
    if (partners > 0) score = Math.max(score, alignment / partners);
  }
  return score;
}

/** Official ipTM reduction from one precomputed expected TM term per residue pair. */
export function predictedInterfaceTmScoreFromExpected(
  tmScoreTerms: Float32Array,
  length: number,
  asymId: Float32Array,
): number {
  if (!Number.isSafeInteger(length) || length <= 0 || tmScoreTerms.length !== length * length) {
    throw new RangeError("invalid expected TM score shape");
  }
  validateAsymId(asymId, length);
  let score = 0;
  for (let anchor = 0; anchor < length; anchor += 1) {
    let alignment = 0;
    let partners = 0;
    const base = anchor * length;
    for (let residue = 0; residue < length; residue += 1) {
      if (asymId[anchor] === asymId[residue]) continue;
      alignment += tmScoreTerms[base + residue]!;
      partners += 1;
    }
    if (partners > 0) score = Math.max(score, alignment / partners);
  }
  return score;
}

export function multimerRankingConfidence(ptm: number, iptm: number): number {
  if (![ptm, iptm].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) {
    throw new RangeError("pTM and ipTM must be finite scores between zero and one");
  }
  return 0.2 * ptm + 0.8 * iptm;
}

interface ConfidenceDimensions {
  readonly structureChannels: number;
  readonly pairChannels: number;
  readonly hiddenChannels: number;
  readonly lddtBins: number;
  readonly paeBins: number;
}

function validateConfidenceInputs(
  structureRepresentation: Float32Array,
  pairRepresentation: Float32Array,
  length: number,
  lddtWeights: PredictedLddtWeights,
  paeWeights: PredictedAlignedErrorWeights,
  breaks: Float32Array,
  pairBuffer: GPUBuffer | undefined,
): ConfidenceDimensions {
  if (!Number.isSafeInteger(length) || length <= 0) {
    throw new RangeError("confidence length must be a positive safe integer");
  }
  if (structureRepresentation.length === 0 || structureRepresentation.length % length !== 0) {
    throw new RangeError("invalid structure representation shape");
  }
  const structureChannels = structureRepresentation.length / length;
  const hiddenChannels = lddtWeights.act0Bias.length;
  const lddtBins = lddtWeights.logitsBias.length;
  const paeBins = paeWeights.logitsBias.length;
  if (![structureChannels, hiddenChannels, lddtBins, paeBins]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("confidence channel dimensions must be positive safe integers");
  }
  if (paeWeights.logitsWeight.length % paeBins !== 0) {
    throw new RangeError("invalid PAE projection weight shape");
  }
  const pairChannels = paeWeights.logitsWeight.length / paeBins;
  const expectedPairElements = length * length * pairChannels;
  if (!Number.isSafeInteger(expectedPairElements)
    || (pairBuffer === undefined && pairRepresentation.length !== expectedPairElements)
    || (pairBuffer !== undefined && pairRepresentation.length !== 0
      && pairRepresentation.length !== expectedPairElements)) {
    throw new RangeError("invalid pair representation shape");
  }
  const expectedWeights: ReadonlyArray<readonly [string, Float32Array, number]> = [
    ["lddt norm scale", lddtWeights.normScale, structureChannels],
    ["lddt norm offset", lddtWeights.normOffset, structureChannels],
    ["lddt act0 weight", lddtWeights.act0Weight, structureChannels * hiddenChannels],
    ["lddt act1 weight", lddtWeights.act1Weight, hiddenChannels * hiddenChannels],
    ["lddt act1 bias", lddtWeights.act1Bias, hiddenChannels],
    ["lddt logits weight", lddtWeights.logitsWeight, hiddenChannels * lddtBins],
    ["PAE logits weight", paeWeights.logitsWeight, pairChannels * paeBins],
  ];
  for (const [name, value, expected] of expectedWeights) {
    if (value.length !== expected) throw new RangeError(`${name} has ${value.length} values; expected ${expected}`);
  }
  if (breaks.length !== paeBins - 1 || breaks.length < 2
    || breaks.some((value, index) => !Number.isFinite(value)
      || (index > 0 && value <= breaks[index - 1]!))) {
    throw new RangeError(`PAE breaks must contain ${paeBins - 1} increasing finite values`);
  }
  return { structureChannels, pairChannels, hiddenChannels, lddtBins, paeBins };
}

export class ConfidenceHeadsGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;

  constructor(device: GPUDevice) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(
    structureRepresentation: Float32Array,
    pairRepresentation: Float32Array,
    length: number,
    lddtWeights: PredictedLddtWeights,
    paeWeights: PredictedAlignedErrorWeights,
    breaks: Float32Array = Float32Array.from({ length: 63 }, (_, index) => index * 0.5),
    pairBuffer?: GPUBuffer,
  ): Promise<ConfidenceResult> {
    return this.#run(structureRepresentation, pairRepresentation, length, lddtWeights, paeWeights,
      breaks, pairBuffer, undefined) as Promise<ConfidenceResult>;
  }

  /**
   * Production confidence path with bounded PAE logits and GPU-side softmax
   * expectations. All windows are encoded into one GPU submission.
   */
  async runReduced(
    structureRepresentation: Float32Array,
    pairRepresentation: Float32Array,
    length: number,
    lddtWeights: PredictedLddtWeights,
    paeWeights: PredictedAlignedErrorWeights,
    breaks: Float32Array = Float32Array.from({ length: 63 }, (_, index) => index * 0.5),
    options: ReducedConfidenceOptions = {},
  ): Promise<ReducedConfidenceResult> {
    return this.#run(structureRepresentation, pairRepresentation, length, lddtWeights, paeWeights,
      breaks, options.pairBuffer, options.maxPaeLogitsBytes ?? PAE_LOGITS_WINDOW_BYTES
    ) as Promise<ReducedConfidenceResult>;
  }

  async #run(
    structureRepresentation: Float32Array,
    pairRepresentation: Float32Array,
    length: number,
    lddtWeights: PredictedLddtWeights,
    paeWeights: PredictedAlignedErrorWeights,
    breaks: Float32Array,
    pairBuffer: GPUBuffer | undefined,
    maxPaeLogitsBytes: number | undefined,
  ): Promise<ConfidenceResult | ReducedConfidenceResult> {
    const reduced = maxPaeLogitsBytes !== undefined;
    const { structureChannels, pairChannels, hiddenChannels, lddtBins, paeBins }
      = validateConfidenceInputs(
        structureRepresentation, pairRepresentation, length, lddtWeights, paeWeights, breaks, pairBuffer,
      );
    const centers = paeCenters(breaks);
    const effectiveLength = Math.max(length, 19);
    const d0 = 1.24 * Math.cbrt(effectiveLength - 15) - 1.8;
    const tmPerBin = centers.map((center) => 1 / (1 + center * center / (d0 * d0)));
    const tensors: readonly Float32Array[] = [
      lddtWeights.normScale, lddtWeights.normOffset, lddtWeights.act0Weight,
      lddtWeights.act0Bias, lddtWeights.act1Weight, lddtWeights.act1Bias, lddtWeights.logitsWeight,
      lddtWeights.logitsBias, paeWeights.logitsWeight, paeWeights.logitsBias,
      ...(reduced ? [centers, tmPerBin] : []),
    ];
    const offsets: number[] = [];
    let packedSize = 0;
    for (const tensor of tensors) { offsets.push(packedSize); packedSize += tensor.length; }
    const packed = new Float32Array(packedSize);
    tensors.forEach((tensor, index) => packed.set(tensor, offsets[index]));
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    const upload = (label: string, data: ArrayBufferView, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.upload(label, data, usage));
    const allocate = (label: string, elements: number, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    try {
      const [linear, normalize, relu, expectation] = await Promise.all([
        this.pipelines.get("confidence:linear", LINEAR_SHADER),
        this.pipelines.get("confidence:normalize", ATTENTION_NORMALIZE_SHADER),
        this.pipelines.get("confidence:relu", RELU_SHADER),
        reduced
          ? this.pipelines.get("confidence:pae-expectation", PAE_EXPECTATION_SHADER)
          : Promise.resolve(undefined),
      ]);
      const structure = upload("confidence.structure", structureRepresentation);
      const pair = pairBuffer === undefined
        ? upload("confidence.pair", pairRepresentation)
        : { buffer: pairBuffer };
      const weights = upload("confidence.weights", packed);
      const normParams = upload("confidence.norm-params", createAttentionNormParameters(
        length, structureChannels, offsets[0]!, offsets[1]!, false, 1, length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const linearParams = (label: string, rows: number, inner: number, columns: number, weight: number, bias: number) =>
        upload(label, new Uint32Array([rows, inner, columns, weight, bias, 0, 0, 0]), GPUBufferUsage.UNIFORM);
      const params = [
        linearParams("confidence.act0-params", length, structureChannels, hiddenChannels, offsets[2]!, offsets[3]!),
        linearParams("confidence.act1-params", length, hiddenChannels, hiddenChannels, offsets[4]!, offsets[5]!),
        linearParams("confidence.lddt-params", length, hiddenChannels, lddtBins, offsets[6]!, offsets[7]!),
      ];
      const normalized = allocate("confidence.normalized", length * structureChannels);
      const act0Raw = allocate("confidence.act0-raw", length * hiddenChannels);
      const act0 = allocate("confidence.act0", length * hiddenChannels);
      const act1Raw = allocate("confidence.act1-raw", length * hiddenChannels);
      const act1 = allocate("confidence.act1", length * hiddenChannels);
      const lddtLogits = allocate("confidence.lddt-logits", length * lddtBins,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const encoder = this.device.createCommandEncoder({
        label: reduced ? "confidence-heads-reduced" : "confidence-heads",
      });
      interface Binding { readonly buffer: GPUBuffer; readonly offset?: number; readonly size?: number }
      const dispatch = (pipeline: GPUComputePipeline,
        buffers: readonly Binding[], x: number, y = 1) => {
        const pass = encoder.beginComputePass(); pass.setPipeline(pipeline);
        pass.setBindGroup(0, this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: buffers.map((buffer, binding) => ({ binding, resource: {
            buffer: buffer.buffer,
            ...(buffer.offset === undefined ? {} : { offset: buffer.offset }),
            ...(buffer.size === undefined ? {} : { size: buffer.size }),
          } })),
        }));
        pass.dispatchWorkgroups(x, y); pass.end();
      };
      const linearDispatch = (source: Binding, parameter: AllocatedGpuBuffer,
        output: AllocatedGpuBuffer, rows: number, columns: number) =>
        dispatch(linear, [source, weights, parameter, output],
          Math.ceil(columns / TRANSITION_TILE_COLUMNS), Math.ceil(rows / TRANSITION_TILE_ROWS));
      dispatch(normalize, [structure, weights, normParams, normalized], length);
      linearDispatch(normalized, params[0]!, act0Raw, length, hiddenChannels);
      dispatch(relu, [act0Raw, act0], Math.ceil(act0Raw.byteLength / 4 / 64));
      linearDispatch(act0, params[1]!, act1Raw, length, hiddenChannels);
      dispatch(relu, [act1Raw, act1], Math.ceil(act1Raw.byteLength / 4 / 64));
      linearDispatch(act1, params[2]!, lddtLogits, length, lddtBins);

      let paeLogits: AllocatedGpuBuffer | undefined;
      let predictedAlignedErrorGpu: AllocatedGpuBuffer | undefined;
      let tmScoreTermsGpu: AllocatedGpuBuffer | undefined;
      if (!reduced) {
        paeLogits = allocate("confidence.pae-logits", length * length * paeBins,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const paeParams = linearParams(
          "confidence.pae-params", length * length, pairChannels, paeBins, offsets[8]!, offsets[9]!,
        );
        linearDispatch(pair, paeParams, paeLogits, length * length, paeBins);
      } else {
        const totalRows = length * length;
        const windowRows = paeWindowRows(
          totalRows, pairChannels, paeBins, this.device.limits.maxStorageBufferBindingSize,
          this.device.limits.minStorageBufferOffsetAlignment, maxPaeLogitsBytes!,
        );
        const paeLogitsWindow = allocate("confidence.pae-logits-window", windowRows * paeBins);
        predictedAlignedErrorGpu = allocate(
          "confidence.predicted-aligned-error", totalRows,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        );
        tmScoreTermsGpu = allocate(
          "confidence.tm-score-terms", totalRows,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        );
        for (let rowOffset = 0, window = 0; rowOffset < totalRows;
          rowOffset += windowRows, window += 1) {
          const rows = Math.min(windowRows, totalRows - rowOffset);
          const paeParams = linearParams(
            `confidence.pae-params-${window}`, rows, pairChannels, paeBins, offsets[8]!, offsets[9]!,
          );
          linearDispatch({
            buffer: pair.buffer,
            offset: rowOffset * pairChannels * Float32Array.BYTES_PER_ELEMENT,
            size: rows * pairChannels * Float32Array.BYTES_PER_ELEMENT,
          }, paeParams, paeLogitsWindow, rows, paeBins);
          const expectationParams = upload(
            `confidence.pae-expectation-params-${window}`,
            new Uint32Array([rows, paeBins, rowOffset, offsets[10]!, offsets[11]!, 0, 0, 0]),
            GPUBufferUsage.UNIFORM,
          );
          dispatch(expectation!, [paeLogitsWindow, weights, expectationParams,
            predictedAlignedErrorGpu, tmScoreTermsGpu], Math.ceil(rows / 64));
        }
      }
      const readbackSources = reduced
        ? [lddtLogits, predictedAlignedErrorGpu!, tmScoreTermsGpu!]
        : [lddtLogits, paeLogits!];
      const readbacks = readbackSources.map((source, index) => {
        const target = allocate(`confidence.readback-${index}`, source.byteLength / 4,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        encoder.copyBufferToBuffer(source.buffer, 0, target.buffer, 0, source.byteLength); return target;
      });
      this.device.queue.submit([encoder.finish()]);
      await Promise.all(readbacks.map((buffer) => buffer.buffer.mapAsync(GPUMapMode.READ)));
      const values = readbacks.map((buffer) => {
        const value = new Float32Array(buffer.buffer.getMappedRange().slice(0)); buffer.buffer.unmap(); return value;
      });
      const lddtLogitValues = values[0]!;
      const lddtCenters = Float32Array.from({ length: lddtBins }, (_, index) => (index + 0.5) / lddtBins * 100);
      const plddt = softmaxExpected(lddtLogitValues, length, lddtBins, lddtCenters);
      const common = {
        plddt,
        meanPlddt: plddt.reduce((sum, value) => sum + value, 0) / length,
        maxPredictedAlignedError: centers[paeBins - 1]!,
        memory: this.allocator.snapshot(),
      };
      if (reduced) {
        const predictedAlignedError = values[1]!;
        const tmScoreTerms = values[2]!;
        return {
          ...common, predictedAlignedError, tmScoreTerms,
          ptm: predictedTmScoreFromExpected(tmScoreTerms, length),
        };
      }
      const paeLogitValues = values[1]!;
      return {
        ...common, lddtLogits: lddtLogitValues, paeLogits: paeLogitValues,
        predictedAlignedError: softmaxExpected(paeLogitValues, length * length, paeBins, centers),
        ptm: predictedTmScore(paeLogitValues, length, breaks),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index]!.release();
    }
  }
}
