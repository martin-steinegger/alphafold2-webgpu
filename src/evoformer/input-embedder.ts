import {
  ATTENTION_NORMALIZE_IN_PLACE_SHADER, ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters,
} from "./attention.js";
import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "../runtime/allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "../runtime/pipeline-cache.js";
import { WebGpuExecution, type GpuTensor } from "../runtime/execution.js";

export interface InputEmbedderWeights {
  readonly preprocess1dWeight: Float32Array;
  readonly preprocess1dBias: Float32Array;
  readonly preprocessMsaWeight: Float32Array;
  readonly preprocessMsaBias: Float32Array;
  readonly leftSingleWeight: Float32Array;
  readonly leftSingleBias: Float32Array;
  readonly rightSingleWeight: Float32Array;
  readonly rightSingleBias: Float32Array;
  readonly previousPositionWeight: Float32Array;
  readonly previousPositionBias: Float32Array;
  readonly previousMsaNormScale: Float32Array;
  readonly previousMsaNormOffset: Float32Array;
  readonly previousPairNormScale: Float32Array;
  readonly previousPairNormOffset: Float32Array;
  readonly relativePositionWeight: Float32Array;
  readonly relativePositionBias: Float32Array;
  readonly extraMsaWeight: Float32Array;
  readonly extraMsaBias: Float32Array;
}

export interface InputEmbedderInput {
  readonly targetFeatures: Float32Array;
  readonly msaFeatures: Float32Array;
  readonly extraMsa: Float32Array;
  readonly extraHasDeletion: Float32Array;
  readonly extraDeletionValue: Float32Array;
  readonly residueIndex: Float32Array;
  readonly aatype: Float32Array;
  readonly previousMsaFirstRow: Float32Array;
  readonly previousPair: Float32Array;
  readonly previousPositions: Float32Array;
  readonly length: number;
  readonly msaSequences: number;
  readonly extraSequences: number;
  readonly targetChannels: number;
  readonly msaFeatureChannels: number;
  readonly msaChannels: number;
  readonly pairChannels: number;
  readonly extraMsaChannels: number;
  readonly weights: InputEmbedderWeights;
  /** Enables AlphaFold-Multimer's 73-channel chain-relative position encoding. */
  readonly chainRelative?: {
    readonly asymId: Float32Array;
    readonly entityId: Float32Array;
    readonly symId: Float32Array;
  };
}

export interface InputEmbedderResult {
  readonly msa: Float32Array;
  readonly pairWithoutTemplates: Float32Array;
  readonly extraMsa: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

const GRID_WIDTH = 32_768;
const ceilDivide = (value: number, divisor: number): number => Math.ceil(value / divisor);

function packWeights(input: InputEmbedderInput): { data: Float32Array; offsets: readonly number[] } {
  const w = input.weights;
  const tensors = [
    w.preprocess1dWeight, w.preprocess1dBias, w.preprocessMsaWeight, w.preprocessMsaBias,
    w.leftSingleWeight, w.leftSingleBias, w.rightSingleWeight, w.rightSingleBias,
    w.previousPositionWeight, w.previousPositionBias,
    w.previousMsaNormScale, w.previousMsaNormOffset,
    w.previousPairNormScale, w.previousPairNormOffset,
    w.relativePositionWeight, w.relativePositionBias,
    w.extraMsaWeight, w.extraMsaBias,
  ] as const;
  const offsets: number[] = [];
  let size = 0;
  for (const tensor of tensors) { offsets.push(size); size += tensor.length; }
  const data = new Float32Array(size);
  tensors.forEach((tensor, index) => data.set(tensor, offsets[index]));
  return { data, offsets };
}

function parameters(input: InputEmbedderInput, offsets: readonly number[]): Uint8Array {
  const buffer = new ArrayBuffer(128);
  const view = new DataView(buffer);
  const dimensions = [
    input.length, input.msaSequences, input.extraSequences, input.targetChannels,
    input.msaFeatureChannels, input.msaChannels, input.pairChannels, input.extraMsaChannels,
    25, 15, 32,
  ];
  [...dimensions, ...offsets].forEach((value, index) => view.setUint32(index * 4, value!, true));
  view.setFloat32(116, 3.25, true);
  view.setFloat32(120, 20.75, true);
  view.setUint32(124, input.chainRelative === undefined ? 0 : 1, true);
  return new Uint8Array(buffer);
}

const COMMON = `
struct Parameters {
  length: u32, msa_sequences: u32, extra_sequences: u32, target_channels: u32,
  msa_feature_channels: u32, msa_channels: u32, pair_channels: u32, extra_channels: u32,
  extra_feature_channels: u32, dgram_bins: u32, max_relative: u32,
  preprocess_1d_weight: u32, preprocess_1d_bias: u32,
  preprocess_msa_weight: u32, preprocess_msa_bias: u32,
  left_weight: u32, left_bias: u32, right_weight: u32, right_bias: u32,
  previous_position_weight: u32, previous_position_bias: u32,
  previous_msa_scale: u32, previous_msa_offset: u32,
  previous_pair_scale: u32, previous_pair_offset: u32,
  relative_weight: u32, relative_bias: u32,
  extra_weight: u32, extra_bias: u32,
  min_bin: f32, max_bin: f32, chain_relative: u32,
};
const GRID_WIDTH: u32 = 32768u;
`;

const MSA_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> target_features: array<f32>;
@group(0) @binding(1) var<storage, read> msa_features: array<f32>;
@group(0) @binding(2) var<storage, read> previous_msa: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.msa_sequences * p.length * p.msa_channels) { return; }
  let channel = index % p.msa_channels;
  let row = index / p.msa_channels;
  let residue = row % p.length;
  let sequence = row / p.length;
  var result = weights[p.preprocess_1d_bias + channel] + weights[p.preprocess_msa_bias + channel];
  for (var c = 0u; c < p.target_channels; c += 1u) {
    result += target_features[residue * p.target_channels + c]
      * weights[p.preprocess_1d_weight + c * p.msa_channels + channel];
  }
  for (var c = 0u; c < p.msa_feature_channels; c += 1u) {
    result += msa_features[row * p.msa_feature_channels + c]
      * weights[p.preprocess_msa_weight + c * p.msa_channels + channel];
  }
  if (sequence == 0u) { result += previous_msa[residue * p.msa_channels + channel]; }
  output[index] = result;
}`;

const EXTRA_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> extra_msa: array<f32>;
@group(0) @binding(1) var<storage, read> has_deletion: array<f32>;
@group(0) @binding(2) var<storage, read> deletion_value: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.extra_sequences * p.length * p.extra_channels) { return; }
  let channel = index % p.extra_channels;
  let row = index / p.extra_channels;
  let code = u32(extra_msa[row]);
  var result = weights[p.extra_bias + channel];
  result += weights[p.extra_weight + code * p.extra_channels + channel];
  result += has_deletion[row] * weights[p.extra_weight + 23u * p.extra_channels + channel];
  result += deletion_value[row] * weights[p.extra_weight + 24u * p.extra_channels + channel];
  output[index] = result;
}`;

const PAIR_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> target_features: array<f32>;
@group(0) @binding(1) var<storage, read> previous_pair: array<f32>;
@group(0) @binding(2) var<storage, read> previous_positions: array<f32>;
@group(0) @binding(3) var<storage, read> aatype: array<f32>;
@group(0) @binding(4) var<storage, read> residue_index: array<f32>;
@group(0) @binding(5) var<storage, read> chain_ids: array<f32>;
@group(0) @binding(6) var<storage, read> weights: array<f32>;
@group(0) @binding(7) var<uniform> p: Parameters;
@group(0) @binding(8) var<storage, read_write> output: array<f32>;

fn pseudo_beta_coordinate(residue: u32, coordinate: u32) -> f32 {
  let atom = select(3u, 1u, u32(aatype[residue]) == 7u);
  return previous_positions[(residue * 37u + atom) * 3u + coordinate];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.length * p.length * p.pair_channels) { return; }
  let channel = index % p.pair_channels;
  let pair = index / p.pair_channels;
  let i = pair / p.length;
  let j = pair % p.length;
  var result = weights[p.left_bias + channel] + weights[p.right_bias + channel];
  for (var c = 0u; c < p.target_channels; c += 1u) {
    result += target_features[i * p.target_channels + c] * weights[p.left_weight + c * p.pair_channels + channel];
    result += target_features[j * p.target_channels + c] * weights[p.right_weight + c * p.pair_channels + channel];
  }
  result += weights[p.previous_position_bias + channel];
  var distance_squared = 0.0;
  for (var coordinate = 0u; coordinate < 3u; coordinate += 1u) {
    let delta = pseudo_beta_coordinate(i, coordinate) - pseudo_beta_coordinate(j, coordinate);
    distance_squared += delta * delta;
  }
  let bin_width = (p.max_bin - p.min_bin) / f32(p.dgram_bins - 1u);
  for (var bin = 0u; bin < p.dgram_bins; bin += 1u) {
    let lower = p.min_bin + f32(bin) * bin_width;
    let upper = select(p.max_bin + bin_width, p.min_bin + f32(bin + 1u) * bin_width, bin + 1u < p.dgram_bins);
    if (distance_squared > lower * lower && (bin + 1u == p.dgram_bins || distance_squared < upper * upper)) {
      result += weights[p.previous_position_weight + bin * p.pair_channels + channel];
    }
  }
  result += previous_pair[index];
  result += weights[p.relative_bias + channel];
  let raw_offset = i32(residue_index[i]) - i32(residue_index[j]) + i32(p.max_relative);
  var relative = u32(clamp(raw_offset, 0, i32(2u * p.max_relative)));
  if (p.chain_relative != 0u) {
    let same_chain = u32(chain_ids[i]) == u32(chain_ids[j]);
    if (!same_chain) { relative = 2u * p.max_relative + 1u; }
    result += weights[p.relative_weight + relative * p.pair_channels + channel];
    let same_entity = u32(chain_ids[p.length + i]) == u32(chain_ids[p.length + j]);
    if (same_entity) { result += weights[p.relative_weight + 66u * p.pair_channels + channel]; }
    var relative_chain = 5u;
    if (same_entity) {
      relative_chain = u32(clamp(i32(chain_ids[2u * p.length + i])
        - i32(chain_ids[2u * p.length + j]) + 2, 0, 4));
    }
    result += weights[p.relative_weight + (67u + relative_chain) * p.pair_channels + channel];
  } else {
    result += weights[p.relative_weight + relative * p.pair_channels + channel];
  }
  output[index] = result;
}`;

/**
 * The pair embedding written over the recycled pair it reads. Each invocation
 * reads `previous_pair` only at the element it stores, so a single read-write
 * binding is exact and one pair-shaped tensor serves as input and output.
 */
const PAIR_IN_PLACE_SHADER = PAIR_SHADER
  .replace("var<storage, read> previous_pair", "var<storage, read_write> previous_pair")
  .replace("@group(0) @binding(8) var<storage, read_write> output: array<f32>;\n", "")
  .replace(/\boutput\[/g, "previous_pair[");

function validateChainRelative(input: InputEmbedderInput): void {
  const chain = input.chainRelative;
  const relativeChannels = chain === undefined ? 65 : 73;
  if (input.weights.relativePositionWeight.length !== relativeChannels * input.pairChannels) {
    throw new RangeError(`relative position weights must have shape [${relativeChannels}, pairChannels]`);
  }
  if (chain === undefined) return;
  for (const [name, values] of [
    ["asymId", chain.asymId], ["entityId", chain.entityId], ["symId", chain.symId],
  ] as const) {
    if (values.length !== input.length || values.some((value) => !Number.isSafeInteger(value) || value <= 0)) {
      throw new RangeError(`${name} must contain one positive integer per residue`);
    }
  }
}

function packedChainIdentifiers(input: InputEmbedderInput): Float32Array {
  const output = new Float32Array(input.length * 3);
  if (input.chainRelative !== undefined) {
    output.set(input.chainRelative.asymId);
    output.set(input.chainRelative.entityId, input.length);
    output.set(input.chainRelative.symId, input.length * 2);
  }
  return output;
}

export interface EncodedInputEmbedder {
  /** The new pair, written in place over the `previousPair` tensor passed in. */
  readonly pairWithoutTemplates: GpuTensor;
  readonly extraMsa: GpuTensor;
  /** Inputs that may be pooled once the pair and extra-MSA command buffer is submitted. */
  readonly temporaries: readonly GpuTensor[];
  /**
   * Encodes the clustered-MSA embedding. Nothing in the extra-MSA stack reads
   * it, so a caller may defer this until that stack has finished and keep the
   * largest tensor of the prediction out of the extra stack's peak. It
   * normalizes `previousMsa` in place; release it and `msaTemporaries` only
   * after the returned command buffer is submitted.
   */
  readonly encodeMsa: (encoder: GPUCommandEncoder) => GpuTensor;
  readonly msaTemporaries: readonly GpuTensor[];
}

/**
 * Encode the input embedding into an existing execution without crossing the
 * CPU boundary. `previousPair` and `previousMsa` are consumed: the new pair is
 * written over `previousPair`, so callers must not release it separately.
 */
export async function encodeInputEmbedder(
  execution: WebGpuExecution,
  encoder: GPUCommandEncoder,
  input: InputEmbedderInput,
  previousMsa: GpuTensor,
  previousPair: GpuTensor,
  previousPositions: GpuTensor,
): Promise<EncodedInputEmbedder> {
  validateChainRelative(input);
  const expectedPreviousMsa = input.length * input.msaChannels;
  const expectedPreviousPair = input.length * input.length * input.pairChannels;
  const expectedPreviousPositions = input.length * 37 * 3;
  if (previousMsa.elements < expectedPreviousMsa || previousPair.elements !== expectedPreviousPair
      || previousPositions.elements !== expectedPreviousPositions) {
    throw new RangeError("resident recycle tensor shape mismatch");
  }
  const packed = packWeights(input);
  const [normalize, msaPipeline, pairPipeline, extraPipeline] = await Promise.all([
    execution.pipelines.get("embed:normalize-in-place", ATTENTION_NORMALIZE_IN_PLACE_SHADER),
    execution.pipelines.get("embed:msa", MSA_SHADER),
    execution.pipelines.get("embed:pair-in-place", PAIR_IN_PLACE_SHADER),
    execution.pipelines.get("embed:extra", EXTRA_SHADER),
  ]);
  const temporaries: GpuTensor[] = [];
  const temporaryUpload = (label: string, data: ArrayBufferView, usage = GPUBufferUsage.STORAGE): GpuTensor => {
    const tensor = execution.upload(label, data, usage); temporaries.push(tensor); return tensor;
  };
  const temporaryAllocate = (label: string, elements: number): GpuTensor => {
    const tensor = execution.allocate(label, elements); temporaries.push(tensor); return tensor;
  };
  const target = temporaryUpload("embed.target", input.targetFeatures);
  const msaFeatures = temporaryUpload("embed.msa-features", input.msaFeatures);
  const extraMsaInput = temporaryUpload("embed.extra-codes", input.extraMsa);
  const hasDeletion = temporaryUpload("embed.extra-has-deletion", input.extraHasDeletion);
  const deletionValue = temporaryUpload("embed.extra-deletion-value", input.extraDeletionValue);
  const residueIndex = temporaryUpload("embed.residue-index", input.residueIndex);
  const aatype = temporaryUpload("embed.aatype", input.aatype);
  const chainIds = temporaryUpload("embed.chain-identifiers", packedChainIdentifiers(input));
  const weights = temporaryUpload("embed.weights", packed.data);
  const params = temporaryUpload("embed.parameters", parameters(input, packed.offsets), GPUBufferUsage.UNIFORM);
  const previousMsaNormParams = temporaryUpload("embed.previous-msa-norm-params", createAttentionNormParameters(
    input.length, input.msaChannels, packed.offsets[10]!, packed.offsets[11]!,
    false, 1, input.length, 1e-5,
  ), GPUBufferUsage.UNIFORM);
  const previousPairNormParams = temporaryUpload("embed.previous-pair-norm-params", createAttentionNormParameters(
    input.length * input.length, input.pairChannels, packed.offsets[12]!, packed.offsets[13]!,
    false, 1, input.length * input.length, 1e-5,
  ), GPUBufferUsage.UNIFORM);
  const msaElements = input.msaSequences * input.length * input.msaChannels;
  const pairElements = expectedPreviousPair;
  const extraElements = input.extraSequences * input.length * input.extraMsaChannels;
  // The recycled pair is consumed only here, so both the normalization and the
  // new pair overwrite it in place: LayerNorm finishes every read of a row
  // before any invocation writes its own elements, and the pair kernel reads
  // exactly the element it stores. This keeps one pair-shaped tensor live at
  // the embedder instead of three.
  const pair = previousPair;
  const extra = execution.allocate("embed.extra", extraElements);
  let grid = execution.linearGrid(input.length * input.length, 1);
  execution.dispatch(encoder, normalize, [previousPair, weights, previousPairNormParams], grid[0], grid[1]);
  grid = execution.linearGrid(pairElements);
  execution.dispatch(encoder, pairPipeline,
    [target, previousPair, previousPositions, aatype, residueIndex, chainIds, weights, params],
    grid[0], grid[1]);
  grid = execution.linearGrid(extraElements);
  execution.dispatch(encoder, extraPipeline, [extraMsaInput, hasDeletion, deletionValue, weights, params, extra],
    grid[0], grid[1]);
  const msaTemporaries = [target, msaFeatures, weights, params, previousMsaNormParams];
  const pairTemporaries = temporaries.filter((tensor) => !msaTemporaries.includes(tensor));
  const encodeMsa = (msaEncoder: GPUCommandEncoder): GpuTensor => {
    const msa = execution.allocate("embed.msa", msaElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    let msaGrid = execution.linearGrid(input.length, 1);
    execution.dispatch(msaEncoder, normalize, [previousMsa, weights, previousMsaNormParams], msaGrid[0], msaGrid[1]);
    msaGrid = execution.linearGrid(msaElements);
    execution.dispatch(msaEncoder, msaPipeline, [target, msaFeatures, previousMsa, weights, params, msa],
      msaGrid[0], msaGrid[1]);
    return msa;
  };
  return { pairWithoutTemplates: pair, extraMsa: extra, temporaries: pairTemporaries, msaTemporaries, encodeMsa };
}

export class InputEmbedderGpu {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  constructor(device: GPUDevice) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device);
    this.pipelines = pipelineCacheForDevice(device);
  }

  async run(input: InputEmbedderInput): Promise<InputEmbedderResult> {
    validateChainRelative(input);
    const packed = packWeights(input);
    const [normalize, msaPipeline, pairPipeline, extraPipeline] = await Promise.all([
      this.pipelines.get("embed:normalize", ATTENTION_NORMALIZE_SHADER),
      this.pipelines.get("embed:msa", MSA_SHADER),
      this.pipelines.get("embed:pair", PAIR_SHADER),
      this.pipelines.get("embed:extra", EXTRA_SHADER),
    ]);
    const allocations: AllocatedGpuBuffer[] = [];
    const keep = (value: AllocatedGpuBuffer): AllocatedGpuBuffer => { allocations.push(value); return value; };
    const upload = (label: string, value: ArrayBufferView, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.upload(label, value, usage));
    const allocate = (label: string, elements: number, usage = GPUBufferUsage.STORAGE): AllocatedGpuBuffer =>
      keep(this.allocator.allocate(label, elements * 4, usage));
    const grid = (elements: number): readonly [number, number] => {
      const groups = ceilDivide(elements, 64);
      return [Math.min(groups, GRID_WIDTH), ceilDivide(groups, GRID_WIDTH)];
    };
    const rowGrid = (rows: number): readonly [number, number] => [
      Math.min(rows, GRID_WIDTH), ceilDivide(rows, GRID_WIDTH),
    ];
    try {
      const target = upload("embed.target", input.targetFeatures);
      const msaFeatures = upload("embed.msa-features", input.msaFeatures);
      const extraMsaInput = upload("embed.extra-codes", input.extraMsa);
      const hasDeletion = upload("embed.extra-has-deletion", input.extraHasDeletion);
      const deletionValue = upload("embed.extra-deletion-value", input.extraDeletionValue);
      const residueIndex = upload("embed.residue-index", input.residueIndex);
      const aatype = upload("embed.aatype", input.aatype);
      const chainIds = upload("embed.chain-identifiers", packedChainIdentifiers(input));
      const previousMsa = upload("embed.previous-msa", input.previousMsaFirstRow);
      const previousPair = upload("embed.previous-pair", input.previousPair);
      const previousPositions = upload("embed.previous-positions", input.previousPositions);
      const weights = upload("embed.weights", packed.data);
      const params = upload("embed.parameters", parameters(input, packed.offsets), GPUBufferUsage.UNIFORM);
      const previousMsaNormParams = upload("embed.previous-msa-norm-params", createAttentionNormParameters(
        input.length, input.msaChannels, packed.offsets[10]!, packed.offsets[11]!,
        false, 1, input.length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const previousPairNormParams = upload("embed.previous-pair-norm-params", createAttentionNormParameters(
        input.length * input.length, input.pairChannels, packed.offsets[12]!, packed.offsets[13]!,
        false, 1, input.length * input.length, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const previousMsaNormalized = allocate("embed.previous-msa-normalized", input.length * input.msaChannels);
      const previousPairNormalized = allocate(
        "embed.previous-pair-normalized", input.length * input.length * input.pairChannels,
      );
      const msaElements = input.msaSequences * input.length * input.msaChannels;
      const pairElements = input.length * input.length * input.pairChannels;
      const extraElements = input.extraSequences * input.length * input.extraMsaChannels;
      const msa = allocate("embed.msa", msaElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pair = allocate("embed.pair", pairElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const extra = allocate("embed.extra", extraElements, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaReadback = allocate("embed.msa-readback", msaElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const pairReadback = allocate("embed.pair-readback", pairElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const extraReadback = allocate("embed.extra-readback", extraElements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const encoder = this.device.createCommandEncoder({ label: "input-embedder" });
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
      let dispatch = rowGrid(input.length);
      pass(normalize, [previousMsa, weights, previousMsaNormParams, previousMsaNormalized],
        dispatch[0], dispatch[1]);
      dispatch = rowGrid(input.length * input.length);
      pass(normalize, [previousPair, weights, previousPairNormParams, previousPairNormalized],
        dispatch[0], dispatch[1]);
      dispatch = grid(msaElements);
      pass(msaPipeline, [target, msaFeatures, previousMsaNormalized, weights, params, msa], dispatch[0], dispatch[1]);
      dispatch = grid(pairElements);
      pass(pairPipeline, [target, previousPairNormalized, previousPositions, aatype, residueIndex,
        chainIds, weights, params, pair], dispatch[0], dispatch[1]);
      dispatch = grid(extraElements);
      pass(extraPipeline, [extraMsaInput, hasDeletion, deletionValue, weights, params, extra], dispatch[0], dispatch[1]);
      encoder.copyBufferToBuffer(msa.buffer, 0, msaReadback.buffer, 0, msaElements * 4);
      encoder.copyBufferToBuffer(pair.buffer, 0, pairReadback.buffer, 0, pairElements * 4);
      encoder.copyBufferToBuffer(extra.buffer, 0, extraReadback.buffer, 0, extraElements * 4);
      const start = performance.now();
      this.device.queue.submit([encoder.finish()]);
      const validationError = await this.device.popErrorScope();
      if (validationError !== null) throw new Error(`WebGPU validation failed: ${validationError.message}`);
      await Promise.all([msaReadback.buffer.mapAsync(GPUMapMode.READ), pairReadback.buffer.mapAsync(GPUMapMode.READ),
        extraReadback.buffer.mapAsync(GPUMapMode.READ)]);
      const msaOutput = new Float32Array(msaReadback.buffer.getMappedRange().slice(0));
      const pairOutput = new Float32Array(pairReadback.buffer.getMappedRange().slice(0));
      const extraOutput = new Float32Array(extraReadback.buffer.getMappedRange().slice(0));
      msaReadback.buffer.unmap(); pairReadback.buffer.unmap(); extraReadback.buffer.unmap();
      return {
        msa: msaOutput, pairWithoutTemplates: pairOutput, extraMsa: extraOutput,
        elapsedMilliseconds: performance.now() - start, memory: this.allocator.snapshot(),
      };
    } finally {
      for (let index = allocations.length - 1; index >= 0; index -= 1) allocations[index]!.release();
    }
  }
}
