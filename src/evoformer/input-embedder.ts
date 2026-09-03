import {
  ATTENTION_NORMALIZE_IN_PLACE_SHADER, ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters,
  createAttentionNormalizeInPlaceShader,
} from "./attention.js";
import { type ActivationStorage, storageArray, storageWords, storedElement } from "../runtime/storage.js";
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
  /** Clustered MSA in the compact layout of `src/input/msa-features.ts`. */
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
  /** Channels per row and position of `msaFeatures`; the compact layout has 27. */
  readonly msaFeatureChannels: number;
  readonly msaChannels: number;
  readonly pairChannels: number;
  readonly extraMsaChannels: number;
  readonly weights: InputEmbedderWeights;
  /** Storage of the MSA activations the embedder produces (GPU path only); `f16` halves them inexactly. */
  readonly msaStorage?: ActivationStorage;
  /** Storage of the pair this writes and of the recycled pair it reads. */
  readonly pairStorage?: ActivationStorage;
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
// x is the first element this dispatch writes and y how many it spans; the
// whole tensor may be past what one binding reaches.
@group(0) @binding(6) var<uniform> w: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local = id.x + id.y * GRID_WIDTH * 64u;
  if (local >= w.y) { return; }
  let index = w.x + local;
  let channel = index % p.msa_channels;
  let row = index / p.msa_channels;
  let residue = row % p.length;
  let sequence = row / p.length;
  var result = weights[p.preprocess_1d_bias + channel] + weights[p.preprocess_msa_bias + channel];
  for (var c = 0u; c < p.target_channels; c += 1u) {
    result += target_features[residue * p.target_channels + c]
      * weights[p.preprocess_1d_weight + c * p.msa_channels + channel];
  }
  // Compact features: a cluster code that indexes the same weight matrix in
  // place of a 23-way one-hot, then the deletion scalars, the profile and the
  // cluster's deletion mean. The terms the one-hot dropped were all zero.
  let feature_base = row * p.msa_feature_channels;
  let code = u32(msa_features[feature_base]);
  if (code < 23u) { result += weights[p.preprocess_msa_weight + code * p.msa_channels + channel]; }
  result += msa_features[feature_base + 1u]
    * weights[p.preprocess_msa_weight + 23u * p.msa_channels + channel];
  result += msa_features[feature_base + 2u]
    * weights[p.preprocess_msa_weight + 24u * p.msa_channels + channel];
  for (var c = 0u; c < 23u; c += 1u) {
    result += msa_features[feature_base + 3u + c]
      * weights[p.preprocess_msa_weight + (25u + c) * p.msa_channels + channel];
  }
  result += msa_features[feature_base + 26u]
    * weights[p.preprocess_msa_weight + 48u * p.msa_channels + channel];
  if (sequence == 0u) { result += previous_msa[residue * p.msa_channels + channel]; }
  output[local] = result;
}`;

function createMsaEmbedShader(storage: ActivationStorage): string {
  if (storage === "f32") return MSA_SHADER;
  // One invocation per pair of adjacent channels, packed into one word.
  return `${COMMON}
@group(0) @binding(0) var<storage, read> target_features: array<f32>;
@group(0) @binding(1) var<storage, read> msa_features: array<f32>;
@group(0) @binding(2) var<storage, read> previous_msa: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<u32>;
@group(0) @binding(6) var<uniform> w: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local_word = id.x + id.y * GRID_WIDTH * 64u;
  if (local_word >= w.y / 2u) { return; }
  let pair = w.x / 2u + local_word;
  let pairs_per_row = p.msa_channels / 2u;
  let channel = (pair % pairs_per_row) * 2u;
  let row = pair / pairs_per_row;
  let residue = row % p.length;
  let sequence = row / p.length;
  var result = vec2<f32>(weights[p.preprocess_1d_bias + channel] + weights[p.preprocess_msa_bias + channel],
    weights[p.preprocess_1d_bias + channel + 1u] + weights[p.preprocess_msa_bias + channel + 1u]);
  for (var c = 0u; c < p.target_channels; c += 1u) {
    let base = p.preprocess_1d_weight + c * p.msa_channels + channel;
    result += target_features[residue * p.target_channels + c] * vec2<f32>(weights[base], weights[base + 1u]);
  }
  let feature_base = row * p.msa_feature_channels;
  let code = u32(msa_features[feature_base]);
  if (code < 23u) {
    let one_hot = p.preprocess_msa_weight + code * p.msa_channels + channel;
    result += vec2<f32>(weights[one_hot], weights[one_hot + 1u]);
  }
  for (var c = 23u; c < 25u; c += 1u) {
    let base = p.preprocess_msa_weight + c * p.msa_channels + channel;
    result += msa_features[feature_base + c - 22u] * vec2<f32>(weights[base], weights[base + 1u]);
  }
  for (var c = 0u; c < 23u; c += 1u) {
    let base = p.preprocess_msa_weight + (25u + c) * p.msa_channels + channel;
    result += msa_features[feature_base + 3u + c] * vec2<f32>(weights[base], weights[base + 1u]);
  }
  let mean_base = p.preprocess_msa_weight + 48u * p.msa_channels + channel;
  result += msa_features[feature_base + 26u] * vec2<f32>(weights[mean_base], weights[mean_base + 1u]);
  if (sequence == 0u) {
    let base = residue * p.msa_channels + channel;
    result += vec2<f32>(previous_msa[base], previous_msa[base + 1u]);
  }
  output[local_word] = pack2x16float(result);
}`;
}


const EXTRA_SHADER = `${COMMON}
@group(0) @binding(0) var<storage, read> extra_msa: array<f32>;
@group(0) @binding(1) var<storage, read> has_deletion: array<f32>;
@group(0) @binding(2) var<storage, read> deletion_value: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform> w: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local = id.x + id.y * GRID_WIDTH * 64u;
  if (local >= w.y) { return; }
  let index = w.x + local;
  let channel = index % p.extra_channels;
  let row = index / p.extra_channels;
  let code = u32(extra_msa[row]);
  var result = weights[p.extra_bias + channel];
  result += weights[p.extra_weight + code * p.extra_channels + channel];
  result += has_deletion[row] * weights[p.extra_weight + 23u * p.extra_channels + channel];
  result += deletion_value[row] * weights[p.extra_weight + 24u * p.extra_channels + channel];
  output[local] = result;
}`;

function createExtraEmbedShader(storage: ActivationStorage): string {
  if (storage === "f32") return EXTRA_SHADER;
  return `${COMMON}
@group(0) @binding(0) var<storage, read> extra_msa: array<f32>;
@group(0) @binding(1) var<storage, read> has_deletion: array<f32>;
@group(0) @binding(2) var<storage, read> deletion_value: array<f32>;
@group(0) @binding(3) var<storage, read> weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> output: array<u32>;
@group(0) @binding(6) var<uniform> w: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local_word = id.x + id.y * GRID_WIDTH * 64u;
  if (local_word >= w.y / 2u) { return; }
  let pair = w.x / 2u + local_word;
  let pairs_per_row = p.extra_channels / 2u;
  let channel = (pair % pairs_per_row) * 2u;
  let row = pair / pairs_per_row;
  let code = u32(extra_msa[row]);
  var result = vec2<f32>(weights[p.extra_bias + channel], weights[p.extra_bias + channel + 1u]);
  let coded = p.extra_weight + code * p.extra_channels + channel;
  result += vec2<f32>(weights[coded], weights[coded + 1u]);
  let deletion = p.extra_weight + 23u * p.extra_channels + channel;
  result += has_deletion[row] * vec2<f32>(weights[deletion], weights[deletion + 1u]);
  let value = p.extra_weight + 24u * p.extra_channels + channel;
  result += deletion_value[row] * vec2<f32>(weights[value], weights[value + 1u]);
  output[local_word] = pack2x16float(result);
}`;
}


export function createPairShader(storage: ActivationStorage): string {
  const previous = storedElement(storage, "previous_pair", "local");
  const body = PAIR_SHADER
    .replace("var<storage, read> previous_pair: array<f32>", `var<storage, read> previous_pair: array<${storageArray(storage)}>`)
    .replace("result += previous_pair[local];", `result += ${previous};`);
  if (storage === "f32") return body;
  // One invocation per word: two adjacent channels of one pair, which is what
  // a packed store owns. The pair channel count is even, so a word never
  // straddles two pairs.
  return body
    .replace("var<storage, read_write> output: array<f32>", "var<storage, read_write> output: array<u32>")
    .replace(`@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local = id.x + id.y * GRID_WIDTH * 64u;
  if (local >= w.y) { return; }
  let index = w.x + local;
  output[local] = pair_element(index / p.pair_channels, index % p.pair_channels, local);
}`, `@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local_word = id.x + id.y * GRID_WIDTH * 64u;
  if (local_word >= w.y / 2u) { return; }
  let word = w.x / 2u + local_word;
  let channels = p.pair_channels / 2u;
  let pair = word / channels;
  let channel = (word % channels) * 2u;
  output[local_word] = pack2x16float(vec2<f32>(pair_element(pair, channel, local_word * 2u),
    pair_element(pair, channel + 1u, local_word * 2u + 1u)));
}`);
}

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
// x is the first pair element this dispatch covers and y how many it spans:
// the whole pair passes what one binding may reach, so it is written in
// windows, and every binding of it covers one window.
@group(0) @binding(9) var<uniform> w: vec4<u32>;

fn pseudo_beta_coordinate(residue: u32, coordinate: u32) -> f32 {
  let atom = select(3u, 1u, u32(aatype[residue]) == 7u);
  return previous_positions[(residue * 37u + atom) * 3u + coordinate];
}

fn pair_element(pair: u32, channel: u32, local: u32) -> f32 {
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
  result += previous_pair[local];
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
  return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local = id.x + id.y * GRID_WIDTH * 64u;
  if (local >= w.y) { return; }
  let index = w.x + local;
  output[local] = pair_element(index / p.pair_channels, index % p.pair_channels, local);
}`;

/**
 * The pair embedding written over the recycled pair it reads. Each invocation
 * reads `previous_pair` only at the element it stores, so a single read-write
 * binding is exact and one pair-shaped tensor serves as input and output.
 */
export function createPairInPlaceShader(storage: ActivationStorage): string {
  return createPairShader(storage)
    .replace("var<storage, read> previous_pair", "var<storage, read_write> previous_pair")
    .replace(`@group(0) @binding(8) var<storage, read_write> output: array<${storageArray(storage)}>;\n`, "")
    .replace("@group(0) @binding(9) var<uniform> w: vec4<u32>;", "@group(0) @binding(8) var<uniform> w: vec4<u32>;")
    .replace(/\boutput\[/g, "previous_pair[");
}

const PAIR_IN_PLACE_SHADER = createPairInPlaceShader("f32");

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
  const pairStorage = input.pairStorage ?? "f32";
  if (previousMsa.elements < expectedPreviousMsa
      || previousPair.elements !== storageWords(expectedPreviousPair, pairStorage)
      || previousPositions.elements !== expectedPreviousPositions) {
    throw new RangeError("resident recycle tensor shape mismatch");
  }
  const packed = packWeights(input);
  const storage = input.msaStorage ?? "f32";
  if (pairStorage === "f16" && input.pairChannels % 2 !== 0) {
    throw new RangeError("packed pair storage needs an even channel count");
  }
  if (storage === "f16" && (input.msaChannels % 2 !== 0 || input.extraMsaChannels % 2 !== 0)) {
    throw new RangeError("packed MSA storage needs even channel counts");
  }
  // The recycled pair and the recycled MSA row are normalized in place by the
  // same kernel but need not share a storage: the MSA row is always f32.
  const [normalizePair, normalizeMsa, msaPipeline, pairPipeline, extraPipeline] = await Promise.all([
    execution.pipelines.get(`embed:normalize-in-place:${pairStorage}`,
      createAttentionNormalizeInPlaceShader(pairStorage)),
    execution.pipelines.get("embed:normalize-in-place:f32", ATTENTION_NORMALIZE_IN_PLACE_SHADER),
    execution.pipelines.get(`embed:msa:${storage}`, createMsaEmbedShader(storage)),
    execution.pipelines.get(`embed:pair-in-place:${pairStorage}`, createPairInPlaceShader(pairStorage)),
    execution.pipelines.get(`embed:extra:${storage}`, createExtraEmbedShader(storage)),
  ]);
  const temporaries: GpuTensor[] = [];
  const temporaryUpload = (label: string, data: ArrayBufferView, usage = GPUBufferUsage.STORAGE): GpuTensor => {
    const tensor = execution.upload(label, data, usage); temporaries.push(tensor); return tensor;
  };
  const temporaryAllocate = (label: string, elements: number): GpuTensor => {
    const tensor = execution.allocate(label, elements); temporaries.push(tensor); return tensor;
  };
  const target = temporaryUpload("embed.target", input.targetFeatures);
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
  const extra = execution.allocate("embed.extra", storageWords(extraElements, storage));
  // Both passes over the pair are per pair row, and the whole pair outgrows
  // what one binding may cover, so both walk it in windows of whole rows.
  // Windows hold whole rows of a tensor and fit one binding.
  const windowElementsFor = (channels: number, tensorStorage: ActivationStorage): number =>
    Math.max(channels, Math.floor(execution.bindingLimitBytes
      / (channels * (tensorStorage === "f16" ? 2 : 4))) * channels);
  // One row is one pair's channels; there are length squared of them.
  const rowElements = input.pairChannels;
  const bindingRows = Math.max(1, Math.floor(
    execution.bindingLimitBytes
    / (rowElements * (pairStorage === "f16" ? 2 : 4))));
  let grid = execution.linearGrid(input.length * input.length, 1);
  for (let row = 0; row < input.length * input.length; row += bindingRows) {
    const rows = Math.min(bindingRows, input.length * input.length - row);
    const windowParams = rows === input.length * input.length ? previousPairNormParams
      : temporaryUpload(`embed.previous-pair-norm-params-${row}`, createAttentionNormParameters(
        rows, input.pairChannels, packed.offsets[12]!, packed.offsets[13]!, false, 1, rows, 1e-5,
      ), GPUBufferUsage.UNIFORM);
    const window = execution.view(previousPair, storageWords(row * rowElements, pairStorage),
      storageWords(rows * rowElements, pairStorage));
    grid = execution.linearGrid(rows, 1);
    execution.dispatch(encoder, normalizePair, [window, weights, windowParams], grid[0], grid[1],
      1, `embed.pair-normalize-${row}`);
  }
  // A packed pair is written a word at a time, two channels per invocation.
  const windowElements = bindingRows * rowElements;
  for (let offset = 0; offset < pairElements; offset += windowElements) {
    const count = Math.min(windowElements, pairElements - offset);
    const window = temporaryUpload(`embed.pair-window-${offset}`,
      new Uint32Array([offset, count, 0, 0]), GPUBufferUsage.UNIFORM);
    const pairWindow = execution.view(previousPair, storageWords(offset, pairStorage),
      storageWords(count, pairStorage));
    grid = execution.linearGrid(storageWords(count, pairStorage));
    execution.dispatch(encoder, pairPipeline,
      [target, pairWindow, previousPositions, aatype, residueIndex, chainIds, weights, params, window],
      grid[0], grid[1], 1, `embed.pair-${offset}`);
  }
  // Written in windows for the same reason as the pair.
  const extraWindow = windowElementsFor(input.extraMsaChannels, storage);
  for (let offset = 0; offset < extraElements; offset += extraWindow) {
    const count = Math.min(extraWindow, extraElements - offset);
    const bounds = temporaryUpload(`embed.extra-window-${offset}`,
      new Uint32Array([offset, count, 0, 0]), GPUBufferUsage.UNIFORM);
    const target = execution.view(extra, storageWords(offset, storage), storageWords(count, storage));
    grid = execution.linearGrid(storageWords(count, storage));
    execution.dispatch(encoder, extraPipeline,
      [extraMsaInput, hasDeletion, deletionValue, weights, params, target, bounds],
      grid[0], grid[1], 1, `embed.extra-${offset}`);
  }
  const msaTemporaries = [target, weights, params, previousMsaNormParams];
  const pairTemporaries = temporaries.filter((tensor) => !msaTemporaries.includes(tensor));
  const encodeMsa = (msaEncoder: GPUCommandEncoder): GpuTensor => {
    // The clustered features are read only here, after the extra stack has
    // run, so they are uploaded here rather than sitting on the device
    // through it.
    const msaFeatures = execution.upload("embed.msa-features", input.msaFeatures);
    msaTemporaries.push(msaFeatures);
    const msa = execution.allocate("embed.msa", storageWords(msaElements, storage),
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    let msaGrid = execution.linearGrid(input.length, 1);
    execution.dispatch(msaEncoder, normalizeMsa, [previousMsa, weights, previousMsaNormParams],
      msaGrid[0], msaGrid[1], 1, "embed.msa-normalize");
    const msaWindow = windowElementsFor(input.msaChannels, storage);
    for (let offset = 0; offset < msaElements; offset += msaWindow) {
      const count = Math.min(msaWindow, msaElements - offset);
      const bounds = execution.upload(`embed.msa-window-${offset}`,
        new Uint32Array([offset, count, 0, 0]), GPUBufferUsage.UNIFORM);
      msaTemporaries.push(bounds);
      const window = execution.view(msa, storageWords(offset, storage), storageWords(count, storage));
      msaGrid = execution.linearGrid(storageWords(count, storage));
      execution.dispatch(msaEncoder, msaPipeline,
        [target, msaFeatures, previousMsa, weights, params, window, bounds],
        msaGrid[0], msaGrid[1], 1, `embed.msa-${offset}`);
    }
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
      const msaWindow = upload("embed.msa-window", new Uint32Array([0, msaElements, 0, 0]),
        GPUBufferUsage.UNIFORM);
      pass(msaPipeline, [target, msaFeatures, previousMsaNormalized, weights, params, msa, msaWindow],
        dispatch[0], dispatch[1]);
      dispatch = grid(pairElements);
      const pairWindow = upload("embed.pair-window", new Uint32Array([0, pairElements, 0, 0]),
        GPUBufferUsage.UNIFORM);
      pass(pairPipeline, [target, previousPairNormalized, previousPositions, aatype, residueIndex,
        chainIds, weights, params, pair, pairWindow], dispatch[0], dispatch[1]);
      dispatch = grid(extraElements);
      const extraWindow = upload("embed.extra-window", new Uint32Array([0, extraElements, 0, 0]),
        GPUBufferUsage.UNIFORM);
      pass(extraPipeline, [extraMsaInput, hasDeletion, deletionValue, weights, params, extra, extraWindow],
        dispatch[0], dispatch[1]);
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
