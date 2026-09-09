import { encodeTemplatePairBlock, type TemplatePairBlockWeights } from "./block.js";
import { rowWindows } from "../runtime/sharded.js";
import { storageArray, storageWords, storedElement, type ActivationStorage } from "../runtime/storage.js";
import { type GpuTensor, WebGpuExecution } from "../runtime/execution.js";
import type { AllocationSnapshot } from "../runtime/allocator.js";
import type { TemplateMsaWeights } from "../input/template-msa-row.js";

export interface QueryOnlyTemplateWeights {
  /** embedding2d, the [88, 64] projection of the template pair features. */
  readonly embeddingWeight: Float32Array;
  readonly embeddingBias: Float32Array;
  readonly blockWeights: readonly TemplatePairBlockWeights[];
  readonly outputNormScale: Float32Array;
  readonly outputNormOffset: Float32Array;
  readonly valueWeight: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
  readonly heads: number;
  /**
   * template_single_embedding and template_projection, which turn a
   * template's torsion angles into an MSA row.
   *
   * Optional because they are siblings of the template module in AlphaFold
   * rather than part of it, so a bundle exported before templates were
   * supported carries the whole pair stack and not these. A prediction without
   * a template never reads them.
   */
  readonly msa?: TemplateMsaWeights;
}

/**
 * What the pair features are built from, all of it one value per residue.
 *
 * The 88-channel feature itself is never built: at 1500 residues it would be
 * 792 MiB, and it holds at most five non-zero entries per pair. Everything the
 * kernel needs is here, on the residue axis.
 */
export interface TemplatePairInput {
  /** CB, or CA for glycine, [length, 3]. */
  readonly pseudoBeta: Float32Array;
  readonly pseudoBetaMask: Float32Array;
  /** 1 where N, CA and C are all present. */
  readonly backboneMask: Float32Array;
  /** Template residue per query position, 21 for a gap. */
  readonly aatype: Int32Array;
}

export interface QueryOnlyTemplateInput {
  readonly length: number;
  readonly templateChannels: number;
  readonly pairChannels: number;
  readonly pairMask: Float32Array;
  readonly weights: QueryOnlyTemplateWeights;
  /** Absent for the query-only case, which is a template of nothing. */
  readonly template?: TemplatePairInput;
  /**
   * Storage of the module's own 64-channel pair.
   *
   * Packed it is half the size, which at 597 residues is 44 MiB off the run's
   * peak, and it is the same storage the trunk keeps its pair in.
   */
  readonly templateStorage?: ActivationStorage;
  /**
   * Add the update straight into a pair the model is already holding.
   *
   * Without this the module writes it to a tensor of its own, which at 597
   * residues is 182 MiB that nothing else ever reads.
   */
  readonly residual?: {
    readonly pair: GpuTensor;
    readonly storage: ActivationStorage;
  };
  /**
   * Run inside the model's own execution.
   *
   * Without this the module allocates for itself and reads the update home,
   * which is what the tests want and what a pair-sized readback costs.
   */
  readonly execution?: WebGpuExecution;
}

export interface QueryOnlyTemplateResult {
  readonly pairUpdate: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
  /** Present, in place of pairUpdate, when the model supplied its execution. */
  readonly pairUpdateTensor?: GpuTensor;
}

/**
 * The template pair features and embedding2d, in one pass.
 *
 * AlphaFold builds an 88-channel feature per pair and projects it to 64. Both
 * halves are done here at once, because the feature is far too big to hold —
 * 792 MiB at 1500 residues — and far too sparse to be worth holding. Of its 88
 * channels, 39 are a one-hot distance bin, 22 a one-hot residue along j, 22 a
 * one-hot residue along i, three are the unit vector that model_1_ptm switches
 * off, and the remaining two are masks. At most five entries are non-zero, so
 * the projection is five rows of the weight matrix added to the bias rather
 * than an 88-long dot product.
 *
 * With no template every mask is zero and every channel with it, which leaves
 * the bias alone: the same answer the query-only path has always produced.
 */
function createTemplateEmbedShader(storage: ActivationStorage = "f32"): string {
  const write = storage === "f16"
    ? `output[(base + channel) >> 1u] = pack2x16float(vec2<f32>(first, second));`
    : `output[base + channel] = first;
      output[base + channel + 1u] = second;`;
  return `
struct Parameters {
  length: u32, channels: u32, pair_offset: u32, pairs: u32,
};
const GRID_WIDTH: u32 = 32768u;
const RESIDUE_STRIDE: u32 = 8u;
const DGRAM_BINS: u32 = 39u;
const DGRAM_MINIMUM: f32 = 3.25;
const DGRAM_STEP: f32 = 1.25;
const CHANNEL_PSEUDO_BETA_MASK: u32 = 39u;
const CHANNEL_AATYPE_J: u32 = 40u;
const CHANNEL_AATYPE_I: u32 = 62u;
const CHANNEL_FRAME_MASK: u32 = 87u;

@group(0) @binding(0) var<storage, read> residues: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<${storageArray(storage)}>;

fn lower_edge(bin: u32) -> f32 {
  let edge = DGRAM_MINIMUM + DGRAM_STEP * f32(bin);
  return edge * edge;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let local = id.x + id.y * GRID_WIDTH * 64u;
  if (local >= p.pairs) { return; }
  let pair = p.pair_offset + local;
  let i = pair / p.length;
  let j = pair % p.length;
  let a = i * RESIDUE_STRIDE;
  let b = j * RESIDUE_STRIDE;

  let mask_2d = residues[a + 3u] * residues[b + 3u];
  let base = local * p.channels;
  if (mask_2d == 0.0) {
    for (var channel = 0u; channel < p.channels; channel += 2u) {
      let first = bias[channel];
      let second = bias[channel + 1u];
      ${write}
    }
    return;
  }

  // AlphaFold's distogram is a half-open bucket, and its bins are the squares
  // of 39 evenly spaced distances from 3.25 to 50.75 angstroms. A distance
  // below the first edge falls in no bin at all, and so does one landing
  // exactly on an edge, which both comparisons have to keep.
  var difference = 0.0;
  for (var axis = 0u; axis < 3u; axis += 1u) {
    let delta = residues[a + axis] - residues[b + axis];
    difference += delta * delta;
  }
  var bin = -1;
  for (var candidate = 0u; candidate < DGRAM_BINS; candidate += 1u) {
    if (difference > lower_edge(candidate)) { bin = i32(candidate); }
  }
  if (bin >= 0 && bin + 1 < i32(DGRAM_BINS) && !(difference < lower_edge(u32(bin) + 1u))) {
    bin = -1;
  }

  let aatype_i = u32(residues[a + 5u]);
  let aatype_j = u32(residues[b + 5u]);
  let frame_2d = residues[a + 4u] * residues[b + 4u];
  let row_mask = CHANNEL_PSEUDO_BETA_MASK * p.channels;
  let row_j = (CHANNEL_AATYPE_J + aatype_j) * p.channels;
  let row_i = (CHANNEL_AATYPE_I + aatype_i) * p.channels;
  let row_frame = CHANNEL_FRAME_MASK * p.channels;
  let row_bin = u32(max(bin, 0)) * p.channels;
  // Two channels at a time, because a packed pair holds two in one word.
  for (var channel = 0u; channel < p.channels; channel += 2u) {
    var first = bias[channel] + weights[row_mask + channel]
      + weights[row_j + channel] + weights[row_i + channel];
    var second = bias[channel + 1u] + weights[row_mask + channel + 1u]
      + weights[row_j + channel + 1u] + weights[row_i + channel + 1u];
    if (bin >= 0) {
      first += weights[row_bin + channel];
      second += weights[row_bin + channel + 1u];
    }
    if (frame_2d != 0.0) {
      first += weights[row_frame + channel];
      second += weights[row_frame + channel + 1u];
    }
    ${write}
  }
}`;
}

/** The residue axis the embedding kernel reads, packed one row per residue. */
const RESIDUE_STRIDE = 8;

export function packTemplateResidues(length: number, template?: TemplatePairInput): Float32Array {
  const packed = new Float32Array(length * RESIDUE_STRIDE);
  if (template === undefined) return packed;
  for (let residue = 0; residue < length; residue += 1) {
    const row = residue * RESIDUE_STRIDE;
    packed[row] = template.pseudoBeta[residue * 3]!;
    packed[row + 1] = template.pseudoBeta[residue * 3 + 1]!;
    packed[row + 2] = template.pseudoBeta[residue * 3 + 2]!;
    packed[row + 3] = template.pseudoBetaMask[residue]!;
    packed[row + 4] = template.backboneMask[residue]!;
    packed[row + 5] = template.aatype[residue]!;
  }
  return packed;
}


/**
 * The output norm, the pointwise attention and the residual, in one pass.
 *
 * Written separately these are three tensors the size of the pair: a
 * normalized copy of the template pair, the attention's value projection, and
 * the 128-channel update itself, which at 597 residues is 91, 91 and 182 MiB.
 * None of them is read by anything else, so none of them needs to exist. One
 * invocation takes one pair row, normalises it, applies the attention and adds
 * the result into the pair the model is already holding.
 *
 * The attention's two matrices are composed on the host first. With a single
 * template the softmax is 1, so the value and output projections are just
 * multiplied through, and 64 by 64 by 128 is half a million products once per
 * prediction against 8192 per pair every time.
 */
export function createTemplateOutputShader(
  storage: ActivationStorage, residual = false, sourceStorage: ActivationStorage = "f32",
): string {
  const source = (index: string): string => storedElement(sourceStorage, "source", index);
  const word = "(base + channel) >> 1u";
  const store = storage === "f16"
    ? residual
      ? `let existing = unpack2x16float(destination[${word}]);
    destination[${word}] = pack2x16float(vec2<f32>(existing.x + first, existing.y + second));`
      : `destination[${word}] = pack2x16float(vec2<f32>(first, second));`
    : residual
      ? `destination[base + channel] = destination[base + channel] + first;
    destination[base + channel + 1u] = destination[base + channel + 1u] + second;`
      : `destination[base + channel] = first;
    destination[base + channel + 1u] = second;`;
  return `
struct Parameters { pairs: u32, template_channels: u32, pair_channels: u32, epsilon: f32 };
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<${storageArray(sourceStorage)}>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> norm_weights: array<f32>;
@group(0) @binding(4) var<uniform> p: Parameters;
@group(0) @binding(5) var<storage, read_write> destination: array<${storageArray(storage)}>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x + id.y * GRID_WIDTH * 64u;
  if (row >= p.pairs) { return; }
  let source_base = row * p.template_channels;

  var mean = 0.0;
  for (var c = 0u; c < p.template_channels; c += 1u) { mean += ${source("source_base + c")}; }
  mean /= f32(p.template_channels);
  var variance = 0.0;
  for (var c = 0u; c < p.template_channels; c += 1u) {
    let difference = ${source("source_base + c")} - mean;
    variance += difference * difference;
  }
  let inverse_deviation = inverseSqrt(variance / f32(p.template_channels) + p.epsilon);

  // Two output channels at a time: a packed pair holds two values in one word,
  // and either way it halves the work of normalising the row again.
  let base = row * p.pair_channels;
  for (var channel = 0u; channel < p.pair_channels; channel += 2u) {
    var first = bias[channel];
    var second = bias[channel + 1u];
    for (var c = 0u; c < p.template_channels; c += 1u) {
      let normalized = (${source("source_base + c")} - mean) * inverse_deviation * norm_weights[c]
        + norm_weights[p.template_channels + c];
      first += normalized * weight[c * p.pair_channels + channel];
      second += normalized * weight[c * p.pair_channels + channel + 1u];
    }
    ${store}
  }
}`;
}

/**
 * The pointwise attention's value and output projections, multiplied through.
 *
 * value_w is [template channels, heads, head channels] and output_w is
 * [heads, head channels, pair channels]; with one template the attention
 * between them is the identity, so their product is a single [template
 * channels, pair channels] matrix.
 */
export function composePointwiseWeights(
  valueWeight: Float32Array, outputWeight: Float32Array,
  templateChannels: number, pairChannels: number,
): Float32Array {
  const projected = valueWeight.length / templateChannels;
  if (outputWeight.length !== projected * pairChannels) {
    throw new RangeError("template attention value and output projections disagree on shape");
  }
  const composed = new Float32Array(templateChannels * pairChannels);
  for (let input = 0; input < templateChannels; input += 1) {
    for (let hidden = 0; hidden < projected; hidden += 1) {
      const factor = valueWeight[input * projected + hidden]!;
      if (factor === 0) continue;
      for (let output = 0; output < pairChannels; output += 1) {
        composed[input * pairChannels + output] = composed[input * pairChannels + output]!
          + factor * outputWeight[hidden * pairChannels + output]!;
      }
    }
  }
  return composed;
}

export class QueryOnlyTemplateGpu {
  readonly device: GPUDevice;
  constructor(device: GPUDevice) { this.device = device; }

  /**
   * Computes the template pair update for one length and mask.
   *
   * A prediction does not call this: with no template search the update is a
   * constant, which queryOnlyTemplateConstant probes once and the model
   * folds into the pair projection's bias. The module runs for the
   * differential tests, and for the padded masks the constant does not cover.
   */
  async run(
    input: QueryOnlyTemplateInput,
  ): Promise<QueryOnlyTemplateResult> {
    const execution = input.execution ?? new WebGpuExecution(this.device);
    const entryCheckpoint = execution.checkpoint();
    let retainOutput = false;
    try {
      const pairs = input.length * input.length;
      const templateStorage = input.templateStorage ?? "f32";
      const pair = execution.allocate(
        "template.pair", storageWords(pairs * input.templateChannels, templateStorage),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const pairMask = execution.upload("template.pair-mask", input.pairMask);
      const bias = execution.upload("template.embedding-bias", input.weights.embeddingBias);
      const embeddingWeight = execution.upload("template.embedding-weight", input.weights.embeddingWeight);
      const residues = execution.upload(
        "template.residues", packTemplateResidues(input.length, input.template),
      );
      const embed = await execution.pipelines.get(`template:embed:${templateStorage}`,
        createTemplateEmbedShader(templateStorage));
      // One command buffer and one validation scope for the whole module: every
      // submission awaited from the page is a round trip to the GPU process.
      const encoder = this.device.createCommandEncoder({ label: "template" });
      this.device.pushErrorScope("validation");
      // The pair outgrows what one binding may cover well before the lengths
      // this has to reach, so the embedding walks it a window of rows at a time
      // and each window is told which pair it starts at.
      let grid = execution.linearGrid(pairs);
      for (const window of rowWindows(pairs, execution.bindingLimitBytes,
        [storageWords(input.templateChannels, templateStorage) * 4])) {
        const outputWindow = execution.view(pair,
          storageWords(window.offset * input.templateChannels, templateStorage),
          storageWords(window.count * input.templateChannels, templateStorage));
        const embedParams = execution.upload(`template.embed-parameters-${window.offset}`, new Uint32Array([
          input.length, input.templateChannels, window.offset, window.count,
        ]), GPUBufferUsage.UNIFORM);
        grid = execution.linearGrid(window.count);
        execution.dispatch(encoder, embed, [residues, embeddingWeight, bias, embedParams, outputWindow],
          grid[0], grid[1], 1, `template.embed-${window.offset}`);
      }
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();

      for (let block = 0; block < input.weights.blockWeights.length; block += 1) {
        await encodeTemplatePairBlock(execution, encoder, {
          sequences: 1,
          length: input.length,
          cM: input.templateChannels,
          cZ: input.templateChannels,
          cOuter: 0,
          triangleHidden: input.weights.blockWeights[block]!.triangleMultiplicationOutgoing.linearAPBias.length,
          // The module's pair is the one these blocks operate on, so its
          // storage is theirs, and the projection they keep whole follows it.
          pairStorage: templateStorage, triangleWholeStorage: templateStorage,
        }, input.weights.blockWeights[block]!, pair, pairMask);
        execution.releaseSince(persistentCheckpoint);
      }

      const normWeights = new Float32Array(input.templateChannels * 2);
      normWeights.set(input.weights.outputNormScale);
      normWeights.set(input.weights.outputNormOffset, input.templateChannels);
      const normWeightBuffer = execution.upload("template.output-norm-weights", normWeights);
      const composed = execution.upload("template.pointwise-weight", composePointwiseWeights(
        input.weights.valueWeight, input.weights.outputWeight, input.templateChannels, input.pairChannels,
      ));
      const outputBias = execution.upload("template.output-bias", input.weights.outputBias);
      const storage = input.residual?.storage ?? "f32";
      const output = input.residual?.pair ?? execution.allocate(
        "template.output", pairs * input.pairChannels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const outputPipeline = await execution.pipelines.get(
        `template:output:${storage}:${input.residual === undefined ? "write" : "add"}:${templateStorage}`,
        createTemplateOutputShader(storage, input.residual !== undefined, templateStorage),
      );
      // The update is written a window of rows at a time, like the embedding,
      // and reads its own window of the template pair alongside.
      for (const window of rowWindows(pairs, execution.bindingLimitBytes,
        [storageWords(input.templateChannels, templateStorage) * 4,
          storageWords(input.pairChannels, storage) * 4])) {
        const sourceWindow = execution.view(pair,
          storageWords(window.offset * input.templateChannels, templateStorage),
          storageWords(window.count * input.templateChannels, templateStorage));
        const destinationWindow = execution.view(output,
          storageWords(window.offset * input.pairChannels, storage),
          storageWords(window.count * input.pairChannels, storage));
        // Three counts and an epsilon, so the buffer is built by hand.
        const parameterBytes = new ArrayBuffer(16);
        const parameterView = new DataView(parameterBytes);
        parameterView.setUint32(0, window.count, true);
        parameterView.setUint32(4, input.templateChannels, true);
        parameterView.setUint32(8, input.pairChannels, true);
        parameterView.setFloat32(12, 1e-5, true);
        const outputParams = execution.upload(`template.output-parameters-${window.offset}`,
          new Uint32Array(parameterBytes), GPUBufferUsage.UNIFORM);
        grid = execution.linearGrid(window.count);
        execution.dispatch(encoder, outputPipeline,
          [sourceWindow, composed, outputBias, normWeightBuffer, outputParams, destinationWindow],
          grid[0], grid[1], 1, `template.output-${window.offset}`);
      }
      const readback = input.execution === undefined
        ? execution.createReadback("template.readback", output, encoder) : undefined;
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU template module failed: ${error.message}`);
      if (readback === undefined) {
        retainOutput = true;
        return {
          pairUpdate: new Float32Array(0), elapsedMilliseconds: performance.now() - start,
          memory: execution.snapshot(), pairUpdateTensor: output,
        };
      }
      return {
        pairUpdate: await execution.mapFloat32(readback),
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      if (input.execution === undefined) execution.release();
      else if (!retainOutput) execution.releaseSince(entryCheckpoint);
    }
  }
}

/** Residues used to probe the update; large enough to exercise every kernel, small enough to be free. */
const TEMPLATE_PROBE_LENGTH = 8;

/**
 * The query-only template update as the single pair vector it turns out to be.
 *
 * With template search off the module has no sequence input at all: it sees a
 * length, the pair mask and the weights. Every pair of an unpadded chain
 * therefore receives the same vector, which measurement confirms is identical
 * to the last bit within a run and stable across lengths to about 2e-7 out of
 * values around 0.28. So the whole module can run once on a handful of
 * residues instead of once per recycle on the real chain, and its result can
 * be carried as 128 numbers rather than an L by L by 128 tensor.
 *
 * Returns undefined when the probe's own rows disagree, which would mean the
 * assumption does not hold for these weights and the caller should run the
 * module for real.
 */
export async function queryOnlyTemplateConstant(
  device: GPUDevice, weights: QueryOnlyTemplateWeights,
  templateChannels = 64, pairChannels = 128,
): Promise<Float32Array | undefined> {
  const length = TEMPLATE_PROBE_LENGTH;
  const { pairUpdate } = await new QueryOnlyTemplateGpu(device).run({
    length, templateChannels, pairChannels,
    pairMask: new Float32Array(length * length).fill(1), weights,
  });
  const constant = pairUpdate.slice(0, pairChannels);
  for (let pair = 1; pair < length * length; pair += 1) {
    for (let channel = 0; channel < pairChannels; channel += 1) {
      if (pairUpdate[pair * pairChannels + channel] !== constant[channel]) return undefined;
    }
  }
  return constant;
}
