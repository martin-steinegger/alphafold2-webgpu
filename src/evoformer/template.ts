import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "./attention.js";
import { encodeTemplatePairBlock, type TemplatePairBlockWeights } from "./block.js";
import { rowWindows } from "../runtime/sharded.js";
import { type GpuTensor, WebGpuExecution } from "../runtime/execution.js";
import type { AllocationSnapshot } from "../runtime/allocator.js";
import type { TemplateMsaWeights } from "../input/template-msa-row.js";

export interface QueryOnlyTemplateWeights {
  /** `embedding2d`, the [88, 64] projection of the template pair features. */
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
   * `template_single_embedding` and `template_projection`, which turn a
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
  /** CB, or CA for glycine, `[length, 3]`. */
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
   * Run inside the model's own execution and hand the update back as a tensor.
   *
   * Without this the module allocates for itself and reads the update home,
   * which is what the tests want and what a pair-sized readback costs. The
   * model instead adds the tensor straight into its resident pair.
   */
  readonly execution?: WebGpuExecution;
}

export interface QueryOnlyTemplateResult {
  readonly pairUpdate: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
  /** Present, in place of `pairUpdate`, when the model supplied its execution. */
  readonly pairUpdateTensor?: GpuTensor;
}

/**
 * The template pair features and `embedding2d`, in one pass.
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
const TEMPLATE_EMBED_SHADER = `
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
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

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
    for (var channel = 0u; channel < p.channels; channel += 1u) {
      output[base + channel] = bias[channel];
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
  for (var channel = 0u; channel < p.channels; channel += 1u) {
    var value = bias[channel] + weights[row_mask + channel]
      + weights[row_j + channel] + weights[row_i + channel];
    if (bin >= 0) { value += weights[row_bin + channel]; }
    if (frame_2d != 0.0) { value += weights[row_frame + channel]; }
    output[base + channel] = value;
  }
}`;

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


const VALUE_SHADER = `
struct Parameters { pairs: u32, template_channels: u32, projected: u32, pair_channels: u32 };
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> p: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.pairs * p.projected) { return; }
  let row = index / p.projected;
  let channel = index % p.projected;
  var result = 0.0;
  for (var c = 0u; c < p.template_channels; c += 1u) {
    result += source[row * p.template_channels + c] * weights[c * p.projected + channel];
  }
  output[index] = result;
}`;

const OUTPUT_SHADER = `
struct Parameters { pairs: u32, template_channels: u32, projected: u32, pair_channels: u32 };
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> output_weight: array<f32>;
@group(0) @binding(2) var<storage, read> output_bias: array<f32>;
@group(0) @binding(3) var<uniform> p: Parameters;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.pairs * p.pair_channels) { return; }
  let row = index / p.pair_channels;
  let channel = index % p.pair_channels;
  var result = output_bias[channel];
  for (var c = 0u; c < p.projected; c += 1u) {
    result += source[row * p.projected + c] * output_weight[c * p.pair_channels + channel];
  }
  output[index] = result;
}`;

export class QueryOnlyTemplateGpu {
  readonly device: GPUDevice;
  constructor(device: GPUDevice) { this.device = device; }

  /**
   * Computes the template pair update for one length and mask.
   *
   * A prediction does not call this: with no template search the update is a
   * constant, which `queryOnlyTemplateConstant` probes once and the model
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
      const pair = execution.allocate(
        "template.pair", pairs * input.templateChannels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const pairMask = execution.upload("template.pair-mask", input.pairMask);
      const bias = execution.upload("template.embedding-bias", input.weights.embeddingBias);
      const embeddingWeight = execution.upload("template.embedding-weight", input.weights.embeddingWeight);
      const residues = execution.upload(
        "template.residues", packTemplateResidues(input.length, input.template),
      );
      const embed = await execution.pipelines.get("template:embed", TEMPLATE_EMBED_SHADER);
      // One command buffer and one validation scope for the whole module: every
      // submission awaited from the page is a round trip to the GPU process.
      const encoder = this.device.createCommandEncoder({ label: "template" });
      this.device.pushErrorScope("validation");
      // The pair outgrows what one binding may cover well before the lengths
      // this has to reach, so the embedding walks it a window of rows at a time
      // and each window is told which pair it starts at.
      let grid = execution.linearGrid(pairs);
      for (const window of rowWindows(pairs, execution.bindingLimitBytes, [input.templateChannels * 4])) {
        const outputWindow = execution.view(pair,
          window.offset * input.templateChannels, window.count * input.templateChannels);
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
        }, input.weights.blockWeights[block]!, pair, pairMask);
        execution.releaseSince(persistentCheckpoint);
      }

      const normWeights = new Float32Array(input.templateChannels * 2);
      normWeights.set(input.weights.outputNormScale);
      normWeights.set(input.weights.outputNormOffset, input.templateChannels);
      const normWeightBuffer = execution.upload("template.output-norm-weights", normWeights);
      const normParams = execution.upload("template.output-norm-parameters", createAttentionNormParameters(
        pairs, input.templateChannels, 0, input.templateChannels, false, 1, pairs, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const normalized = execution.allocate("template.normalized", pair.elements);
      const projected = input.weights.valueWeight.length / input.templateChannels;
      const value = execution.allocate("template.value", pairs * projected);
      const output = execution.allocate(
        "template.output", pairs * input.pairChannels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const valueWeight = execution.upload("template.value-weight", input.weights.valueWeight);
      const outputWeight = execution.upload("template.output-weight", input.weights.outputWeight);
      const outputBias = execution.upload("template.output-bias", input.weights.outputBias);
      const params = execution.upload("template.pointwise-parameters", new Uint32Array([
        pairs, input.templateChannels, projected, input.pairChannels,
      ]), GPUBufferUsage.UNIFORM);
      const [normalize, valuePipeline, outputPipeline] = await Promise.all([
        execution.pipelines.get("template:normalize", ATTENTION_NORMALIZE_SHADER),
        execution.pipelines.get("template:value", VALUE_SHADER),
        execution.pipelines.get("template:output", OUTPUT_SHADER),
      ]);
      grid = execution.linearGrid(pairs, 1);
      execution.dispatch(encoder, normalize, [pair, normWeightBuffer, normParams, normalized],
        grid[0], grid[1], 1,
        "template.output-normalize");
      grid = execution.linearGrid(value.elements);
      execution.dispatch(encoder, valuePipeline, [normalized, valueWeight, params, value],
        grid[0], grid[1], 1, "template.value");
      grid = execution.linearGrid(output.elements);
      execution.dispatch(encoder, outputPipeline, [value, outputWeight, outputBias, params, output],
        grid[0], grid[1], 1, "template.output");
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
