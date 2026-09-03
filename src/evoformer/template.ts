import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "./attention.js";
import { encodeTemplatePairBlock, type TemplatePairBlockWeights } from "./block.js";
import { type GpuTensor, WebGpuExecution } from "../runtime/execution.js";
import type { AllocationSnapshot } from "../runtime/allocator.js";

export interface QueryOnlyTemplateWeights {
  readonly embeddingBias: Float32Array;
  readonly blockWeights: readonly TemplatePairBlockWeights[];
  readonly outputNormScale: Float32Array;
  readonly outputNormOffset: Float32Array;
  readonly valueWeight: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
  readonly heads: number;
}

export interface QueryOnlyTemplateInput {
  readonly length: number;
  readonly templateChannels: number;
  readonly pairChannels: number;
  readonly pairMask: Float32Array;
  readonly weights: QueryOnlyTemplateWeights;
}

export interface QueryOnlyTemplateResult {
  readonly pairUpdate: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

const INIT_SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> bias: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= arrayLength(&output)) { return; }
  output[index] = bias[index % arrayLength(&bias)];
}`;

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
    const execution = new WebGpuExecution(this.device);
    try {
      const pairs = input.length * input.length;
      const pair = execution.allocate(
        "template.pair", pairs * input.templateChannels, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const pairMask = execution.upload("template.pair-mask", input.pairMask);
      const bias = execution.upload("template.embedding-bias", input.weights.embeddingBias);
      const init = await execution.pipelines.get("template:init", INIT_SHADER);
      // One command buffer and one validation scope for the whole module: every
      // submission awaited from the page is a round trip to the GPU process.
      const encoder = this.device.createCommandEncoder({ label: "template" });
      this.device.pushErrorScope("validation");
      let grid = execution.linearGrid(pair.elements);
      execution.dispatch(encoder, init, [bias, pair], grid[0], grid[1], 1, "template.initialize");
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
      const readback = execution.createReadback("template.readback", output, encoder);
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU template module failed: ${error.message}`);
      return {
        pairUpdate: await execution.mapFloat32(readback),
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      execution.release();
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
