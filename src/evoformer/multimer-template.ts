import { ATTENTION_NORMALIZE_SHADER, createAttentionNormParameters } from "./attention.js";
import { encodeMultimerTemplatePairBlock, type TemplatePairBlockWeights } from "./block.js";
import { WebGpuExecution, type GpuTensor } from "../runtime/execution.js";

export interface MultimerMockTemplateWeights {
  readonly queryNormScale: Float32Array;
  readonly queryNormOffset: Float32Array;
  readonly pairInputWeight: Float32Array;
  readonly pairInputBias: Float32Array;
  readonly blockWeights: readonly TemplatePairBlockWeights[];
  readonly outputNormScale: Float32Array;
  readonly outputNormOffset: Float32Array;
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
  readonly msaInputWeight: Float32Array;
  readonly msaInputBias: Float32Array;
  readonly msaOutputWeight: Float32Array;
  readonly msaOutputBias: Float32Array;
  readonly templateRows: number;
}

export interface MultimerMockTemplateResult {
  readonly pairUpdate: Float32Array;
  readonly msaRows: Float32Array;
  readonly msaMask: Float32Array;
  readonly submissions: number;
  /** Present only for the internal device-resident trunk handoff. */
  readonly pairUpdateTensor?: GpuTensor;
  /** Present only for the internal device-resident trunk handoff. */
  readonly msaRowsTensor?: GpuTensor;
}

export interface MultimerMockTemplateResidentInput {
  readonly pair: GpuTensor;
  readonly pairMask: GpuTensor;
}

const PAIR_INIT_SHADER = `
struct P { pairs: u32, input_channels: u32, output_channels: u32 };
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.pairs * p.output_channels) { return; }
  let row = index / p.output_channels;
  let out = index % p.output_channels;
  var value = bias[out];
  for (var c = 0u; c < p.input_channels; c += 1u) {
    value += source[row * p.input_channels + c] * weight[c * p.output_channels + out];
  }
  output[index] = value;
}`;

const OUTPUT_SHADER = `
struct P { rows: u32, input_channels: u32, output_channels: u32 };
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.rows * p.output_channels) { return; }
  let row = index / p.output_channels;
  let out = index % p.output_channels;
  var value = bias[out];
  for (var c = 0u; c < p.input_channels; c += 1u) {
    value += max(source[row * p.input_channels + c], 0.0) * weight[c * p.output_channels + out];
  }
  output[index] = value;
}`;

const MSA_SHADER = `
struct P { length: u32, rows: u32, channels: u32 };
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> input_weight: array<f32>;
@group(0) @binding(1) var<storage, read> input_bias: array<f32>;
@group(0) @binding(2) var<storage, read> output_weight: array<f32>;
@group(0) @binding(3) var<storage, read> output_bias: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  if (index >= p.rows * p.length * p.channels) { return; }
  let out = index % p.channels;
  var value = output_bias[out];
  for (var c = 0u; c < p.channels; c += 1u) {
    // Mock templates are alanine with zero coordinates and chi masks.
    let hidden = max(input_bias[c] + input_weight[c], 0.0);
    value += hidden * output_weight[c * p.channels + out];
  }
  output[index] = value;
}`;

function validateFloat32(name: string, value: Float32Array, elements: number): void {
  if (!(value instanceof Float32Array)) throw new TypeError(`${name} must be a Float32Array`);
  if (value.length !== elements || value.byteLength !== elements * Float32Array.BYTES_PER_ELEMENT) {
    throw new RangeError(`${name} must contain ${elements} float32 values`);
  }
}

export class MultimerMockTemplateGpu {
  readonly device: GPUDevice;
  constructor(device: GPUDevice) { this.device = device; }

  async run(pairValue: Float32Array, pairMaskValue: Float32Array, length: number,
    weightsValue: MultimerMockTemplateWeights,
    sharedExecution?: WebGpuExecution,
    residentInput?: MultimerMockTemplateResidentInput): Promise<MultimerMockTemplateResult> {
    if (!Number.isSafeInteger(length) || length <= 0) throw new RangeError("template length must be positive");
    const pairs = length * length;
    if (!Number.isSafeInteger(pairs * 128)) throw new RangeError("template tensor size exceeds JavaScript precision");
    if (residentInput === undefined) {
      if (pairValue.length !== pairs * 128) throw new RangeError("template pair tensor must have shape [L, L, 128]");
      if (pairMaskValue.length !== pairs) throw new RangeError("template pair mask must have shape [L, L]");
      validateFloat32("template pair tensor", pairValue, pairs * 128);
      validateFloat32("template pair mask", pairMaskValue, pairs);
    } else if (sharedExecution === undefined || residentInput.pair.elements !== pairs * 128
      || residentInput.pairMask.elements !== pairs) {
      throw new RangeError("resident template pair tensors have the wrong shape or execution");
    }
    if (!Number.isSafeInteger(weightsValue.templateRows) || weightsValue.templateRows <= 0) {
      throw new RangeError("template row count must be positive");
    }
    validateFloat32("template query norm scale", weightsValue.queryNormScale, 128);
    validateFloat32("template query norm offset", weightsValue.queryNormOffset, 128);
    validateFloat32("template pair input weight", weightsValue.pairInputWeight, 128 * 64);
    validateFloat32("template pair input bias", weightsValue.pairInputBias, 64);
    validateFloat32("template output norm scale", weightsValue.outputNormScale, 64);
    validateFloat32("template output norm offset", weightsValue.outputNormOffset, 64);
    validateFloat32("template output weight", weightsValue.outputWeight, 64 * 128);
    validateFloat32("template output bias", weightsValue.outputBias, 128);
    validateFloat32("template MSA input weight", weightsValue.msaInputWeight, 34 * 256);
    validateFloat32("template MSA input bias", weightsValue.msaInputBias, 256);
    validateFloat32("template MSA output weight", weightsValue.msaOutputWeight, 256 * 256);
    validateFloat32("template MSA output bias", weightsValue.msaOutputBias, 256);
    if (weightsValue.blockWeights.length === 0) throw new RangeError("template pair stack must contain blocks");
    const execution = sharedExecution ?? new WebGpuExecution(this.device);
    const entryCheckpoint = execution.checkpoint();
    let retainResidentOutput = false;
    try {
      const pairChannels = 128;
      const templateChannels = 64;
      const pair = residentInput?.pair ?? execution.upload("multimer-template.query", pairValue);
      const pairMask = residentInput?.pairMask ?? execution.upload("multimer-template.mask", pairMaskValue);
      const queryNormWeights = new Float32Array(pairChannels * 2);
      queryNormWeights.set(weightsValue.queryNormScale);
      queryNormWeights.set(weightsValue.queryNormOffset, pairChannels);
      const normWeights = execution.upload("multimer-template.query-norm-weights", queryNormWeights);
      const normParams = execution.upload("multimer-template.query-norm-params", createAttentionNormParameters(
        pairs, pairChannels, 0, pairChannels, false, 1, pairs, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const normalized = execution.allocate("multimer-template.query-normalized", pairs * pairChannels);
      const templatePair = execution.allocate("multimer-template.pair", pairs * templateChannels,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const inputWeight = execution.upload("multimer-template.input-weight", weightsValue.pairInputWeight);
      const inputBias = execution.upload("multimer-template.input-bias", weightsValue.pairInputBias);
      const inputParams = execution.upload("multimer-template.input-params",
        new Uint32Array([pairs, pairChannels, templateChannels]), GPUBufferUsage.UNIFORM);
      const [normalize, pairInit] = await Promise.all([
        execution.pipelines.get("multimer-template:normalize", ATTENTION_NORMALIZE_SHADER),
        execution.pipelines.get("multimer-template:pair-init", PAIR_INIT_SHADER),
      ]);
      let encoder = this.device.createCommandEncoder({ label: "multimer-template.initialize" });
      this.device.pushErrorScope("validation");
      let grid = execution.linearGrid(pairs, 1);
      execution.dispatch(encoder, normalize, [pair, normWeights, normParams, normalized], grid[0], grid[1]);
      grid = execution.linearGrid(templatePair.elements);
      execution.dispatch(encoder, pairInit, [normalized, inputWeight, inputBias, inputParams, templatePair],
        grid[0], grid[1]);
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      let error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU Multimer template initialization failed: ${error.message}`);
      const persistent = execution.checkpoint();
      for (let block = 0; block < weightsValue.blockWeights.length; block += 1) {
        encoder = this.device.createCommandEncoder({ label: `multimer-template.block-${block}` });
        this.device.pushErrorScope("validation");
        await encodeMultimerTemplatePairBlock(execution, encoder, {
          sequences: 1, length, cM: templateChannels, cZ: templateChannels, cOuter: 0,
          triangleHidden: weightsValue.blockWeights[block]!.triangleMultiplicationOutgoing.linearAPBias.length,
        }, weightsValue.blockWeights[block]!, templatePair, pairMask);
        execution.endComputePass(encoder);
        this.device.queue.submit([encoder.finish()]);
        execution.noteSubmitted();
        error = await this.device.popErrorScope();
        if (error !== null) throw new Error(`WebGPU Multimer template block ${block} failed: ${error.message}`);
        execution.releaseSince(persistent);
      }
      const outputNormWeightsValue = new Float32Array(templateChannels * 2);
      outputNormWeightsValue.set(weightsValue.outputNormScale);
      outputNormWeightsValue.set(weightsValue.outputNormOffset, templateChannels);
      const outputNormWeights = execution.upload("multimer-template.output-norm-weights", outputNormWeightsValue);
      const outputNormParams = execution.upload("multimer-template.output-norm-params", createAttentionNormParameters(
        pairs, templateChannels, 0, templateChannels, false, 1, pairs, 1e-5,
      ), GPUBufferUsage.UNIFORM);
      const outputNormalized = execution.allocate("multimer-template.output-normalized", templatePair.elements);
      const outputWeight = execution.upload("multimer-template.output-weight", weightsValue.outputWeight);
      const outputBias = execution.upload("multimer-template.output-bias", weightsValue.outputBias);
      const outputParams = execution.upload("multimer-template.output-params",
        new Uint32Array([pairs, templateChannels, pairChannels]), GPUBufferUsage.UNIFORM);
      const pairUpdate = execution.allocate("multimer-template.pair-update", pairs * pairChannels,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaInputWeight = execution.upload("multimer-template.msa-input-weight", weightsValue.msaInputWeight);
      const msaInputBias = execution.upload("multimer-template.msa-input-bias", weightsValue.msaInputBias);
      const msaOutputWeight = execution.upload("multimer-template.msa-output-weight", weightsValue.msaOutputWeight);
      const msaOutputBias = execution.upload("multimer-template.msa-output-bias", weightsValue.msaOutputBias);
      const msaParams = execution.upload("multimer-template.msa-params",
        new Uint32Array([length, weightsValue.templateRows, 256]), GPUBufferUsage.UNIFORM);
      const msaRows = execution.allocate("multimer-template.msa-rows", weightsValue.templateRows * length * 256,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const [outputPipeline, msaPipeline] = await Promise.all([
        execution.pipelines.get("multimer-template:output", OUTPUT_SHADER),
        execution.pipelines.get("multimer-template:msa", MSA_SHADER),
      ]);
      encoder = this.device.createCommandEncoder({ label: "multimer-template.output" });
      this.device.pushErrorScope("validation");
      grid = execution.linearGrid(pairs, 1);
      execution.dispatch(encoder, normalize, [templatePair, outputNormWeights, outputNormParams, outputNormalized],
        grid[0], grid[1]);
      grid = execution.linearGrid(pairUpdate.elements);
      execution.dispatch(encoder, outputPipeline, [outputNormalized, outputWeight, outputBias, outputParams, pairUpdate],
        grid[0], grid[1]);
      grid = execution.linearGrid(msaRows.elements);
      execution.dispatch(encoder, msaPipeline,
        [msaInputWeight, msaInputBias, msaOutputWeight, msaOutputBias, msaParams, msaRows], grid[0], grid[1]);
      execution.endComputePass(encoder);
      const pairReadback = residentInput === undefined
        ? execution.createReadback("multimer-template.pair-readback", pairUpdate, encoder) : undefined;
      const msaReadback = residentInput === undefined
        ? execution.createReadback("multimer-template.msa-readback", msaRows, encoder) : undefined;
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU Multimer template output failed: ${error.message}`);
      if (residentInput !== undefined) {
        retainResidentOutput = true;
        return {
          pairUpdate: new Float32Array(0), msaRows: new Float32Array(0),
          msaMask: new Float32Array(weightsValue.templateRows * length),
          submissions: weightsValue.blockWeights.length + 2,
          pairUpdateTensor: pairUpdate, msaRowsTensor: msaRows,
        };
      }
      const [pairResult, msaResult] = await Promise.all([
        execution.mapFloat32(pairReadback!), execution.mapFloat32(msaReadback!),
      ]);
      return { pairUpdate: pairResult, msaRows: msaResult,
        msaMask: new Float32Array(weightsValue.templateRows * length),
        submissions: weightsValue.blockWeights.length + 2 };
    } finally {
      if (sharedExecution === undefined) execution.release();
      else if (!retainResidentOutput) execution.releaseSince(entryCheckpoint);
    }
  }
}
