/**
 * The Multimer template module under a binding limit below its tensors.
 *
 * Its query pair, template pair and pair update are all length-squared
 * tensors, so a complex of a few hundred residues already passes what a
 * device binds by default and the module walks them in windows of rows.
 * Nothing it computes depends on where those windows fall, so the same input
 * must give the same output whatever the limit. There is no released-weight
 * fixture for this module in the repository, so the weights are random: the
 * comparison is against the module itself, not against AlphaFold.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  MultimerMockTemplateGpu, type MultimerMockTemplateWeights,
} from "../src/evoformer/multimer-template.js";
import type { TemplatePairBlockWeights } from "../src/evoformer/block.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const LENGTH = 24;
const PAIR_CHANNELS = 128;
const TEMPLATE_CHANNELS = 64;
const HEADS = 4;
const HIDDEN = 16;

let state = 0x2545f491;
function random(): number {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  return state / 0x100000000 - 0.5;
}
const values = (count: number): Float32Array => Float32Array.from({ length: count }, () => random() * 0.2);

function attention(channels: number): TemplatePairBlockWeights["triangleAttentionStarting"] {
  const headDim = channels / HEADS;
  return {
    heads: HEADS,
    pairProjectionWeight: values(channels * HEADS),
    attention: {
      queryNormScale: values(channels), queryNormOffset: values(channels),
      queryWeight: values(channels * HEADS * headDim), keyWeight: values(channels * HEADS * headDim),
      valueWeight: values(channels * HEADS * headDim), gatingWeight: values(channels * HEADS * headDim),
      gatingBias: values(HEADS * headDim), outputWeight: values(HEADS * headDim * channels),
      outputBias: values(channels),
    },
  };
}

function multiplication(): TemplatePairBlockWeights["triangleMultiplicationOutgoing"] {
  const c = TEMPLATE_CHANNELS;
  return {
    layerNormInWeight: values(c), layerNormInBias: values(c),
    linearAPWeight: values(c * HIDDEN), linearAPBias: values(HIDDEN),
    linearAGWeight: values(c * HIDDEN), linearAGBias: values(HIDDEN),
    linearBPWeight: values(c * HIDDEN), linearBPBias: values(HIDDEN),
    linearBGWeight: values(c * HIDDEN), linearBGBias: values(HIDDEN),
    layerNormOutWeight: values(HIDDEN), layerNormOutBias: values(HIDDEN),
    linearZWeight: values(HIDDEN * c), linearZBias: values(c),
    linearGWeight: values(c * c), linearGBias: values(c),
  };
}

describe.skipIf(!enabled)("Multimer template windows", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("gives the same update whether its tensors are one binding or several", async () => {
    const weights: MultimerMockTemplateWeights = {
      queryNormScale: values(PAIR_CHANNELS), queryNormOffset: values(PAIR_CHANNELS),
      pairInputWeight: values(PAIR_CHANNELS * TEMPLATE_CHANNELS), pairInputBias: values(TEMPLATE_CHANNELS),
      blockWeights: [{
        triangleMultiplicationOutgoing: multiplication(),
        triangleMultiplicationIncoming: multiplication(),
        triangleAttentionStarting: attention(TEMPLATE_CHANNELS),
        triangleAttentionEnding: attention(TEMPLATE_CHANNELS),
        pairTransition: {
          layerNormScale: values(TEMPLATE_CHANNELS), layerNormOffset: values(TEMPLATE_CHANNELS),
          firstWeight: values(TEMPLATE_CHANNELS * 128), firstBias: values(128),
          secondWeight: values(128 * TEMPLATE_CHANNELS), secondBias: values(TEMPLATE_CHANNELS),
        },
      }],
      outputNormScale: values(TEMPLATE_CHANNELS), outputNormOffset: values(TEMPLATE_CHANNELS),
      outputWeight: values(TEMPLATE_CHANNELS * PAIR_CHANNELS), outputBias: values(PAIR_CHANNELS),
      msaInputWeight: values(34 * 256), msaInputBias: values(256),
      msaOutputWeight: values(256 * 256), msaOutputBias: values(256),
      templateRows: 4,
    };
    const pair = values(LENGTH * LENGTH * PAIR_CHANNELS);
    const mask = Float32Array.from({ length: LENGTH * LENGTH }, () => 1);
    const whole = await new MultimerMockTemplateGpu(device).run(pair, mask, LENGTH, weights);
    // 576 pair rows of 512 bytes: this limit takes them 128 at a time.
    const windowed = await new MultimerMockTemplateGpu(device, 64 * 1024).run(pair, mask, LENGTH, weights);
    expect(errorMetrics(windowed.pairUpdate, whole.pairUpdate).maxAbsoluteError).toBe(0);
    expect(errorMetrics(windowed.msaRows, whole.msaRows).maxAbsoluteError).toBe(0);
    expect(whole.pairUpdate.some((value) => value !== 0)).toBe(true);
  }, 300_000);
});
