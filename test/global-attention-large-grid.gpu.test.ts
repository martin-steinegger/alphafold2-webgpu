import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  GlobalAttentionGpu, type GlobalAttentionInput, type GlobalAttentionWeights,
} from "../src/evoformer/block.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

function reference(input: GlobalAttentionInput): Float32Array {
  const { activations, mask, sequences, length, channels, weights } = input;
  const headDim = weights.gatingBias.length / weights.heads;
  if (channels !== 2 || weights.heads !== 1 || headDim !== 1) throw new Error("reference shape changed");
  const normalized = new Float32Array(activations.length);
  for (let row = 0; row < sequences * length; row += 1) {
    const x0 = activations[row * 2]!; const x1 = activations[row * 2 + 1]!;
    const mean = (x0 + x1) / 2;
    const inverseStd = 1 / Math.sqrt(((x0 - mean) ** 2 + (x1 - mean) ** 2) / 2 + 1e-5);
    normalized[row * 2] = (x0 - mean) * inverseStd;
    normalized[row * 2 + 1] = (x1 - mean) * inverseStd;
  }
  const output = new Float32Array(activations.length);
  for (let column = 0; column < length; column += 1) {
    let denominator = 1e-10; let average0 = 0; let average1 = 0;
    for (let sequence = 0; sequence < sequences; sequence += 1) {
      const keep = mask[sequence * length + column]!;
      const base = (sequence * length + column) * 2;
      denominator += keep;
      average0 += normalized[base]! * keep; average1 += normalized[base + 1]! * keep;
    }
    const query = (average0 / denominator) * weights.queryWeight[0]!
      + (average1 / denominator) * weights.queryWeight[1]!;
    let maximum = -1e30; let softmaxSum = 0; let attended = 0;
    for (let sequence = 0; sequence < sequences; sequence += 1) {
      const base = (sequence * length + column) * 2;
      const key = normalized[base]! * weights.keyWeight[0]!
        + normalized[base + 1]! * weights.keyWeight[1]!;
      const value = normalized[base]! * weights.valueWeight[0]!
        + normalized[base + 1]! * weights.valueWeight[1]!;
      const logit = mask[sequence * length + column] === 0 ? -1e9 : query * key;
      const nextMaximum = Math.max(maximum, logit);
      const previousScale = Math.exp(maximum - nextMaximum);
      const weight = Math.exp(logit - nextMaximum);
      softmaxSum = softmaxSum * previousScale + weight;
      attended = attended * previousScale + weight * value;
      maximum = nextMaximum;
    }
    attended /= softmaxSum;
    for (let sequence = 0; sequence < sequences; sequence += 1) {
      const base = (sequence * length + column) * 2;
      const gateLogit = weights.gatingBias[0]! + normalized[base]! * weights.gatingWeight[0]!
        + normalized[base + 1]! * weights.gatingWeight[1]!;
      const gated = attended / (1 + Math.exp(-gateLogit));
      output[base] = weights.outputBias[0]! + gated * weights.outputWeight[0]!;
      output[base + 1] = weights.outputBias[1]! + gated * weights.outputWeight[1]!;
    }
  }
  return output;
}

describe.skipIf(!enabled)("extra-MSA global attention large dispatch grid", () => {
  let device: GPUDevice;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.AFWEBGPU_ADAPTER;
    const adapter = await create(adapterName === undefined ? [] : [`adapter=${adapterName}`]).requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    device = await adapter.requestDevice();
  });

  afterAll(() => device?.destroy());

  it("matches an independent reference beyond 32,768 workgroups", async () => {
    const sequences = 1_025; const length = 2_048; const channels = 2;
    const activations = new Float32Array(sequences * length * channels);
    for (let row = 0; row < sequences * length; row += 1) {
      activations[row * 2] = ((row % 31) - 15) / 8;
      activations[row * 2 + 1] = ((row % 17) - 8) / 7;
    }
    const weights: GlobalAttentionWeights = {
      heads: 1,
      queryNormScale: new Float32Array([1, 1]), queryNormOffset: new Float32Array(2),
      queryWeight: new Float32Array([0.3, -0.2]),
      keyWeight: new Float32Array([0.4, 0.1]), valueWeight: new Float32Array([0.2, -0.5]),
      gatingWeight: new Float32Array([0.15, -0.25]), gatingBias: new Float32Array([0.05]),
      outputWeight: new Float32Array([0.7, -0.4]), outputBias: new Float32Array([0.1, -0.2]),
    };
    const input: GlobalAttentionInput = {
      activations, mask: new Float32Array(sequences * length).fill(1),
      sequences, length, channels, weights,
    };
    const expected = reference(input);
    const actual = await new GlobalAttentionGpu(device).run(input);
    const error = errorMetrics(actual.output, expected);
    expect(error.meanAbsoluteError).toBeLessThan(2e-5);
    expect(error.maxAbsoluteError).toBeLessThan(2e-4);
  });
});
