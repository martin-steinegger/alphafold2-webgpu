import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { MultimerRelativePositionGpu } from "../src/evoformer/multimer-relative.js";
import { MULTIMER_RELATIVE_CHANNELS, multimerChainIdentifiers } from "../src/input/multimer-features.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

describe.skipIf(!enabled)("AlphaFold-Multimer relative projection WebGPU", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals); gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("matches an independent explicit one-hot projection", async () => {
    const ids = multimerChainIdentifiers(["ACD", "ACD", "G"]);
    const length = ids.residueIndex.length;
    const channels = 7;
    const weight = Float32Array.from({ length: MULTIMER_RELATIVE_CHANNELS * channels },
      (_, index) => Math.sin(index * 0.31) * 0.05);
    const bias = Float32Array.from({ length: channels }, (_, index) => Math.cos(index * 0.7) * 0.1);
    const expected = new Float32Array(length * length * channels);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      const sameChain = ids.asymId[i] === ids.asymId[j];
      const relative = sameChain ? Math.max(0, Math.min(64, ids.residueIndex[i]! - ids.residueIndex[j]! + 32)) : 65;
      const sameEntity = ids.entityId[i] === ids.entityId[j];
      const relativeChain = sameEntity ? Math.max(0, Math.min(4, ids.symId[i]! - ids.symId[j]! + 2)) : 5;
      for (let channel = 0; channel < channels; channel += 1) {
        expected[(i * length + j) * channels + channel] = bias[channel]!
          + weight[relative * channels + channel]!
          + (sameEntity ? weight[66 * channels + channel]! : 0)
          + weight[(67 + relativeChain) * channels + channel]!;
      }
    }
    const actual = await new MultimerRelativePositionGpu(device).run({
      ...ids, length, pairChannels: channels, weight, bias,
    });
    const metrics = errorMetrics(actual.output, expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(2e-8);
    expect(metrics.maxAbsoluteError).toBeLessThan(2e-7);
  });
});
