import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { EvoformerBlockGpu } from "../src/evoformer/block.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { planShards } from "../src/runtime/sharded.js";
import { errorMetrics } from "../src/triangle/types.js";

/**
 * A pair too large for one binding is bound as several windows of the same
 * buffer. Splitting a binding changes nothing a shader computes, so the block
 * must return exactly what it returns unsharded. No device here has a limit
 * low enough to trigger it, so the budget is set by hand.
 */
const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("pair binding shards", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: Math.min(adapter.limits.maxStorageBuffersPerShaderStage, 16),
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
  }, 120_000);
  afterAll(() => device?.destroy());

  it("returns the same block whether the pair is one binding or several", async () => {
    const store = await FileTensorStore.open(MANIFEST);
    const fixture = AlphaFoldFixture.fromStore(store);
    const shape = store.shape("blockInputMsa");
    const weights = (await fixture.mainStackWeights())[0]!;
    const descriptor = {
      msa: await store.tensor("blockInputMsa"),
      pair: await store.tensor("blockInputPair"),
      msaMask: await store.tensor("blockMsaMask"),
      pairMask: await store.tensor("blockPairMask"),
      sequences: shape[0]!, length: shape[1]!, cM: shape[2]!, cZ: 128,
      cOuter: weights.outerProductMean.leftBias.length,
      triangleHidden: weights.triangleMultiplicationOutgoing.linearAPBias.length,
      weights,
    };
    // The budget below has to actually split both tensors, or the sharded run
    // would be the whole-binding run under another name.
    // Small enough to split the pair, the MSA and the triangle's whole
    // operand several ways, and large enough that the windows of the pair and
    // of the operand still fit the stage's storage slots together.
    const budget = 512 * 1024;
    expect(planShards(descriptor.length * descriptor.length * descriptor.cZ, descriptor.cZ, budget).count)
      .toBeGreaterThan(1);
    const pairs = descriptor.length * descriptor.length;
    expect(planShards((pairs + pairs % 2) * descriptor.triangleHidden, 2, budget).count).toBeGreaterThan(1);
    // The MSA here is smaller than the pair, so it gets a budget of its own.
    const msaBudget = 128 * 1024;
    expect(planShards(descriptor.sequences * descriptor.length * descriptor.cM, descriptor.cM, msaBudget).count)
      .toBeGreaterThan(1);
    const whole = await new EvoformerBlockGpu(device).run(descriptor);
    // 59 residues hold a 1.8 MiB pair, which this budget splits four ways.
    const sharded = await new EvoformerBlockGpu(device).run({
      ...descriptor, pairBindingBytes: budget, msaBindingBytes: msaBudget,
    });
    expect(errorMetrics(sharded.pair, whole.pair).maxAbsoluteError).toBe(0);
    expect(errorMetrics(sharded.msa, whole.msa).maxAbsoluteError).toBe(0);
  }, 600_000);
});
