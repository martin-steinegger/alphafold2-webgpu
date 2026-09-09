import { CLUSTERED_MSA_CHANNELS, expandClusteredMsaFeatures } from "../src/input/msa-features.js";
import { readFile } from "node:fs/promises";
import { beforeAll, describe, expect, it, afterAll } from "vitest";
import { iterateA3mFeatures, makeA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { create, globals } from "webgpu";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

// Featurisation clusters the alignment on the device, so these need one.
const gpuEnabled = process.env.AFWEBGPU_GPU_TESTS === "1";
// Held at module scope rather than left as a local in beforeAll. dawn.node
// schedules InstanceBase::ProcessEvents on the event loop, and a callback that
// runs after the instance is collected dereferences freed memory: a
// segmentation fault inside pthread_mutex_lock on an unaligned mutex.
let gpu: GPU;
let device: GPUDevice;
beforeAll(async () => {
  if (!gpuEnabled) return;
  Object.assign(globalThis, globals);
  gpu = create(dawnInstanceFlags({ unclamped: true }));
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  device = await requestAlphaFoldDevice(adapter!);
});

// dawn.node pumps ProcessEvents from the event loop and deadlocks on the
// instance mutex if the device outlives the worker's threads, which shows up
// as a vitest worker exiting unexpectedly rather than as a failure.
afterAll(() => { device?.destroy(); });


describe.skipIf(!gpuEnabled)("A3M model feature preprocessing", () => {
  it("streams deterministic recycles and shares immutable sequence tensors", async () => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(
      "test/fixtures/evoformer/model1-query-59-stack/manifest.json",
    ));
    const source = iterateA3mFeatures(device, ">query\nACGG\n>homolog\nA-GG\n",
      await fixture.queryOnlyFeatureTables(), { recycles: 2, randomSeed: 7 });
    expect(source.length).toBe(3);
    const iterator = source[Symbol.asyncIterator]();
    const first = (await iterator.next()).value!;
    const second = (await iterator.next()).value!;
    expect(first.aatype).toBe(second.aatype);
    expect(first.targetFeatures).toBe(second.targetFeatures);
    const drain = async (): Promise<number[][]> => {
      const out: number[][] = [];
      for await (const features of source) out.push([...features.msaFeatures]);
      return out;
    };
    expect(await drain()).toEqual(await drain());
  });

  it("keeps block padding as gaps while excluding it from masked-MSA augmentation", async () => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open("test/fixtures/evoformer/model1-query-59-stack/manifest.json"));
    const result = await makeA3mFeatures(device, ">query\nACGG\n>chain\nA---\n", await fixture.queryOnlyFeatureTables(), {
      recycles: 0, randomSeed: 0, maxMsaSequences: 2,
      alignmentMask: Float32Array.of(1, 1, 1, 1, 1, 1, 0, 0),
    });
    const features = result[0]!;
    expect([...features.msaMask]).toEqual(new Array(8).fill(1));
    // Rows six and seven are block padding: gaps, and never masked-augmented.
    const dense = expandClusteredMsaFeatures(features.msaFeatures, features.msaSequences * 4);
    expect(dense[6 * 49 + 21]).toBe(1);
    expect(dense[7 * 49 + 21]).toBe(1);
  });

  it("clusters the uploaded 8,076-row alignment into model-1 tensors", async () => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open("test/fixtures/evoformer/model1-query-59-stack/manifest.json"));
    const result = await makeA3mFeatures(device, await readFile("test.a3m", "utf8"), await fixture.queryOnlyFeatureTables(), {
      recycles: 0, randomSeed: 0,
    });
    const features = result[0]!;
    expect(features.msaSequences).toBe(508);
    expect(features.extraSequences).toBe(1024);
    expect(features.msaFeatures.length).toBe(508 * 59 * CLUSTERED_MSA_CHANNELS);
    expect(features.extraMsa.length).toBe(1024 * 59);
    expect(features.msaMask.every((value) => value === 1)).toBe(true);
    expect(features.extraMsaMask.every((value) => value === 1)).toBe(true);
  }, 30_000);
});
