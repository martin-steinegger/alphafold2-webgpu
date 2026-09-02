import { readFile } from "node:fs/promises";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { iterateA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

/**
 * The packed pair has to reach every consumer. It crosses three interfaces on
 * its way to the structure module, and a dropped option there reads packed
 * words as f32 without failing, so this compares a whole prediction rather
 * than a kernel: pLDDT collapses if any consumer misses the storage.
 */
const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = process.env.AFWEBGPU_MODEL_MANIFEST
  ?? "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

describe.skipIf(!enabled)("packed pair storage", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
  }, 120_000);
  afterAll(() => device?.destroy());

  it("predicts what the exact pair predicts", async () => {
    const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry, tables, paeBreaks] =
      await Promise.all([
        model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(), model.mainStackWeights(),
        model.structureWeights(), model.confidenceWeights(), model.geometryTables(), model.queryOnlyFeatureTables(),
        model.tensor("confidencePaeBreaks"),
      ]);
    const weights = { embedding, template, extraStack, mainStack, structure,
      lddt: confidence.lddt, pae: confidence.pae, geometry };
    const a3m = await readFile("test.a3m", "utf8");
    const run = async (pairStorage: "f32" | "f16") => new AlphaFoldMonomerGpu(device, { pairStorage }).predict(
      iterateA3mFeatures(a3m, tables, { recycles: 1, randomSeed: 0, maxMsaSequences: 64, maxExtraSequences: 128 }),
      weights, paeBreaks,
    );
    const exact = await run("f32");
    const packed = await run("f16");

    // Rounding every write to the pair moves the prediction by far less than
    // the confidence's own resolution; reading it wrongly moves it by tens.
    expect(packed.final.confidence.meanPlddt).toBeCloseTo(exact.final.confidence.meanPlddt, 0);
    expect(packed.final.confidence.ptm).toBeCloseTo(exact.final.confidence.ptm, 2);
    const atoms = errorMetrics(packed.final.structure.atom37, exact.final.structure.atom37);
    expect(atoms.meanAbsoluteError).toBeLessThan(0.1);
    // The packed pair is half the size, which is the point of the option.
    expect(packed.memory.peakBytes).toBeLessThan(exact.memory.peakBytes);
  }, 900_000);
});
