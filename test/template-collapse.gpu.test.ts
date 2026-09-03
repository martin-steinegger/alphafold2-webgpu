import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu, EXACT_STORAGE } from "../src/model/monomer.js";
import { iterateA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { queryOnlyTemplateConstant, QueryOnlyTemplateGpu } from "../src/evoformer/template.js";
import { errorMetrics } from "../src/triangle/types.js";

/**
 * The query-only template branch is a constant, and collapsing it must not
 * move the prediction.
 */
const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = process.env.AFWEBGPU_MODEL_MANIFEST
  ?? "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
const QUERY = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

describe.skipIf(!enabled)("query-only template collapse", () => {
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

  it("reproduces the module's update from one probed vector", async () => {
    const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const weights = await model.templateWeights();
    const constant = await queryOnlyTemplateConstant(device, weights);
    expect(constant).toBeDefined();
    const length = 59;
    const full = await new QueryOnlyTemplateGpu(device).run({
      length, templateChannels: 64, pairChannels: 128,
      pairMask: new Float32Array(length * length).fill(1), weights,
    });
    const broadcast = new Float32Array(length * length * 128);
    for (let pair = 0; pair < length * length; pair += 1) broadcast.set(constant!, pair * 128);
    const metrics = errorMetrics(broadcast, full.pairUpdate);
    // The probe runs on eight residues, so only reduction order differs.
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-6);
  }, 600_000);

  it("leaves the prediction where the module put it", async () => {
    const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(MANIFEST));
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry, tables, paeBreaks] =
      await Promise.all([
        model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(), model.mainStackWeights(),
        model.structureWeights(), model.confidenceWeights(), model.geometryTables(), model.queryOnlyFeatureTables(),
        model.tensor("confidencePaeBreaks"),
      ]);
    const weights = { embedding, template, extraStack, mainStack, structure,
      lddt: confidence.lddt, pae: confidence.pae, geometry };
    const a3m = `>query\n${QUERY}\n`;
    // Both runs use the exact storages so only the template differs; packing is
    // compared on its own in test/pair-storage.gpu.test.ts.
    const run = async (collapseQueryOnlyTemplate: boolean) => new AlphaFoldMonomerGpu(device, {
      collapseQueryOnlyTemplate, ...EXACT_STORAGE,
    }).predict(iterateA3mFeatures(a3m, tables, { recycles: 0, randomSeed: 0 }), weights, paeBreaks);
    const collapsed = await run(true);
    const exact = await run(false);
    expect(collapsed.final.confidence.meanPlddt).toBeCloseTo(exact.final.confidence.meanPlddt, 2);
    expect(collapsed.final.confidence.ptm).toBeCloseTo(exact.final.confidence.ptm, 4);
    const atoms = errorMetrics(collapsed.final.structure.atom37, exact.final.structure.atom37);
    expect(atoms.maxAbsoluteError).toBeLessThan(1e-2);
    const plddt = errorMetrics(collapsed.final.confidence.plddt, exact.final.confidence.plddt);
    expect(plddt.maxAbsoluteError).toBeLessThan(1e-1);
  }, 900_000);
});
