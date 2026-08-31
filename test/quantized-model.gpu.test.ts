import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { AlphaFoldQueryOnlyGpu } from "../src/model/query-only.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const quantizedManifest = process.env.AFWEBGPU_QUANTIZED_MANIFEST;
const enabled = process.env.AFWEBGPU_GPU_TESTS === "1" && quantizedManifest !== undefined;
const REFERENCE_MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
const SEQUENCE = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

describe.skipIf(!enabled)("quantized end-to-end AlphaFold model", () => {
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

  it("stays within the documented comparability envelope", async () => {
    const reference = AlphaFoldFixture.fromStore(await FileTensorStore.open(REFERENCE_MANIFEST));
    const quantized = AlphaFoldFixture.fromStore(await FileTensorStore.open(quantizedManifest!));
    const maskedMsaCodes: Float32Array[] = [];
    for (let recycle = 0; recycle < 4; recycle += 1) {
      const msa = await reference.tensor(`feature_msa_feat_recycle${recycle}`);
      const codes = new Float32Array(SEQUENCE.length);
      for (let residue = 0; residue < SEQUENCE.length; residue += 1) {
        let code = 0;
        for (let channel = 1; channel < 23; channel += 1) {
          if (msa[residue * 49 + channel]! > msa[residue * 49 + code]!) code = channel;
        }
        codes[residue] = code;
      }
      maskedMsaCodes.push(codes);
    }
    const loadWeights = async (fixture: AlphaFoldFixture) => {
      const [embedding, template, extraStack, mainStack, structure, confidence, geometry] = await Promise.all([
        fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraPairStackWeights(),
        fixture.mainStackWeights(), fixture.structureWeights(), fixture.confidenceWeights(), fixture.geometryTables(),
      ]);
      return { embedding, template, extraStack, mainStack, structure,
        lddt: confidence.lddt, pae: confidence.pae, geometry };
    };
    const options = { recycles: 3, maskedMsaCodes } as const;
    const tables = await reference.queryOnlyFeatureTables();
    const runner = new AlphaFoldQueryOnlyGpu(device);
    const float32 = await runner.predictSequence(
      SEQUENCE, await loadWeights(reference), tables, options, await reference.tensor("confidencePaeBreaks"),
    );
    const q8 = await runner.predictSequence(
      SEQUENCE, await loadWeights(quantized), tables, options, await quantized.tensor("confidencePaeBreaks"),
    );
    for (let recycle = 0; recycle < 4; recycle += 1) {
      expect(Math.abs(q8.recycles[recycle]!.confidence.meanPlddt
        - float32.recycles[recycle]!.confidence.meanPlddt)).toBeLessThan(0.25);
      expect(Math.abs(q8.recycles[recycle]!.confidence.ptm
        - float32.recycles[recycle]!.confidence.ptm)).toBeLessThan(0.005);
    }
    const plddt = errorMetrics(q8.final.confidence.plddt, float32.final.confidence.plddt);
    const pae = errorMetrics(
      q8.final.confidence.predictedAlignedError, float32.final.confidence.predictedAlignedError,
    );
    const atoms = errorMetrics(q8.final.structure.atom37, float32.final.structure.atom37);
    process.stdout.write(`${JSON.stringify({
      quantizedModelComparison: {
        recycles: q8.recycles.map((result, recycle) => ({
          recycle,
          meanPlddtDelta: result.confidence.meanPlddt - float32.recycles[recycle]!.confidence.meanPlddt,
          ptmDelta: result.confidence.ptm - float32.recycles[recycle]!.confidence.ptm,
        })),
        plddt, pae, atoms,
      },
    })}\n`);
    expect(plddt.meanAbsoluteError).toBeLessThan(0.5);
    expect(pae.meanAbsoluteError).toBeLessThan(0.5);
    expect(atoms.rootMeanSquareError).toBeLessThan(0.5);
  }, 240_000);
});
