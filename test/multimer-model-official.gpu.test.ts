import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  AlphaFoldMultimerGpu, type MultimerModelWeights, type MultimerPrediction,
} from "../src/model/multimer.js";
import type { MultimerRecycleFeatures } from "../src/input/multimer-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const referenceManifest = process.env.AFWEBGPU_MULTIMER_REFERENCE;
const float32Manifest = process.env.AFWEBGPU_MULTIMER_F32_MANIFEST;
const q8Manifest = process.env.AFWEBGPU_MULTIMER_Q8_MANIFEST;
const gpuEnabled = process.env.AFWEBGPU_GPU_TESTS === "1";

interface MultimerReferenceManifest {
  readonly recycles: number;
  readonly reference: {
    readonly recycleMetrics: readonly {
      readonly recycle: number; readonly meanPlddt: number; readonly ptm: number;
      readonly iptm: number; readonly rankingConfidence: number;
    }[];
  };
}

async function multimerWeights(fixture: AlphaFoldFixture): Promise<MultimerModelWeights> {
  const [embedding, extraStack, mainStack, structure, confidence, geometry] = await Promise.all([
    fixture.multimerEmbeddingWeights(), fixture.extraStackWeights(), fixture.mainStackWeights(),
    fixture.multimerStructureWeights(), fixture.confidenceWeights(), fixture.geometryTables(),
  ]);
  return { embedding, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry };
}

async function capturedFeatures(
  reference: FileTensorStore,
  model: AlphaFoldFixture,
): Promise<readonly MultimerRecycleFeatures[]> {
  const manifest = reference.manifest as unknown as MultimerReferenceManifest;
  const tables = await model.queryOnlyFeatureTables();
  const results: MultimerRecycleFeatures[] = [];
  for (let recycle = 0; recycle <= manifest.recycles; recycle += 1) {
    const tensor = (name: string): Promise<Float32Array> => reference.tensor(`feature_${name}_recycle${recycle}`);
    const msaName = `feature_msa_feat_recycle${recycle}`;
    const extraName = `feature_extra_msa_recycle${recycle}`;
    const aatype = await tensor("aatype");
    const length = aatype.length;
    const atom37ToAtom14 = new Float32Array(length * 37);
    const atom37Mask = new Float32Array(length * 37);
    for (let residue = 0; residue < length; residue += 1) {
      const aa = aatype[residue]!;
      atom37ToAtom14.set(tables.atom37ToAtom14.subarray(aa * 37, (aa + 1) * 37), residue * 37);
      atom37Mask.set(tables.atom37Mask.subarray(aa * 37, (aa + 1) * 37), residue * 37);
    }
    results.push({
      targetFeatures: await tensor("target_feat"), msaFeatures: await tensor("msa_feat"),
      msaMask: await tensor("msa_mask"), extraMsa: await tensor("extra_msa"),
      extraHasDeletion: await tensor("extra_has_deletion"),
      extraDeletionValue: await tensor("extra_deletion_value"), extraMsaMask: await tensor("extra_msa_mask"),
      residueIndex: await tensor("residue_index"), aatype, seqMask: await tensor("seq_mask"),
      atom37ToAtom14, atom37Mask,
      msaSequences: reference.shape(msaName)[0]!, extraSequences: reference.shape(extraName)[0]!,
      targetChannels: reference.shape(`feature_target_feat_recycle${recycle}`).at(-1)!,
      msaFeatureChannels: reference.shape(msaName).at(-1)!,
      chainRelative: {
        asymId: await tensor("asym_id"), entityId: await tensor("entity_id"), symId: await tensor("sym_id"),
      },
    });
  }
  return results;
}

async function predict(
  device: GPUDevice,
  modelManifest: string,
  reference: FileTensorStore,
): Promise<MultimerPrediction> {
  const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(modelManifest));
  return new AlphaFoldMultimerGpu(device, {
    compactTransitions: true, recycleEarlyStopTolerance: -1,
  }).predict(
    await capturedFeatures(reference, model), await multimerWeights(model), await model.tensor("confidencePaeBreaks"),
  );
}

describe.skipIf(!(gpuEnabled && referenceManifest !== undefined && float32Manifest !== undefined))(
  "official AlphaFold-Multimer-v3 model-1 end-to-end reference",
  () => {
    let device: GPUDevice;
    let reference: FileTensorStore;
    beforeAll(async () => {
      Object.assign(globalThis, globals);
      const adapter = await create([]).requestAdapter({ powerPreference: "high-performance" });
      if (adapter === null) throw new Error("no WebGPU adapter");
      device = await adapter.requestDevice();
      reference = await FileTensorStore.open(referenceManifest!);
    });
    afterAll(() => device?.destroy());

    it("matches official float32 confidence, PAE, and atom coordinates", async () => {
      const prediction = await predict(device, float32Manifest!, reference);
      const manifest = reference.manifest as unknown as MultimerReferenceManifest;
      expect(prediction.recycles).toHaveLength(manifest.recycles + 1);
      for (let recycle = 0; recycle < prediction.recycles.length; recycle += 1) {
        const actual = prediction.recycles[recycle]!.confidence;
        const expected = manifest.reference.recycleMetrics[recycle]!;
        expect(Math.abs(actual.meanPlddt - expected.meanPlddt)).toBeLessThan(0.05);
        expect(Math.abs(actual.ptm - expected.ptm)).toBeLessThan(0.001);
        expect(Math.abs(actual.iptm - expected.iptm)).toBeLessThan(0.001);
        expect(Math.abs(actual.rankingConfidence - expected.rankingConfidence)).toBeLessThan(0.001);
      }
      expect(errorMetrics(prediction.final.confidence.plddt,
        await reference.tensor("predictionPlddt")).meanAbsoluteError).toBeLessThan(0.5);
      expect(errorMetrics(prediction.final.confidence.predictedAlignedError,
        await reference.tensor("predictionPae")).meanAbsoluteError).toBeLessThan(0.5);
      expect(errorMetrics(prediction.final.structure.atom37,
        await reference.tensor("predictionAtom37")).meanAbsoluteError).toBeLessThan(0.25);
    }, 300_000);
  },
);

describe.skipIf(!(gpuEnabled && referenceManifest !== undefined
  && float32Manifest !== undefined && q8Manifest !== undefined))(
  "quantized AlphaFold-Multimer-v3 model 1",
  () => {
    let device: GPUDevice;
    let reference: FileTensorStore;
    beforeAll(async () => {
      Object.assign(globalThis, globals);
      const adapter = await create([]).requestAdapter({ powerPreference: "high-performance" });
      if (adapter === null) throw new Error("no WebGPU adapter");
      device = await adapter.requestDevice();
      reference = await FileTensorStore.open(referenceManifest!);
    });
    afterAll(() => device?.destroy());

    it("stays within the predeclared float32 comparability envelope", async () => {
      const float32 = await predict(device, float32Manifest!, reference);
      const q8 = await predict(device, q8Manifest!, reference);
      expect(q8.recycles).toHaveLength(float32.recycles.length);
      for (let recycle = 0; recycle < q8.recycles.length; recycle += 1) {
        expect(Math.abs(q8.recycles[recycle]!.confidence.meanPlddt
          - float32.recycles[recycle]!.confidence.meanPlddt)).toBeLessThan(0.25);
        expect(Math.abs(q8.recycles[recycle]!.confidence.ptm
          - float32.recycles[recycle]!.confidence.ptm)).toBeLessThan(0.005);
        expect(Math.abs(q8.recycles[recycle]!.confidence.iptm
          - float32.recycles[recycle]!.confidence.iptm)).toBeLessThan(0.005);
      }
      expect(errorMetrics(q8.final.confidence.plddt,
        float32.final.confidence.plddt).meanAbsoluteError).toBeLessThan(0.5);
      expect(errorMetrics(q8.final.confidence.predictedAlignedError,
        float32.final.confidence.predictedAlignedError).meanAbsoluteError).toBeLessThan(0.5);
      expect(errorMetrics(q8.final.structure.atom37,
        float32.final.structure.atom37).rootMeanSquareError).toBeLessThan(0.5);
    }, 600_000);
  },
);
