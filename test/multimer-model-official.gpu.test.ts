import { CLUSTERED_MSA_CHANNELS, compactClusteredMsaFeatures } from "../src/input/msa-features.js";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  AlphaFoldMultimerGpu, type MultimerModelWeights, type MultimerPrediction,
} from "../src/model/multimer.js";
import type { MultimerRecycleFeatures } from "../src/input/multimer-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const referenceManifests = (process.env.AFWEBGPU_MULTIMER_REFERENCES
  ?? process.env.AFWEBGPU_MULTIMER_REFERENCE ?? "")
  .split(",").map((value) => value.trim()).filter((value) => value !== "");
const float32Manifest = process.env.AFWEBGPU_MULTIMER_F32_MANIFEST;
const compressedManifest = process.env.AFWEBGPU_MULTIMER_COMPRESSED_MANIFEST
  ?? process.env.AFWEBGPU_MULTIMER_Q8_MANIFEST;
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
  // Load large f32 shards sequentially so native WebGPU test workers do not
  // transiently retain duplicate shard buffers while several readers resolve.
  const embedding = await fixture.multimerEmbeddingWeights();
  const multimerTemplate = await fixture.multimerTemplateWeights();
  const extraStack = await fixture.extraStackWeights();
  const mainStack = await fixture.mainStackWeights();
  const structure = await fixture.multimerStructureWeights();
  const confidence = await fixture.confidenceWeights();
  const geometry = await fixture.geometryTables();
  return { embedding, multimerTemplate, extraStack, mainStack, structure,
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
      targetFeatures: await tensor("target_feat"),
      msaFeatures: compactClusteredMsaFeatures(await tensor("msa_feat"),
        reference.shape(msaName)[0]! * reference.shape(msaName)[1]!),
      msaMask: await tensor("msa_mask"), extraMsa: await tensor("extra_msa"),
      extraHasDeletion: await tensor("extra_has_deletion"),
      extraDeletionValue: await tensor("extra_deletion_value"), extraMsaMask: await tensor("extra_msa_mask"),
      residueIndex: await tensor("residue_index"), aatype, seqMask: await tensor("seq_mask"),
      atom37ToAtom14, atom37Mask,
      msaSequences: reference.shape(msaName)[0]!, extraSequences: reference.shape(extraName)[0]!,
      targetChannels: reference.shape(`feature_target_feat_recycle${recycle}`).at(-1)!,
      msaFeatureChannels: CLUSTERED_MSA_CHANNELS,
      chainRelative: {
        asymId: await tensor("asym_id"), entityId: await tensor("entity_id"), symId: await tensor("sym_id"),
      },
    });
  }
  return results;
}

interface PreparedPrediction {
  readonly features: readonly MultimerRecycleFeatures[];
  readonly weights: MultimerModelWeights;
  readonly paeBreaks: Float32Array;
}

async function preparePrediction(
  modelManifest: string,
  reference: FileTensorStore,
): Promise<PreparedPrediction> {
  const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(modelManifest));
  return {
    features: await capturedFeatures(reference, model),
    weights: await multimerWeights(model),
    paeBreaks: await model.tensor("confidencePaeBreaks"),
  };
}

async function predict(device: GPUDevice, prepared: PreparedPrediction): Promise<MultimerPrediction> {
  return new AlphaFoldMultimerGpu(device, {
    compactTransitions: true, recycleEarlyStopTolerance: -1,
  }).predict(prepared.features, prepared.weights, prepared.paeBreaks);
}

for (const referenceManifest of referenceManifests.length > 0 ? referenceManifests : [undefined]) {
  const referenceLabel = referenceManifest?.split("/").at(-2) ?? "missing reference";
describe.skipIf(!(gpuEnabled && referenceManifest !== undefined && float32Manifest !== undefined))(
  `official AlphaFold-Multimer-v3 model-1 end-to-end reference (${referenceLabel})`,
  () => {
    let device: GPUDevice;
    let reference: FileTensorStore;
    let prepared: PreparedPrediction;
    beforeAll(async () => {
      reference = await FileTensorStore.open(referenceManifest!);
      prepared = await preparePrediction(float32Manifest!, reference);
      Object.assign(globalThis, globals);
      const adapter = await create([]).requestAdapter();
      if (adapter === null) throw new Error("no WebGPU adapter");
      device = await adapter.requestDevice();
    });
    afterAll(() => device?.destroy());

    it("matches official float32 confidence, PAE, and atom coordinates", async () => {
      const prediction = await predict(device, prepared);
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
  && float32Manifest !== undefined && compressedManifest !== undefined))(
  `compressed AlphaFold-Multimer-v3 model 1 (${referenceLabel})`,
  () => {
    let device: GPUDevice;
    let reference: FileTensorStore;
    let float32Prepared: PreparedPrediction;
    let compressedPrepared: PreparedPrediction;
    beforeAll(async () => {
      reference = await FileTensorStore.open(referenceManifest!);
      float32Prepared = await preparePrediction(float32Manifest!, reference);
      compressedPrepared = await preparePrediction(compressedManifest!, reference);
      Object.assign(globalThis, globals);
      const adapter = await create([]).requestAdapter();
      if (adapter === null) throw new Error("no WebGPU adapter");
      device = await adapter.requestDevice();
    });
    afterAll(() => device?.destroy());

    it("stays within the predeclared float32 comparability envelope", async () => {
      const float32 = await predict(device, float32Prepared);
      const compressed = await predict(device, compressedPrepared);
      expect(compressed.recycles).toHaveLength(float32.recycles.length);
      for (let recycle = 0; recycle < compressed.recycles.length; recycle += 1) {
        expect(Math.abs(compressed.recycles[recycle]!.confidence.meanPlddt
          - float32.recycles[recycle]!.confidence.meanPlddt)).toBeLessThan(0.25);
        expect(Math.abs(compressed.recycles[recycle]!.confidence.ptm
          - float32.recycles[recycle]!.confidence.ptm)).toBeLessThan(0.005);
        expect(Math.abs(compressed.recycles[recycle]!.confidence.iptm
          - float32.recycles[recycle]!.confidence.iptm)).toBeLessThan(0.005);
      }
      expect(errorMetrics(compressed.final.confidence.plddt,
        float32.final.confidence.plddt).meanAbsoluteError).toBeLessThan(0.5);
      expect(errorMetrics(compressed.final.confidence.predictedAlignedError,
        float32.final.confidence.predictedAlignedError).meanAbsoluteError).toBeLessThan(0.5);
      expect(errorMetrics(compressed.final.structure.atom37,
        float32.final.structure.atom37).rootMeanSquareError).toBeLessThan(0.5);
    }, 600_000);
  },
);
}
