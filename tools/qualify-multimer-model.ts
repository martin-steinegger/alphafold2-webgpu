import { create, globals } from "webgpu";
import { AlphaFoldMultimerGpu, type MultimerModelWeights } from "../src/model/multimer.js";
import type { MultimerRecycleFeatures } from "../src/input/multimer-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";
import { InputEmbedderGpu } from "../src/evoformer/input-embedder.js";

interface ReferenceManifest {
  readonly recycles: number;
  readonly reference: {
    readonly recycleMetrics: readonly {
      readonly meanPlddt: number; readonly ptm: number; readonly iptm: number;
      readonly rankingConfidence: number;
    }[];
  };
}

function required(name: string): string {
  const value = process.env[name]?.trim();
  if (value === undefined || value === "") throw new Error(`${name} is required`);
  return value;
}

function within(label: string, actual: number, expected: number, threshold: number): void {
  const difference = Math.abs(actual - expected);
  if (!(difference < threshold)) {
    throw new Error(`${label}: |${actual} - ${expected}| = ${difference}, limit ${threshold}`);
  }
}

function metricWithin(label: string, actual: Float32Array, expected: Float32Array,
  metric: "meanAbsoluteError" | "rootMeanSquareError", threshold: number): void {
  const value = errorMetrics(actual, expected)[metric];
  if (!(value < threshold)) throw new Error(`${label}: ${metric} ${value}, limit ${threshold}`);
}

async function multimerWeights(fixture: AlphaFoldFixture): Promise<MultimerModelWeights> {
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

async function capturedFeatures(reference: FileTensorStore, model: AlphaFoldFixture):
Promise<readonly MultimerRecycleFeatures[]> {
  const manifest = reference.manifest as unknown as ReferenceManifest;
  const tables = await model.queryOnlyFeatureTables();
  const results: MultimerRecycleFeatures[] = [];
  for (let recycle = 0; recycle <= manifest.recycles; recycle += 1) {
    const tensor = (name: string): Promise<Float32Array> =>
      reference.tensor(`feature_${name}_recycle${recycle}`);
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

const referencePaths = required("AFWEBGPU_MULTIMER_REFERENCES").split(",")
  .map((value) => value.trim()).filter((value) => value !== "");
if (referencePaths.length === 0) throw new Error("at least one Multimer reference is required");
const float32Model = AlphaFoldFixture.fromStore(await FileTensorStore.open(
  required("AFWEBGPU_MULTIMER_F32_MANIFEST"),
));
const compressedManifest = (process.env.AFWEBGPU_MULTIMER_COMPRESSED_MANIFEST
  ?? process.env.AFWEBGPU_MULTIMER_Q8_MANIFEST)?.trim();
const compressedModel = compressedManifest === undefined || compressedManifest === "" ? undefined
  : AlphaFoldFixture.fromStore(await FileTensorStore.open(compressedManifest));
console.log("Loading model 1 f32 weights before WebGPU device creation...");
const float32Weights = await multimerWeights(float32Model);
if (compressedModel === undefined) console.log("Skipping compressed comparison because its manifest is unset.");
else console.log("Loading model 1 compressed weights before WebGPU device creation...");
const compressedWeights = compressedModel === undefined ? undefined : await multimerWeights(compressedModel);
const float32Breaks = await float32Model.tensor("confidencePaeBreaks");
const compressedBreaks = compressedModel === undefined
  ? undefined : await compressedModel.tensor("confidencePaeBreaks");
const references = await Promise.all(referencePaths.map(async (path) => {
  const store = await FileTensorStore.open(path);
  return {
    path, store, manifest: store.manifest as unknown as ReferenceManifest,
    features: await capturedFeatures(store, float32Model),
  };
}));

Object.assign(globalThis, globals);
const adapter = await create([]).requestAdapter();
if (adapter === null) throw new Error("no WebGPU adapter");
const device = await adapter.requestDevice();
try {
  for (const reference of references) {
    const first = reference.features[0]!;
    console.log(`Preflighting Multimer input embedding for ${reference.path}...`);
    await new InputEmbedderGpu(device).run({
      ...first,
      previousMsaFirstRow: new Float32Array(first.aatype.length * 256),
      previousPair: new Float32Array(first.aatype.length * first.aatype.length * 128),
      previousPositions: new Float32Array(first.aatype.length * 37 * 3),
      length: first.aatype.length, msaChannels: 256, pairChannels: 128, extraMsaChannels: 64,
      weights: float32Weights.embedding, chainRelative: first.chainRelative,
    });
    console.log("Multimer input embedding passed.");
    console.log(`Qualifying ${reference.path} against official JAX...`);
    const f32 = await new AlphaFoldMultimerGpu(device, {
      compactTransitions: true, recycleEarlyStopTolerance: -1,
    }).predict(reference.features, float32Weights, float32Breaks);
    for (let recycle = 0; recycle < f32.recycles.length; recycle += 1) {
      const actual = f32.recycles[recycle]!.confidence;
      const expected = reference.manifest.reference.recycleMetrics[recycle]!;
      within(`f32 recycle ${recycle} mean pLDDT`, actual.meanPlddt, expected.meanPlddt, 0.05);
      within(`f32 recycle ${recycle} pTM`, actual.ptm, expected.ptm, 0.001);
      within(`f32 recycle ${recycle} ipTM`, actual.iptm, expected.iptm, 0.001);
      within(`f32 recycle ${recycle} ranking`, actual.rankingConfidence, expected.rankingConfidence, 0.001);
    }
    metricWithin("f32 pLDDT", f32.final.confidence.plddt,
      await reference.store.tensor("predictionPlddt"), "meanAbsoluteError", 0.5);
    metricWithin("f32 PAE", f32.final.confidence.predictedAlignedError,
      await reference.store.tensor("predictionPae"), "meanAbsoluteError", 0.5);
    metricWithin("f32 atom37", f32.final.structure.atom37,
      await reference.store.tensor("predictionAtom37"), "meanAbsoluteError", 0.25);

    if (compressedWeights === undefined || compressedBreaks === undefined) continue;
    console.log(`Qualifying compressed weights against f32 for ${reference.path}...`);
    const compressed = await new AlphaFoldMultimerGpu(device, {
      compactTransitions: true, recycleEarlyStopTolerance: -1,
    }).predict(reference.features, compressedWeights, compressedBreaks);
    for (let recycle = 0; recycle < compressed.recycles.length; recycle += 1) {
      within(`compressed recycle ${recycle} mean pLDDT`, compressed.recycles[recycle]!.confidence.meanPlddt,
        f32.recycles[recycle]!.confidence.meanPlddt, 0.25);
      within(`compressed recycle ${recycle} pTM`, compressed.recycles[recycle]!.confidence.ptm,
        f32.recycles[recycle]!.confidence.ptm, 0.005);
      within(`compressed recycle ${recycle} ipTM`, compressed.recycles[recycle]!.confidence.iptm,
        f32.recycles[recycle]!.confidence.iptm, 0.005);
    }
    metricWithin("compressed pLDDT", compressed.final.confidence.plddt,
      f32.final.confidence.plddt, "meanAbsoluteError", 0.5);
    metricWithin("compressed PAE", compressed.final.confidence.predictedAlignedError,
      f32.final.confidence.predictedAlignedError, "meanAbsoluteError", 0.5);
    metricWithin("compressed atom37", compressed.final.structure.atom37,
      f32.final.structure.atom37, "rootMeanSquareError", 0.5);
  }
} finally {
  device.destroy();
}
console.log(`Qualified model_1_multimer_v3 f32${compressedWeights === undefined ? "" : " and compressed"} `
  + `on ${references.length} official references.`);
