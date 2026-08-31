import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu, type MonomerRecycleFeatures } from "../src/model/monomer.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

Object.assign(globalThis, globals);
const input = AlphaFoldFixture.fromStore(await FileTensorStore.open(
  "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json",
));
const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(
  "test/fixtures/evoformer/model1-query-59-stack/manifest.json",
));
const features: MonomerRecycleFeatures[] = [];
for (let recycle = 0; recycle < 4; recycle += 1) {
  const msa = `feature_msa_feat_recycle${recycle}`;
  const extra = `feature_extra_msa_recycle${recycle}`;
  features.push({
    targetFeatures: await input.tensor(`feature_target_feat_recycle${recycle}`),
    msaFeatures: await input.tensor(msa),
    msaMask: await input.tensor(`feature_msa_mask_recycle${recycle}`),
    extraMsa: await input.tensor(extra),
    extraHasDeletion: await input.tensor(`feature_extra_has_deletion_recycle${recycle}`),
    extraDeletionValue: await input.tensor(`feature_extra_deletion_value_recycle${recycle}`),
    extraMsaMask: await input.tensor(`feature_extra_msa_mask_recycle${recycle}`),
    residueIndex: await input.tensor(`feature_residue_index_recycle${recycle}`),
    aatype: await input.tensor(`feature_aatype_recycle${recycle}`),
    seqMask: await input.tensor(`feature_seq_mask_recycle${recycle}`),
    atom37ToAtom14: await input.tensor(`feature_residx_atom37_to_atom14_recycle${recycle}`),
    atom37Mask: await input.tensor(`feature_atom37_atom_exists_recycle${recycle}`),
    msaSequences: input.shape(msa)[0]!, extraSequences: input.shape(extra)[0]!,
    targetChannels: 22, msaFeatureChannels: 49,
  });
}
const [embedding, template, extraStack, mainStack, structure, confidence, geometry] = await Promise.all([
  model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(), model.mainStackWeights(),
  model.structureWeights(), model.confidenceWeights(), model.geometryTables(),
]);
const gpu = create([]);
const adapter = await gpu.requestAdapter();
if (adapter === null) throw new Error("no WebGPU adapter");
const device = await requestAlphaFoldDevice(adapter);
// Tally live GPU storage by label so the peak can be attributed.
const tally = new Map<string, { bytes: number; count: number }>();
const originalCreateBuffer = device.createBuffer.bind(device);
(device as { createBuffer: GPUDevice["createBuffer"] }).createBuffer = (descriptor: GPUBufferDescriptor) => {
  const label = (descriptor.label ?? "unlabelled").replace(/-?\d+$/, "").replace(/^(monomer|block|extra|main)\./, "");
  const entry = tally.get(label) ?? { bytes: 0, count: 0 };
  entry.bytes += descriptor.size; entry.count += 1;
  tally.set(label, entry);
  return originalCreateBuffer(descriptor);
};
const compactTransitions = process.env.AFWEBGPU_COMPACT === "1";
try {
  const prediction = await new AlphaFoldMonomerGpu(device, { compactTransitions }).predict(features, {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, await model.tensor("confidencePaeBreaks"));
  console.log(JSON.stringify({
    adapter: adapter.info,
    transitionMode: compactTransitions ? "chunked" : "full",
    shape: { length: 59, msaSequences: 508, extraSequences: 1_024, recycles: 4 },
    elapsedMilliseconds: prediction.elapsedMilliseconds,
    memory: prediction.memory,
    recycles: prediction.recycles.map((result, recycle) => ({
      recycle,
      elapsedMilliseconds: result.elapsedMilliseconds,
      meanPlddt: result.confidence.meanPlddt,
      ptm: result.confidence.ptm,
    })),
  }, null, 2));
  const ranked = [...tally].sort((a, b) => b[1].bytes - a[1].bytes).slice(0, 18);
  const total = [...tally.values()].reduce((sum, entry) => sum + entry.bytes, 0);
  console.error(`\nGPU buffers created, by label (total ${(total / 1024 ** 2).toFixed(0)} MiB):`);
  for (const [label, entry] of ranked) {
    console.error(`  ${(entry.bytes / 1024 ** 2).toFixed(1).padStart(8)} MiB  x${String(entry.count).padStart(3)}  ${label}`);
  }
} finally {
  device.destroy();
}
