/** Reports per-dispatch GPU timestamps for one extra-MSA and one main Evoformer block. */
import { CLUSTERED_MSA_CHANNELS, compactClusteredMsaFeatures } from "../src/input/msa-features.js";
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
const recycles = Number(process.env.AFWEBGPU_RECYCLES ?? "2");
for (let recycle = 0; recycle < recycles; recycle += 1) {
  const msa = `feature_msa_feat_recycle${recycle}`;
  const extra = `feature_extra_msa_recycle${recycle}`;
  features.push({
    targetFeatures: await input.tensor(`feature_target_feat_recycle${recycle}`),
    msaFeatures: compactClusteredMsaFeatures(
      await input.tensor(msa), input.shape(msa)[0]! * input.shape(msa)[1]!),
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
    targetChannels: 22, msaFeatureChannels: CLUSTERED_MSA_CHANNELS,
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
try {
  const prediction = await new AlphaFoldMonomerGpu(device, {
    profile: true, profileRecycle: recycles - 1,
  }).predict(features, {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, await model.tensor("confidencePaeBreaks"));
  const profile = prediction.final.gpuProfile!;
  const counts = { extra: extraStack.length, main: mainStack.length };
  for (const [name, block] of [["extra-MSA", profile.extraMsa], ["main-Evoformer", profile.mainEvoformer]] as const) {
    const gpuMilliseconds = block.entries.reduce((sum, entry) => sum + entry.nanoseconds, 0) / 1e6;
    const blocks = name === "extra-MSA" ? counts.extra : counts.main;
    console.log(`\n== ${name} block ${block.block} (${block.method}) ==`);
    console.log(`dispatches=${block.entries.length} gpu=${gpuMilliseconds.toFixed(3)}ms `
      + `wall=${block.wallMilliseconds.toFixed(3)}ms overhead=${(block.wallMilliseconds - gpuMilliseconds).toFixed(3)}ms`);
    console.log(`stack projection: ${blocks} blocks -> gpu ${(gpuMilliseconds * blocks).toFixed(1)}ms `
      + `wall ${(block.wallMilliseconds * blocks).toFixed(1)}ms`);
    const byLabel = new Map<string, { total: number; count: number }>();
    for (const entry of block.entries) {
      const key = entry.label.replace(/[.-]?\d+$/, "");
      const current = byLabel.get(key) ?? { total: 0, count: 0 };
      byLabel.set(key, { total: current.total + entry.nanoseconds, count: current.count + 1 });
    }
    for (const [label, value] of [...byLabel].sort((a, b) => b[1].total - a[1].total)) {
      console.log(`  ${(value.total / 1e6).toFixed(3)}ms  x${value.count}  ${label}`);
    }
  }
  console.log(`\ntotal=${prediction.elapsedMilliseconds.toFixed(0)}ms `
    + `submissions=${JSON.stringify(prediction.final.trunkSubmissions)}`);
} finally { device.destroy(); }
