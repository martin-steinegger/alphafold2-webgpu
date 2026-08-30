import { arch, platform, release } from "node:os";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu, type MonomerModelWeights, type MonomerRecycleFeatures } from "../src/model/monomer.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { monomerDeviceRequirements, planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";

Object.assign(globalThis, globals);
const INPUT_MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";
const MODEL_MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
interface ReferenceManifest { readonly referencePrediction: { readonly recycleMetrics: readonly {
  readonly meanPlddt: number; readonly ptm: number;
}[] } }

const input = AlphaFoldFixture.fromStore(await FileTensorStore.open(INPUT_MANIFEST));
const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(MODEL_MANIFEST));
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
const weights: MonomerModelWeights = {
  embedding, template, extraStack, mainStack, structure,
  lddt: confidence.lddt, pae: confidence.pae, geometry,
};
const paeBreaks = await model.tensor("confidencePaeBreaks");
const reference = (input.store.manifest as unknown as ReferenceManifest).referencePrediction.recycleMetrics;
const shape = {
  length: features[0]!.aatype.length,
  msaSequences: features[0]!.msaSequences,
  extraSequences: features[0]!.extraSequences,
  recycles: features.length,
};
const gpu = create([]);

async function run(mode: "auto" | "compact") {
  const adapter = await gpu.requestAdapter();
  if (adapter === null) throw new Error("no WebGPU adapter is available");
  const automatic = planMonomerDevice(adapter, shape.length, shape.msaSequences, shape.extraSequences);
  const requirements = mode === "compact"
    ? monomerDeviceRequirements(shape.length, shape.msaSequences, shape.extraSequences)
    : automatic.requirements;
  const device = await requestAlphaFoldDevice(adapter, requirements);
  try {
    const prediction = await new AlphaFoldMonomerGpu(device, {
      compactTransitions: mode === "compact" || automatic.transitionMode === "chunked",
    }).predict(features, weights, paeBreaks);
    const recycles = prediction.recycles.map((result, recycle) => {
      const expected = reference[recycle]!;
      const plddtError = Math.abs(result.confidence.meanPlddt - expected.meanPlddt);
      const ptmError = Math.abs(result.confidence.ptm - expected.ptm);
      if (plddtError >= 0.05 || ptmError >= 0.001) {
        throw new Error(`${mode} recycle ${recycle} differs from the official reference: `
          + `pLDDT error ${plddtError}, pTM error ${ptmError}`);
      }
      return {
        recycle, elapsedMilliseconds: result.elapsedMilliseconds,
        meanPlddt: result.confidence.meanPlddt, ptm: result.confidence.ptm,
        plddtError, ptmError,
      };
    });
    return {
      mode, transitionMode: mode === "compact" ? "chunked" : automatic.transitionMode,
      elapsedMilliseconds: prediction.elapsedMilliseconds,
      memory: prediction.memory,
      recycles,
      hardware: {
        adapter: {
          vendor: adapter.info.vendor,
          architecture: adapter.info.architecture,
          device: adapter.info.device,
          description: adapter.info.description,
        },
        features: [...adapter.features].sort(),
        limits: {
          maxBufferSize: adapter.limits.maxBufferSize,
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        },
      },
    };
  } finally { device.destroy(); }
}

const automatic = await run("auto");
const compact = await run("compact");
for (let recycle = 0; recycle < automatic.recycles.length; recycle += 1) {
  const left = automatic.recycles[recycle]!; const right = compact.recycles[recycle]!;
  if (Math.abs(left.meanPlddt - right.meanPlddt) >= 1e-5 || Math.abs(left.ptm - right.ptm) >= 1e-5) {
    throw new Error(`compact-path regression at recycle ${recycle}`);
  }
}
const maximumMilliseconds = Number(process.env.AFWEBGPU_MAX_MS ?? "0");
if (maximumMilliseconds > 0 && automatic.elapsedMilliseconds > maximumMilliseconds) {
  throw new Error(`automatic path took ${automatic.elapsedMilliseconds.toFixed(1)} ms; `
    + `the configured limit is ${maximumMilliseconds.toFixed(1)} ms`);
}
console.log(JSON.stringify({
  schemaVersion: 1,
  passed: true,
  timestamp: new Date().toISOString(),
  host: { platform: platform(), release: release(), arch: arch(), node: process.version },
  adapter: automatic.hardware.adapter,
  features: automatic.hardware.features,
  limits: automatic.hardware.limits,
  shape,
  runs: [
    { ...automatic, hardware: undefined },
    { ...compact, hardware: undefined },
  ],
}, null, 2));
