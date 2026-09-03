/**
 * The whole prediction under a binding limit far below its tensors.
 *
 * A device may bind only a bounded slice of a buffer, and the adapters here
 * allow gigabytes, so every kernel that walks a whole pair or MSA passes at
 * these lengths and would fail on a device with the 128 MiB default at 1500
 * residues. Lowering the limit to a few mebibytes at 59 residues puts the
 * short prediction on the same path, and the answer must not move: windows
 * and shards change where a kernel reads, never what it computes.
 */
import { CLUSTERED_MSA_CHANNELS, compactClusteredMsaFeatures } from "../src/input/msa-features.js";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu, type MonomerRecycleFeatures } from "../src/model/monomer.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const A3M_MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";
const WEIGHT_MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
/**
 * Under the MSA activations, which are 15 MiB packed at this shape, and low
 * enough to split them eight ways while the windows still fit the stage's
 * storage slots. The pair here is under a mebibyte, so its own windowing is
 * covered by the block-level test rather than this one.
 */
const BUDGET = 2 * 1024 * 1024;

describe.skipIf(!enabled)("prediction under a small binding limit", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await requestAlphaFoldDevice(adapter);
  });
  afterAll(() => device?.destroy());

  it("gives the same answer as the same prediction with whole bindings", async () => {
    const input = AlphaFoldFixture.fromStore(await FileTensorStore.open(A3M_MANIFEST));
    const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(WEIGHT_MANIFEST));
    const features: MonomerRecycleFeatures[] = [];
    for (let recycle = 0; recycle < 2; recycle += 1) {
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
    const weights = {
      embedding, template, extraStack, mainStack, structure,
      lddt: confidence.lddt, pae: confidence.pae, geometry,
    };
    const breaks = await model.tensor("confidencePaeBreaks");
    const whole = await new AlphaFoldMonomerGpu(device).predict(features, weights, breaks);
    const windowed = new AlphaFoldMonomerGpu(device, { bindingBudgetBytes: BUDGET });
    const sharded = await windowed.predict(features, weights, breaks);
    expect([...windowed.oversizedBindings].map(([label, bytes]) =>
      `${label} ${(bytes / 1024 ** 2).toFixed(2)} MiB`)).toEqual([]);
    expect(sharded.final.confidence.meanPlddt).toBeCloseTo(whole.final.confidence.meanPlddt, 6);
    expect(sharded.final.confidence.ptm).toBeCloseTo(whole.final.confidence.ptm, 6);
    const errors = sharded.final.confidence.predictedAlignedError;
    let largest = 0;
    for (let index = 0; index < errors.length; index += 1) {
      largest = Math.max(largest, Math.abs(errors[index]! - whole.final.confidence.predictedAlignedError[index]!));
    }
    expect(largest).toBe(0);
  }, 600_000);
});
