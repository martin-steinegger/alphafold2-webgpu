import { AlphaFoldMultimerGpu, type MultimerModelWeights } from "../src/model/multimer.js";
import type { MultimerRecycleFeatures } from "../src/input/multimer-features.js";
import { AlphaFoldFixture, type TensorStore } from "../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";
import { EvoformerBlockGpu } from "../src/evoformer/block.js";
import { EvoformerStackGpu, ExtraMsaStackGpu } from "../src/evoformer/stack.js";
import { InputEmbedderGpu } from "../src/evoformer/input-embedder.js";
import { StructureInitializeGpu } from "../src/structure/initialize.js";
import { MultimerMockTemplateGpu } from "../src/evoformer/multimer-template.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

interface ReferenceManifest {
  readonly recycles: number;
  readonly reference: { readonly recycleMetrics: readonly {
    readonly meanPlddt: number; readonly ptm: number; readonly iptm: number;
    readonly rankingConfidence: number;
  }[] };
}

export interface MultimerQualificationResult {
  readonly meanPlddt: number;
  readonly ptm: number;
  readonly iptm: number;
  readonly atom37MeanAbsoluteError: number;
  readonly plddtMeanAbsoluteError: number;
  readonly paeMeanAbsoluteError: number;
  readonly pairMeanAbsoluteError: number;
  readonly rankingConfidence: number;
  readonly plddt: readonly number[];
  readonly pae: readonly number[];
  readonly atom37: readonly number[];
}

function within(label: string, actual: number, expected: number, threshold: number): void {
  const difference = Math.abs(actual - expected);
  if (!(difference < threshold)) throw new Error(`${label}: difference ${difference}, limit ${threshold}`);
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

async function capturedFeatures(reference: TensorStore, model: AlphaFoldFixture,
  recycles: number): Promise<readonly MultimerRecycleFeatures[]> {
  const tables = await model.queryOnlyFeatureTables();
  const results: MultimerRecycleFeatures[] = [];
  for (let recycle = 0; recycle <= recycles; recycle += 1) {
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

export async function qualifyMultimer(referenceManifestUrl: string,
  modelManifestUrl: string, enforceOfficial = true): Promise<MultimerQualificationResult> {
  const reference = await HttpTensorStore.open(referenceManifestUrl);
  const referenceManifest = reference.manifest as unknown as ReferenceManifest;
  const modelStore = await HttpTensorStore.open(modelManifestUrl);
  const model = AlphaFoldFixture.fromStore(modelStore);
  const weights = await multimerWeights(model);
  const breaks = await model.tensor("confidencePaeBreaks");
  const features = await capturedFeatures(reference, model, referenceManifest.recycles);
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("no WebGPU adapter");
  const device = await requestAlphaFoldDevice(adapter);
  try {
    if (enforceOfficial && reference.manifest.tensors.mainBlock0InputMsa !== undefined) {
      const pairMask = new Float32Array(features[0]!.aatype.length ** 2).fill(1);
      const embedded = await new InputEmbedderGpu(device).run({
        ...features[0]!,
        previousMsaFirstRow: new Float32Array(features[0]!.aatype.length * 256),
        previousPair: new Float32Array(features[0]!.aatype.length ** 2 * 128),
        previousPositions: new Float32Array(features[0]!.aatype.length * 37 * 3),
        length: features[0]!.aatype.length, msaChannels: 256, pairChannels: 128,
        extraMsaChannels: 64, weights: weights.embedding,
      });
      const officialMainMsa = await reference.tensor("mainBlock0InputMsa");
      console.log("Multimer embedding errors", JSON.stringify({
        msa: errorMetrics(embedded.msa, officialMainMsa.subarray(0, embedded.msa.length)),
        pair: errorMetrics(embedded.pairWithoutTemplates, await reference.tensor("extraBlock0InputPair")),
        extraMsa: errorMetrics(embedded.extraMsa, await reference.tensor("extraBlock0InputMsa")),
      }));
      const mockTemplate = await new MultimerMockTemplateGpu(device).run(
        embedded.pairWithoutTemplates, pairMask, features[0]!.aatype.length, weights.multimerTemplate,
      );
      const officialPairWithTemplate = await reference.tensor("extraBlock0InputPair");
      const officialTemplateUpdate = Float32Array.from(
        officialPairWithTemplate, (value, index) => value - embedded.pairWithoutTemplates[index]!,
      );
      console.log("Multimer mock-template errors", JSON.stringify({
        pair: errorMetrics(mockTemplate.pairUpdate, officialTemplateUpdate),
        msa: errorMetrics(mockTemplate.msaRows, officialMainMsa.subarray(embedded.msa.length)),
      }));
      const extra = await new ExtraMsaStackGpu(device).run({
        msa: await reference.tensor("extraBlock0InputMsa"),
        pair: await reference.tensor("extraBlock0InputPair"),
        msaMask: await reference.tensor("extraBlock0MsaMask"),
        pairMask: await reference.tensor("extraBlock0PairMask"),
        sequences: reference.shape("extraBlock0InputMsa")[0]!, length: features[0]!.aatype.length,
        cM: 64, cZ: 128, cOuter: weights.extraStack[0]!.outerProductMean.leftBias.length,
        triangleHidden: weights.extraStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
        outerProductMeanFirst: true, blockWeights: [weights.extraStack[0]!],
      });
      console.log("Multimer extra block 0 errors", JSON.stringify({
        msa: errorMetrics(extra.msa, await reference.tensor("extraBlock0ExpectedMsa")),
        pair: errorMetrics(extra.pair, await reference.tensor("extraBlock0ExpectedPair")),
      }));
      const main = await new EvoformerBlockGpu(device).run({
        msa: await reference.tensor("mainBlock0InputMsa"),
        pair: await reference.tensor("mainBlock0InputPair"),
        msaMask: await reference.tensor("mainBlock0MsaMask"),
        pairMask: await reference.tensor("mainBlock0PairMask"),
        sequences: reference.shape("mainBlock0InputMsa")[0]!, length: features[0]!.aatype.length,
        cM: 256, cZ: 128, cOuter: weights.mainStack[0]!.outerProductMean.leftBias.length,
        triangleHidden: weights.mainStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
        outerProductMeanFirst: true, weights: weights.mainStack[0]!,
      });
      console.log("Multimer main block 0 errors", JSON.stringify({
        msa: errorMetrics(main.msa, await reference.tensor("mainBlock0ExpectedMsa")),
        pair: errorMetrics(main.pair, await reference.tensor("mainBlock0ExpectedPair")),
      }));
      const mainStack = await new EvoformerStackGpu(device).run({
        msa: await reference.tensor("mainBlock0InputMsa"),
        pair: await reference.tensor("mainBlock0InputPair"),
        msaMask: await reference.tensor("mainBlock0MsaMask"),
        pairMask: await reference.tensor("mainBlock0PairMask"),
        sequences: reference.shape("mainBlock0InputMsa")[0]!, length: features[0]!.aatype.length,
        cM: 256, cZ: 128, cOuter: weights.mainStack[0]!.outerProductMean.leftBias.length,
        triangleHidden: weights.mainStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
        outerProductMeanFirst: true, blockWeights: weights.mainStack,
      });
      console.log("Multimer full main stack pair error", JSON.stringify(
        errorMetrics(mainStack.pair, await reference.tensor("structureInputPair")),
      ));
      console.log("Multimer full main stack MSA first-row error", JSON.stringify(
        errorMetrics(mainStack.msa.subarray(0, features[0]!.aatype.length * 256),
          await reference.tensor("structureInputMsaFirstRow")),
      ));
    }
    const prediction = await new AlphaFoldMultimerGpu(device, {
      compactTransitions: true, recycleEarlyStopTolerance: -1,
    }).predict(features, weights, breaks);
    const initialized = await new StructureInitializeGpu(device).run(
      prediction.final.msaFirstRow, features[0]!.aatype.length, 256, 384, weights.structure.initialize,
    );
    console.log("Multimer structure initialization errors", JSON.stringify({
      activations: errorMetrics(initialized.activations, await reference.tensor("ipaActivations")),
      affine: errorMetrics(initialized.affine, await reference.tensor("ipaAffine")),
    }));
    for (let recycle = 0; recycle < prediction.recycles.length; recycle += 1) {
      const actual = prediction.recycles[recycle]!.confidence;
      const expected = referenceManifest.reference.recycleMetrics[recycle]!;
      console.log("Multimer confidence", { recycle, actual, expected });
    }
    const plddtMae = errorMetrics(prediction.final.confidence.plddt,
      await reference.tensor("predictionPlddt")).meanAbsoluteError;
    const paeMae = errorMetrics(prediction.final.confidence.predictedAlignedError,
      await reference.tensor("predictionPae")).meanAbsoluteError;
    const atomMae = errorMetrics(prediction.final.structure.atom37,
      await reference.tensor("predictionAtom37")).meanAbsoluteError;
    const pairMae = errorMetrics(prediction.final.pair,
      await reference.tensor("structureInputPair")).meanAbsoluteError;
    console.log("Multimer tensor errors", { pairMae, plddtMae, paeMae, atomMae });
    if (enforceOfficial) {
      for (let recycle = 0; recycle < prediction.recycles.length; recycle += 1) {
        const actual = prediction.recycles[recycle]!.confidence;
        const expected = referenceManifest.reference.recycleMetrics[recycle]!;
        within(`recycle ${recycle} mean pLDDT`, actual.meanPlddt, expected.meanPlddt, 0.05);
        within(`recycle ${recycle} pTM`, actual.ptm, expected.ptm, 0.001);
        within(`recycle ${recycle} ipTM`, actual.iptm, expected.iptm, 0.001);
        within(`recycle ${recycle} ranking`, actual.rankingConfidence, expected.rankingConfidence, 0.001);
      }
      if (!(plddtMae < 0.5)) throw new Error(`pLDDT MAE ${plddtMae}, limit 0.5`);
      if (!(paeMae < 0.5)) throw new Error(`PAE MAE ${paeMae}, limit 0.5`);
      if (!(atomMae < 0.25)) throw new Error(`atom37 MAE ${atomMae}, limit 0.25`);
    }
    const confidence = prediction.final.confidence;
    return { meanPlddt: confidence.meanPlddt, ptm: confidence.ptm, iptm: confidence.iptm,
      atom37MeanAbsoluteError: atomMae, plddtMeanAbsoluteError: plddtMae, paeMeanAbsoluteError: paeMae,
      pairMeanAbsoluteError: pairMae, rankingConfidence: confidence.rankingConfidence,
      plddt: Array.from(confidence.plddt), pae: Array.from(confidence.predictedAlignedError),
      atom37: Array.from(prediction.final.structure.atom37) };
  } finally {
    device.destroy();
  }
}
