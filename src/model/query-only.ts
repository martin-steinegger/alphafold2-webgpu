import { ConfidenceHeadsGpu, type ConfidenceResult, type PredictedAlignedErrorWeights, type PredictedLddtWeights } from "../heads/confidence.js";
import { InputEmbedderGpu, type InputEmbedderWeights } from "../evoformer/input-embedder.js";
import { EvoformerStackGpu, ExtraMsaPairStackGpu } from "../evoformer/stack.js";
import type { EvoformerBlockWeights, EvoformerPairBlockWeights } from "../evoformer/block.js";
import { QueryOnlyTemplateGpu, type QueryOnlyTemplateWeights } from "../evoformer/template.js";
import { ElementwiseAddGpu } from "../runtime/elementwise.js";
import { StructureModuleGpu, type StructureModuleResult, type StructureModuleWeights } from "../structure/module.js";
import type { ResidueGeometryTables } from "../structure/geometry.js";
import { makeQueryOnlyFeatures, type QueryOnlyFeatureOptions, type QueryOnlyFeatureTables } from "../input/query-only-features.js";

export interface QueryOnlyRecycleFeatures {
  readonly targetFeatures: Float32Array;
  readonly msaFeatures: Float32Array;
  readonly msaMask: Float32Array;
  readonly extraMsa: Float32Array;
  readonly extraHasDeletion: Float32Array;
  readonly extraDeletionValue: Float32Array;
  readonly extraMsaMask: Float32Array;
  readonly residueIndex: Float32Array;
  readonly aatype: Float32Array;
  readonly seqMask: Float32Array;
  readonly atom37ToAtom14: Float32Array;
  readonly atom37Mask: Float32Array;
  readonly targetChannels: number;
  readonly msaFeatureChannels: number;
  readonly extraSequences: number;
}

export interface QueryOnlyModelWeights {
  readonly embedding: InputEmbedderWeights;
  readonly template: QueryOnlyTemplateWeights;
  readonly extraStack: readonly EvoformerPairBlockWeights[];
  readonly mainStack: readonly EvoformerBlockWeights[];
  readonly structure: StructureModuleWeights;
  readonly lddt: PredictedLddtWeights;
  readonly pae: PredictedAlignedErrorWeights;
  readonly geometry: ResidueGeometryTables;
}

export interface QueryOnlyRecycleResult {
  readonly msaFirstRow: Float32Array;
  readonly pair: Float32Array;
  readonly structure: StructureModuleResult;
  readonly confidence: ConfidenceResult;
  readonly elapsedMilliseconds: number;
}

export interface QueryOnlyPrediction {
  readonly recycles: readonly QueryOnlyRecycleResult[];
  readonly final: QueryOnlyRecycleResult;
  readonly elapsedMilliseconds: number;
}

/** Model-1 query-only path, including templates, 54 pair blocks, structure, confidence, and recycling. */
export class AlphaFoldQueryOnlyGpu {
  readonly device: GPUDevice;
  constructor(device: GPUDevice) { this.device = device; }

  async predictSequence(
    sequence: string,
    weights: QueryOnlyModelWeights,
    featureTables: QueryOnlyFeatureTables,
    options: QueryOnlyFeatureOptions = {},
    paeBreaks?: Float32Array,
  ): Promise<QueryOnlyPrediction> {
    return this.predict(makeQueryOnlyFeatures(sequence, featureTables, options), weights, paeBreaks);
  }

  async predict(
    recycleFeatures: readonly QueryOnlyRecycleFeatures[],
    weights: QueryOnlyModelWeights,
    paeBreaks?: Float32Array,
  ): Promise<QueryOnlyPrediction> {
    if (recycleFeatures.length === 0) throw new RangeError("at least one recycle feature set is required");
    const length = recycleFeatures[0]!.aatype.length;
    const pairMask = new Float32Array(length * length);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      pairMask[i * length + j] = recycleFeatures[0]!.seqMask[i]! * recycleFeatures[0]!.seqMask[j]!;
    }
    const template = await new QueryOnlyTemplateGpu(this.device).run({
      length, templateChannels: 64, pairChannels: 128, pairMask, weights: weights.template,
    });
    let previousMsa: Float32Array = new Float32Array(length * 256);
    let previousPair: Float32Array = new Float32Array(length * length * 128);
    let previousPositions: Float32Array = new Float32Array(length * 37 * 3);
    const results: QueryOnlyRecycleResult[] = [];
    const start = performance.now();
    for (let recycle = 0; recycle < recycleFeatures.length; recycle += 1) {
      const recycleStart = performance.now();
      const features = recycleFeatures[recycle]!;
      if (features.aatype.length !== length) throw new RangeError("recycle lengths differ");
      const embedding = await new InputEmbedderGpu(this.device).run({
        targetFeatures: features.targetFeatures,
        msaFeatures: features.msaFeatures,
        extraMsa: features.extraMsa,
        extraHasDeletion: features.extraHasDeletion,
        extraDeletionValue: features.extraDeletionValue,
        residueIndex: features.residueIndex,
        aatype: features.aatype,
        previousMsaFirstRow: previousMsa,
        previousPair,
        previousPositions,
        length,
        msaSequences: 1,
        extraSequences: features.extraSequences,
        targetChannels: features.targetChannels,
        msaFeatureChannels: features.msaFeatureChannels,
        msaChannels: 256,
        pairChannels: 128,
        extraMsaChannels: 64,
        weights: weights.embedding,
      });
      const pairWithTemplate = await new ElementwiseAddGpu(this.device).run(
        embedding.pairWithoutTemplates, template.pairUpdate,
      );
      const extra = await new ExtraMsaPairStackGpu(this.device).run({
        msa: embedding.extraMsa,
        pair: pairWithTemplate,
        msaMask: features.extraMsaMask,
        pairMask,
        sequences: features.extraSequences,
        length,
        cM: 64,
        cZ: 128,
        cOuter: weights.extraStack[0]!.outerProductMean.leftBias.length,
        triangleHidden: weights.extraStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
        blockWeights: weights.extraStack,
      });
      const trunk = await new EvoformerStackGpu(this.device).run({
        msa: embedding.msa,
        pair: extra.pair,
        msaMask: features.msaMask,
        pairMask,
        sequences: 1,
        length,
        cM: 256,
        cZ: 128,
        cOuter: weights.mainStack[0]!.outerProductMean.leftBias.length,
        triangleHidden: weights.mainStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
        blockWeights: weights.mainStack,
      });
      const msaFirstRow = trunk.msa.subarray(0, length * 256).slice();
      const structure = await new StructureModuleGpu(this.device).run({
        msaFirstRow,
        pair: trunk.pair,
        mask: features.seqMask,
        aatype: features.aatype,
        atom37ToAtom14: features.atom37ToAtom14,
        atom37Mask: features.atom37Mask,
        length,
        weights: weights.structure,
        geometry: weights.geometry,
      });
      const confidence = await new ConfidenceHeadsGpu(this.device).run(
        structure.finalRepresentation, trunk.pair, length, weights.lddt, weights.pae, paeBreaks,
      );
      results.push({
        msaFirstRow, pair: trunk.pair, structure, confidence,
        elapsedMilliseconds: performance.now() - recycleStart,
      });
      previousMsa = msaFirstRow;
      previousPair = trunk.pair;
      previousPositions = structure.atom37;
    }
    return { recycles: results, final: results[results.length - 1]!, elapsedMilliseconds: performance.now() - start };
  }
}
