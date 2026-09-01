import {
  AlphaFoldMonomerGpu,
  summarizeMonomerRecycle,
  type MonomerGpuOptions,
  type MonomerPrediction,
  type MonomerRecycleDetails,
  type MonomerRecycleResult,
  type MonomerRecycleSummary,
  type MultimerCompatibleModelWeights,
  type PredictionConfidenceResult,
} from "./monomer.js";
import {
  iterateMultimerQueryOnlyFeatures,
  type MultimerFeatureOptions,
  type MultimerRecycleFeatures,
} from "../input/multimer-features.js";
import type { QueryOnlyFeatureTables } from "../input/query-only-features.js";
import type { RecycleFeatureSource } from "../input/a3m-features.js";
import {
  multimerRankingConfidence, predictedInterfaceTmScoreFromExpected,
} from "../heads/confidence.js";

export type MultimerModelWeights = MultimerCompatibleModelWeights;

export interface MultimerConfidenceResult extends PredictionConfidenceResult {
  readonly iptm: number;
  readonly rankingConfidence: number;
}

export interface MultimerRecycleResult extends Omit<MonomerRecycleResult, "confidence"> {
  readonly confidence: MultimerConfidenceResult;
}

export interface MultimerRecycleSummary extends Omit<MonomerRecycleSummary, "confidence"> {
  readonly confidence: Pick<MultimerConfidenceResult, "meanPlddt" | "ptm" | "iptm" | "rankingConfidence">;
}

export interface MultimerPrediction extends Omit<MonomerPrediction, "recycles" | "final"> {
  readonly recycles: readonly MultimerRecycleSummary[];
  readonly final: MultimerRecycleResult;
}

export type MultimerRecycleCallback = (result: MultimerRecycleSummary, recycle: number) => void;

function summarizeMultimerDetails(
  result: MonomerRecycleDetails,
  asymId: Float32Array,
): MultimerRecycleSummary {
  const base = summarizeMonomerRecycle(result);
  const iptm = predictedInterfaceTmScoreFromExpected(
    result.confidence.tmScoreTerms, asymId.length, asymId,
  );
  return {
    ...base,
    confidence: {
      ...base.confidence,
      iptm,
      rankingConfidence: multimerRankingConfidence(result.confidence.ptm, iptm),
    },
  };
}

/**
 * AlphaFold-Multimer-v3 inference with chain-aware embeddings, native multimer
 * structure-module semantics, ipTM, and official multimer ranking confidence.
 * Query-only and paired/unpaired MSA feature builders both use this shared
 * allocator, compact-transition, scratch-pool, and recycling implementation.
 */
export class AlphaFoldMultimerGpu {
  readonly device: GPUDevice;
  readonly #model: AlphaFoldMonomerGpu;

  constructor(device: GPUDevice, options: MonomerGpuOptions = {}) {
    this.device = device;
    this.#model = new AlphaFoldMonomerGpu(device, {
      ...options, multimer: true,
      recycleEarlyStopTolerance: options.recycleEarlyStopTolerance ?? 0.5,
    });
  }

  async predictChains(
    chains: string | readonly string[],
    weights: MultimerModelWeights,
    featureTables: QueryOnlyFeatureTables,
    options: MultimerFeatureOptions = {},
    paeBreaks?: Float32Array,
    onRecycle?: MultimerRecycleCallback,
  ): Promise<MultimerPrediction> {
    return this.predict(iterateMultimerQueryOnlyFeatures(chains, featureTables, options),
      weights, paeBreaks, onRecycle);
  }

  async predict(
    featuresByRecycle: RecycleFeatureSource<MultimerRecycleFeatures>,
    weights: MultimerModelWeights,
    paeBreaks: Float32Array = Float32Array.from({ length: 63 }, (_, index) => index * 0.5),
    onRecycle?: MultimerRecycleCallback,
  ): Promise<MultimerPrediction> {
    if (featuresByRecycle.length === 0) throw new RangeError("at least one multimer feature set is required");
    const mappedDuringRun: MultimerRecycleSummary[] = [];
    const prediction = await this.#model.predict(featuresByRecycle, weights, paeBreaks, undefined,
      (result, recycle, features) => {
      const asymId = features.chainRelative!.asymId;
      const mapped = summarizeMultimerDetails(result, asymId);
      mappedDuringRun.push(mapped);
      onRecycle?.(mapped, recycle);
    });
    if (mappedDuringRun.length !== prediction.recycles.length) {
      throw new Error("multimer prediction did not report every recycle");
    }
    const finalSummary = mappedDuringRun.at(-1);
    if (finalSummary === undefined) throw new Error("multimer prediction has no confidence summary");
    return {
      ...prediction,
      recycles: mappedDuringRun,
      final: {
        ...prediction.final,
        confidence: {
          ...prediction.final.confidence,
          iptm: finalSummary.confidence.iptm,
          rankingConfidence: finalSummary.confidence.rankingConfidence,
        },
      },
    };
  }
}
