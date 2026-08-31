import {
  AlphaFoldMonomerGpu,
  type MonomerGpuOptions,
  type MonomerPrediction,
  type MonomerRecycleResult,
  type MultimerCompatibleModelWeights,
} from "./monomer.js";
import {
  makeMultimerQueryOnlyFeatures,
  type MultimerFeatureOptions,
  type MultimerRecycleFeatures,
} from "../input/multimer-features.js";
import type { QueryOnlyFeatureTables } from "../input/query-only-features.js";
import {
  multimerRankingConfidence,
  predictedInterfaceTmScore,
  type ConfidenceResult,
} from "../heads/confidence.js";

export type MultimerModelWeights = MultimerCompatibleModelWeights;

export interface MultimerConfidenceResult extends ConfidenceResult {
  readonly iptm: number;
  readonly rankingConfidence: number;
}

export interface MultimerRecycleResult extends Omit<MonomerRecycleResult, "confidence"> {
  readonly confidence: MultimerConfidenceResult;
}

export interface MultimerPrediction extends Omit<MonomerPrediction, "recycles" | "final"> {
  readonly recycles: readonly MultimerRecycleResult[];
  readonly final: MultimerRecycleResult;
}

export type MultimerRecycleCallback = (result: MultimerRecycleResult, recycle: number) => void;

function withInterfaceConfidence(
  result: MonomerRecycleResult,
  asymId: Float32Array,
  breaks: Float32Array,
): MultimerRecycleResult {
  const iptm = predictedInterfaceTmScore(result.confidence.paeLogits, asymId.length, breaks, asymId);
  const confidence: MultimerConfidenceResult = {
    ...result.confidence,
    iptm,
    rankingConfidence: multimerRankingConfidence(result.confidence.ptm, iptm),
  };
  return { ...result, confidence };
}

/**
 * AlphaFold-Multimer-v3 inference with chain-aware embeddings, native multimer
 * structure-module semantics, ipTM, and official multimer ranking confidence.
 * The current public feature builder is the official no-template/no-MSA path.
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
    return this.predict(makeMultimerQueryOnlyFeatures(chains, featureTables, options), weights, paeBreaks, onRecycle);
  }

  async predict(
    featuresByRecycle: readonly MultimerRecycleFeatures[],
    weights: MultimerModelWeights,
    paeBreaks: Float32Array = Float32Array.from({ length: 63 }, (_, index) => index * 0.5),
    onRecycle?: MultimerRecycleCallback,
  ): Promise<MultimerPrediction> {
    if (featuresByRecycle.length === 0) throw new RangeError("at least one multimer feature set is required");
    const asymId = featuresByRecycle[0]!.chainRelative.asymId;
    const mappedDuringRun: MultimerRecycleResult[] = [];
    const prediction = await this.#model.predict(featuresByRecycle, weights, paeBreaks, (result, recycle) => {
      const mapped = withInterfaceConfidence(result, asymId, paeBreaks);
      mappedDuringRun.push(mapped);
      onRecycle?.(mapped, recycle);
    });
    const recycles = mappedDuringRun.length === prediction.recycles.length
      ? mappedDuringRun
      : prediction.recycles.map((result) => withInterfaceConfidence(result, asymId, paeBreaks));
    return { ...prediction, recycles, final: recycles[recycles.length - 1]! };
  }
}
