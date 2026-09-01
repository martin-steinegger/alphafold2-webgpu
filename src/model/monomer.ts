import {
  ConfidenceHeadsGpu,
  type ConfidenceSummaryResult,
  type PredictedAlignedErrorWeights,
  type PredictedLddtWeights,
  type ReducedConfidenceResult,
} from "../heads/confidence.js";
import { encodeInputEmbedder, type InputEmbedderWeights } from "../evoformer/input-embedder.js";
import {
  encodeEvoformerBlock, encodeExtraMsaBlock, type EvoformerBlockWeights, type ExtraMsaBlockWeights,
} from "../evoformer/block.js";
import { QueryOnlyTemplateGpu, type QueryOnlyTemplateWeights } from "../evoformer/template.js";
import { MultimerMockTemplateGpu, type MultimerMockTemplateWeights } from "../evoformer/multimer-template.js";
import { WebGpuExecution, type GpuTensor, type GpuTimestampEntry } from "../runtime/execution.js";
import { StructureModuleGpu, type StructureModuleResult, type StructureModuleWeights } from "../structure/module.js";
import type { ResidueGeometryTables } from "../structure/geometry.js";
import {
  iterateA3mFeatures, type A3mFeatureOptions, type RecycleFeatureSource,
} from "../input/a3m-features.js";
import type { QueryOnlyFeatureTables } from "../input/query-only-features.js";
import { TRANSITION_CHUNK_TARGET_BYTES } from "../evoformer/transition.js";
import { COMPACT_GPU_POOL_BYTES, type AllocationSnapshot } from "../runtime/allocator.js";
import { multimerRecycleDistanceRms } from "./multimer-recycling.js";

export interface MonomerRecycleFeatures {
  readonly targetFeatures: Float32Array; readonly msaFeatures: Float32Array; readonly msaMask: Float32Array;
  readonly extraMsa: Float32Array; readonly extraHasDeletion: Float32Array; readonly extraDeletionValue: Float32Array;
  readonly extraMsaMask: Float32Array; readonly residueIndex: Float32Array; readonly aatype: Float32Array;
  readonly seqMask: Float32Array; readonly atom37ToAtom14: Float32Array; readonly atom37Mask: Float32Array;
  readonly msaSequences: number; readonly extraSequences: number;
  readonly targetChannels: number; readonly msaFeatureChannels: number;
  readonly chainRelative?: {
    readonly asymId: Float32Array; readonly entityId: Float32Array; readonly symId: Float32Array;
  };
}

export interface MonomerModelWeights {
  readonly embedding: InputEmbedderWeights; readonly template: QueryOnlyTemplateWeights;
  readonly extraStack: readonly ExtraMsaBlockWeights[]; readonly mainStack: readonly EvoformerBlockWeights[];
  readonly structure: StructureModuleWeights; readonly lddt: PredictedLddtWeights;
  readonly pae: PredictedAlignedErrorWeights; readonly geometry: ResidueGeometryTables;
}

export type MultimerCompatibleModelWeights = Omit<MonomerModelWeights, "template"> & {
  /** ColabFold's deterministic mock-template path when template search is disabled. */
  readonly multimerTemplate: MultimerMockTemplateWeights;
};

/** User-facing confidence tensors; raw categorical logits are transient implementation details. */
export type PredictionConfidenceResult = ConfidenceSummaryResult;

export interface MonomerRecycleResult {
  readonly msaFirstRow: Float32Array; readonly pair: Float32Array;
  readonly structure: StructureModuleResult; readonly confidence: PredictionConfidenceResult;
  readonly elapsedMilliseconds: number;
  readonly trunkSubmissions: MonomerTrunkSubmissionCounts;
  readonly gpuProfile?: MonomerRecycleGpuProfile;
}

/** Full per-recycle data used transiently before the final pair readback. */
export type MonomerRecycleDetails = Omit<MonomerRecycleResult, "pair" | "confidence"> & {
  readonly confidence: ReducedConfidenceResult;
};

type MonomerFinalDetails = Omit<MonomerRecycleResult, "pair">;

export interface MonomerRecycleSummary {
  readonly confidence: Pick<ConfidenceSummaryResult, "meanPlddt" | "ptm">;
  readonly elapsedMilliseconds: number;
  readonly trunkSubmissions: MonomerTrunkSubmissionCounts;
  readonly gpuProfile?: MonomerRecycleGpuProfile;
}

export interface MonomerPrediction {
  /** Lightweight metrics for every recycle; pair and structure arrays are retained only in final. */
  readonly recycles: readonly MonomerRecycleSummary[]; readonly final: MonomerRecycleResult;
  readonly elapsedMilliseconds: number;
  readonly memory: MonomerMemorySnapshot;
}

export interface MonomerMemorySnapshot extends AllocationSnapshot {
  /** Main trunk allocator peak (same value as peakResidentBytes). */
  readonly mainPeakResidentBytes: number;
  /** Largest structure-core allocator peak across recycles. */
  readonly structureCorePeakResidentBytes: number;
  /** Largest confidence-head allocator peak across recycles. */
  readonly confidencePeakResidentBytes: number;
  /** Conservative simultaneous peak: live trunk residency plus the active auxiliary stage. */
  readonly combinedPeakResidentBytes: number;
}

export interface MonomerTrunkSubmissionCounts {
  readonly embedding: number; readonly template: number;
  readonly extraMsa: number; readonly mainEvoformer: number;
  readonly readback: number; readonly total: number;
}

export interface MonomerGpuOptions {
  /** Profile one extra-MSA and one main Evoformer block in a selected recycle. */
  readonly profile?: boolean;
  readonly profileRecycle?: number;
  readonly profileExtraMsaBlock?: number;
  readonly profileMainEvoformerBlock?: number;
  /** Bounds transition scratch even when the device exposes larger binding limits. */
  readonly compactTransitions?: boolean;
  /** Caps reusable scratch retained between blocks; compact mode uses the bounded shared default. */
  readonly maxPooledBytes?: number;
  /** Internal model architecture selector used by AlphaFoldMultimerGpu. */
  readonly multimer?: boolean;
  /** Multimer CA-distance RMS threshold; negative disables early stopping. */
  readonly recycleEarlyStopTolerance?: number;
}

export interface MonomerBlockGpuProfile {
  readonly block: number;
  readonly method: "timestamp-query" | "wall-clock";
  readonly wallMilliseconds: number;
  readonly entries: readonly GpuTimestampEntry[];
}

export interface MonomerRecycleGpuProfile {
  readonly extraMsa: MonomerBlockGpuProfile;
  readonly mainEvoformer: MonomerBlockGpuProfile;
}

export type MonomerRecycleCallback = (
  result: MonomerRecycleSummary, recycle: number,
) => void;

type MonomerRecycleDetailsCallback = (
  result: MonomerRecycleDetails, recycle: number, features: MonomerRecycleFeatures,
) => void;

export function summarizeMonomerRecycle(result: MonomerRecycleDetails): MonomerRecycleSummary {
  return {
    confidence: { meanPlddt: result.confidence.meanPlddt, ptm: result.confidence.ptm },
    elapsedMilliseconds: result.elapsedMilliseconds,
    trunkSubmissions: result.trunkSubmissions,
    ...(result.gpuProfile === undefined ? {} : { gpuProfile: result.gpuProfile }),
  };
}

/** Full monomer model for clustered MSA/A3M inputs, with all learned operations dispatched through WebGPU. */
export class AlphaFoldMonomerGpu {
  readonly device: GPUDevice;
  readonly profile: boolean;
  readonly profileRecycle: number;
  readonly profileExtraMsaBlock: number;
  readonly profileMainEvoformerBlock: number;
  readonly compactTransitions: boolean;
  readonly maxPooledBytes: number | undefined;
  readonly multimer: boolean;
  readonly recycleEarlyStopTolerance: number;
  constructor(device: GPUDevice, options: MonomerGpuOptions = {}) {
    this.device = device;
    this.profile = options.profile ?? false;
    this.profileRecycle = options.profileRecycle ?? 0;
    this.profileExtraMsaBlock = options.profileExtraMsaBlock ?? 0;
    this.profileMainEvoformerBlock = options.profileMainEvoformerBlock ?? 0;
    this.compactTransitions = options.compactTransitions ?? false;
    this.maxPooledBytes = options.maxPooledBytes
      ?? (this.compactTransitions ? COMPACT_GPU_POOL_BYTES : undefined);
    this.multimer = options.multimer ?? false;
    this.recycleEarlyStopTolerance = options.recycleEarlyStopTolerance ?? -1;
    if (!Number.isFinite(this.recycleEarlyStopTolerance)) {
      throw new RangeError("recycleEarlyStopTolerance must be finite");
    }
    for (const [name, value] of [
      ["profileRecycle", this.profileRecycle],
      ["profileExtraMsaBlock", this.profileExtraMsaBlock],
      ["profileMainEvoformerBlock", this.profileMainEvoformerBlock],
    ] as const) {
      if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${name} must be a non-negative safe integer`);
    }
  }
  async predictA3m(a3mText: string, weights: MonomerModelWeights, featureTables: QueryOnlyFeatureTables,
    options: A3mFeatureOptions = {}, paeBreaks?: Float32Array,
    onRecycle?: MonomerRecycleCallback): Promise<MonomerPrediction> {
    return this.predict(iterateA3mFeatures(a3mText, featureTables, options), weights, paeBreaks, onRecycle);
  }
  async predict(featuresByRecycle: RecycleFeatureSource<MonomerRecycleFeatures>,
    weights: MonomerModelWeights | MultimerCompatibleModelWeights,
    paeBreaks?: Float32Array, onRecycle?: MonomerRecycleCallback,
    onRecycleDetails?: MonomerRecycleDetailsCallback): Promise<MonomerPrediction> {
    if (featuresByRecycle.length === 0) throw new RangeError("at least one feature set is required");
    const featureIterator = featuresByRecycle[Symbol.iterator]();
    let featureStep = featureIterator.next();
    if (featureStep.done) throw new RangeError("at least one feature set is required");
    const length = featureStep.value.aatype.length;
    const pairMask = new Float32Array(length * length);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      pairMask[i * length + j] = featureStep.value.seqMask[i]! * featureStep.value.seqMask[j]!;
    }
    const templateUpdateValue = !this.multimer
      ? (weights as MonomerModelWeights).template === undefined
        ? (() => { throw new Error("AlphaFold monomer weights require a template module"); })()
        : (await new QueryOnlyTemplateGpu(this.device).run({
          length, templateChannels: 64, pairChannels: 128, pairMask,
          weights: (weights as MonomerModelWeights).template,
        })).pairUpdate
      : undefined;
    if (weights.extraStack.length === 0 || weights.mainStack.length === 0) {
      throw new RangeError("AlphaFold monomer requires non-empty extra and main Evoformer stacks");
    }
    if (this.profile && (this.profileRecycle >= featuresByRecycle.length
      || this.profileExtraMsaBlock >= weights.extraStack.length
      || this.profileMainEvoformerBlock >= weights.mainStack.length)) {
      throw new RangeError("requested monomer GPU profile block or recycle is out of range");
    }
    const execution = new WebGpuExecution(this.device, {
      ...(this.compactTransitions ? { transitionBufferLimit: TRANSITION_CHUNK_TARGET_BYTES } : {}),
      ...(this.maxPooledBytes === undefined ? {} : { maxPooledBytes: this.maxPooledBytes }),
    });
    const results: MonomerRecycleSummary[] = [];
    let finalDetails: MonomerFinalDetails | undefined;
    let structureCorePeakResidentBytes = 0;
    let confidencePeakResidentBytes = 0;
    let combinedPeakResidentBytes = 0;
    const start = performance.now();
    const submit = async (encoder: GPUCommandEncoder, label: string): Promise<void> => {
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU ${label} failed: ${error.message}`);
    };
    const releaseTensor = (tensor: GpuTensor): void => {
      tensor.allocation.release();
      execution.allocator.trimPooled();
    };
    try {
      const pairMaskTensor = execution.upload("monomer.pair-mask", pairMask);
      let previousMsa = execution.upload("monomer.recycle-msa-zero", new Float32Array(length * 256));
      let previousPair = execution.upload("monomer.recycle-pair-zero", new Float32Array(length * length * 128),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      let previousPositions = execution.upload(
        "monomer.recycle-positions-zero", new Float32Array(length * 37 * 3),
      );
      let stopAfterRecycle = Number.POSITIVE_INFINITY;
      let previousConvergencePositions: Float32Array | undefined;

      let recycle = 0;
      while (!featureStep.done) {
        const features = featureStep.value;
        if (features.aatype.length !== length) throw new RangeError("all recycle feature lengths must match");
        if (this.multimer && (features.targetChannels !== 21 || features.msaFeatureChannels !== 49
          || features.chainRelative === undefined)) {
          throw new RangeError("Multimer-v3 requires 21 target channels, 49 MSA channels, and chain identifiers");
        }
        const recycleStart = performance.now();
        const msaMask = execution.upload(`monomer.msa-mask-${recycle}`, features.msaMask);
        const extraMsaMask = execution.upload(`monomer.extra-msa-mask-${recycle}`, features.extraMsaMask);
        const embeddingEncoder = this.device.createCommandEncoder({ label: `monomer.embedding-${recycle}` });
        this.device.pushErrorScope("validation");
        const embedding = await encodeInputEmbedder(execution, embeddingEncoder, {
          ...features,
          previousMsaFirstRow: new Float32Array(0), previousPair: new Float32Array(0),
          previousPositions: new Float32Array(0), length,
          msaChannels: 256, pairChannels: 128, extraMsaChannels: 64, weights: weights.embedding,
          ...(features.chainRelative === undefined ? {} : { chainRelative: features.chainRelative }),
        }, previousMsa, previousPair, previousPositions);
        // The template update is constant across recycles but read only here, so
        // it lives on the GPU for one command buffer per recycle instead of the
        // whole prediction; the upload costs far less than holding a pair-shaped
        // tensor through every Evoformer block.
        const templateUpdate = templateUpdateValue === undefined ? undefined
          : execution.upload(`monomer.template-update-${recycle}`, templateUpdateValue);
        if (templateUpdate !== undefined) await execution.addInPlace(
          embeddingEncoder, embedding.pairWithoutTemplates, templateUpdate, `monomer.template-residual-${recycle}`,
        );
        // Multimer merges template rows into the clustered MSA before the extra
        // stack, so it embeds the MSA now. The monomer defers it until the extra
        // stack has finished: nothing there reads it, and keeping the largest
        // tensor of the prediction out of that stack's peak lowers the high-water
        // mark of the whole run.
        const multimerMsa = this.multimer ? embedding.encodeMsa(embeddingEncoder) : undefined;
        await submit(embeddingEncoder, `embedding recycle ${recycle}`);
        // The new pair was written over `previousPair`, which therefore stays live.
        for (const temporary of embedding.temporaries) releaseTensor(temporary);
        releaseTensor(previousPositions);
        if (templateUpdate !== undefined) releaseTensor(templateUpdate);
        const releaseMsaInputs = (): void => {
          for (const temporary of embedding.msaTemporaries) releaseTensor(temporary);
          releaseTensor(previousMsa);
        };
        if (multimerMsa !== undefined) releaseMsaInputs();

        let mainMsaMask = msaMask;
        let mainSequences = features.msaSequences;
        let multimerMainMsa: GpuTensor | undefined;
        let multimerMainMsaMask: GpuTensor | undefined;
        let templateSubmissions = 0;
        if (this.multimer) {
          const multimerWeights = weights as MultimerCompatibleModelWeights;
          mainSequences += multimerWeights.multimerTemplate.templateRows;
          multimerMainMsa = execution.allocate(`multimer.main-msa-${recycle}`,
            mainSequences * length * 256,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
          const combinedMask = new Float32Array(
            features.msaMask.length + multimerWeights.multimerTemplate.templateRows * length,
          );
          combinedMask.set(features.msaMask);
          multimerMainMsaMask = execution.upload(`multimer.main-msa-mask-${recycle}`, combinedMask);
          const templateCheckpoint = execution.checkpoint();
          const template = await new MultimerMockTemplateGpu(this.device).run(
            new Float32Array(0), new Float32Array(0), length, multimerWeights.multimerTemplate, execution,
            { pair: embedding.pairWithoutTemplates, pairMask: pairMaskTensor },
          );
          const templatePair = template.pairUpdateTensor!;
          const templateMsa = template.msaRowsTensor!;
          const templateEncoder = this.device.createCommandEncoder({ label: `multimer.template-merge-${recycle}` });
          this.device.pushErrorScope("validation");
          await execution.addInPlace(templateEncoder, embedding.pairWithoutTemplates, templatePair,
            `multimer.template-pair-residual-${recycle}`);
          execution.endComputePass(templateEncoder);
          templateEncoder.copyBufferToBuffer(multimerMsa!.allocation.buffer, 0,
            multimerMainMsa.allocation.buffer, 0, multimerMsa!.elements * 4);
          templateEncoder.copyBufferToBuffer(templateMsa.allocation.buffer, 0,
            multimerMainMsa.allocation.buffer, multimerMsa!.elements * 4, templateMsa.elements * 4);
          await submit(templateEncoder, `Multimer template merge recycle ${recycle}`);
          templateSubmissions = template.submissions + 1;
          execution.releaseSince(templateCheckpoint);
          mainMsaMask = multimerMainMsaMask;
        }

        const extraShape = {
          sequences: features.extraSequences, length, cM: 64, cZ: 128,
          cOuter: weights.extraStack[0]!.outerProductMean.leftBias.length,
          triangleHidden: weights.extraStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
          ...(this.multimer ? { outerProductMeanFirst: true } : {}),
        };
        const shouldProfileRecycle = this.profile && recycle === this.profileRecycle;
        const timestampProfile = shouldProfileRecycle && this.device.features.has("timestamp-query");
        let extraProfile: MonomerBlockGpuProfile | undefined;
        let extraSubmissions = 0;
        for (let block = 0; block < weights.extraStack.length; block += 1) {
          const profileBlock = shouldProfileRecycle ? this.profileExtraMsaBlock : -1;
          const encoder = this.device.createCommandEncoder({ label: `monomer.extra-${recycle}-${block}` });
          const checkpoint = execution.checkpoint();
          const profiling = block === profileBlock;
          if (profiling && timestampProfile) execution.beginTimestampProfile(512);
          const profileStart = profiling ? performance.now() : 0;
          this.device.pushErrorScope("validation");
          await encodeExtraMsaBlock(execution, encoder, extraShape, weights.extraStack[block]!,
            embedding.extraMsa, embedding.pairWithoutTemplates, extraMsaMask, pairMaskTensor);
          const pendingProfile = profiling && timestampProfile
            ? execution.finishTimestampProfile(encoder) : undefined;
          await submit(encoder, `extra-MSA recycle ${recycle} block ${block}`);
          if (profiling) {
            const entries = pendingProfile === undefined
              ? (await this.device.queue.onSubmittedWorkDone(), [{
                label: `extra-MSA.block-${block}`, nanoseconds: (performance.now() - profileStart) * 1e6,
              }])
              : await execution.readTimestampProfile(pendingProfile);
            extraProfile = {
              block, method: timestampProfile ? "timestamp-query" : "wall-clock",
              wallMilliseconds: performance.now() - profileStart, entries,
            };
          }
          execution.releaseSince(checkpoint);
          extraSubmissions += 1;
        }
        releaseTensor(embedding.extraMsa); releaseTensor(extraMsaMask);
        let clusteredMsa = multimerMsa;
        if (clusteredMsa === undefined) {
          const msaEncoder = this.device.createCommandEncoder({ label: `monomer.msa-embedding-${recycle}` });
          this.device.pushErrorScope("validation");
          clusteredMsa = embedding.encodeMsa(msaEncoder);
          await submit(msaEncoder, `MSA embedding recycle ${recycle}`);
          releaseMsaInputs();
        }
        const mainMsa = multimerMainMsa ?? clusteredMsa;

        const mainDescriptor = {
          msa: new Float32Array(0), pair: new Float32Array(0), msaMask: new Float32Array(0),
          pairMask: new Float32Array(0), sequences: mainSequences, length, cM: 256, cZ: 128,
          cOuter: weights.mainStack[0]!.outerProductMean.leftBias.length,
          triangleHidden: weights.mainStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
          ...(this.multimer ? { outerProductMeanFirst: true } : {}),
        };
        let mainProfile: MonomerBlockGpuProfile | undefined;
        let mainSubmissions = 0;
        for (let block = 0; block < weights.mainStack.length; block += 1) {
          const profileBlock = shouldProfileRecycle ? this.profileMainEvoformerBlock : -1;
          const encoder = this.device.createCommandEncoder({ label: `monomer.main-${recycle}-${block}` });
          const checkpoint = execution.checkpoint();
          const profiling = block === profileBlock;
          if (profiling && timestampProfile) execution.beginTimestampProfile(512);
          const profileStart = profiling ? performance.now() : 0;
          this.device.pushErrorScope("validation");
          await encodeEvoformerBlock(execution, encoder, {
            ...mainDescriptor, weights: weights.mainStack[block]!,
          }, mainMsa, embedding.pairWithoutTemplates, mainMsaMask, pairMaskTensor);
          const pendingProfile = profiling && timestampProfile
            ? execution.finishTimestampProfile(encoder) : undefined;
          await submit(encoder, `main Evoformer recycle ${recycle} block ${block}`);
          if (profiling) {
            const entries = pendingProfile === undefined
              ? (await this.device.queue.onSubmittedWorkDone(), [{
                label: `main-evoformer.block-${block}`, nanoseconds: (performance.now() - profileStart) * 1e6,
              }])
              : await execution.readTimestampProfile(pendingProfile);
            mainProfile = {
              block, method: timestampProfile ? "timestamp-query" : "wall-clock",
              wallMilliseconds: performance.now() - profileStart, entries,
            };
          }
          execution.releaseSince(checkpoint);
          mainSubmissions += 1;
        }
        const readbackEncoder = this.device.createCommandEncoder({ label: `monomer.readback-${recycle}` });
        const msaFirstRowTensor = execution.allocate(
          `monomer.msa-first-row-readback-${recycle}`, length * 256,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        );
        execution.endComputePass(readbackEncoder);
        readbackEncoder.copyBufferToBuffer(
          mainMsa.allocation.buffer, 0, msaFirstRowTensor.allocation.buffer, 0, length * 256 * 4,
        );
        const nextPreviousMsa = execution.allocate(
          `monomer.recycle-msa-${recycle}`, length * 256, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        );
        readbackEncoder.copyBufferToBuffer(
          mainMsa.allocation.buffer, 0, nextPreviousMsa.allocation.buffer, 0, length * 256 * 4,
        );
        this.device.pushErrorScope("validation");
        await submit(readbackEncoder, `readback recycle ${recycle}`);
        const msaFirstRow = await execution.mapFloat32(msaFirstRowTensor);
        releaseTensor(msaFirstRowTensor); releaseTensor(msaMask);
        if (multimerMainMsa !== undefined) releaseTensor(multimerMainMsa);
        if (multimerMainMsaMask !== undefined) releaseTensor(multimerMainMsaMask);
        releaseTensor(clusteredMsa);

        const structure = await new StructureModuleGpu(this.device).run({
          msaFirstRow, pair: new Float32Array(0), mask: features.seqMask, aatype: features.aatype,
          pairBuffer: embedding.pairWithoutTemplates.allocation.buffer,
          atom37ToAtom14: features.atom37ToAtom14, atom37Mask: features.atom37Mask,
          length, weights: weights.structure, geometry: weights.geometry,
          ...(this.multimer ? { multimer: true } : {}),
        });
        const trunkResidentBytes = execution.snapshot().residentBytes;
        const structurePeak = structure.memory?.peakResidentBytes ?? 0;
        structureCorePeakResidentBytes = Math.max(structureCorePeakResidentBytes, structurePeak);
        combinedPeakResidentBytes = Math.max(
          combinedPeakResidentBytes, trunkResidentBytes + structurePeak,
        );
        const confidence = await new ConfidenceHeadsGpu(this.device).runReduced(
          structure.finalRepresentation, new Float32Array(0), length, weights.lddt, weights.pae, paeBreaks,
          { pairBuffer: embedding.pairWithoutTemplates.allocation.buffer },
        );
        const confidencePeak = confidence.memory?.peakResidentBytes ?? 0;
        confidencePeakResidentBytes = Math.max(confidencePeakResidentBytes, confidencePeak);
        combinedPeakResidentBytes = Math.max(
          combinedPeakResidentBytes, trunkResidentBytes + confidencePeak,
        );
        const trunkSubmissions = {
          embedding: 1, template: templateSubmissions,
          extraMsa: extraSubmissions, mainEvoformer: mainSubmissions, readback: 1,
          total: 2 + templateSubmissions + extraSubmissions + mainSubmissions,
        };
        const gpuProfile = extraProfile === undefined || mainProfile === undefined
          ? undefined : { extraMsa: extraProfile, mainEvoformer: mainProfile };
        const recycleDetails = { msaFirstRow, structure, confidence,
          elapsedMilliseconds: performance.now() - recycleStart, trunkSubmissions,
          ...(gpuProfile === undefined ? {} : { gpuProfile }),
        };
        const recycleSummary = summarizeMonomerRecycle(recycleDetails);
        onRecycleDetails?.(recycleDetails, recycle, features);
        onRecycle?.(recycleSummary, recycle);
        results.push(recycleSummary);
        finalDetails = {
          ...recycleDetails,
          confidence: {
            plddt: confidence.plddt,
            meanPlddt: confidence.meanPlddt,
            predictedAlignedError: confidence.predictedAlignedError,
            maxPredictedAlignedError: confidence.maxPredictedAlignedError,
            ptm: confidence.ptm,
            ...(confidence.memory === undefined ? {} : { memory: confidence.memory }),
          },
        };
        previousMsa = nextPreviousMsa;
        previousPair = embedding.pairWithoutTemplates;
        previousPositions = execution.upload(`monomer.recycle-positions-${recycle}`, structure.atom37);
        if (this.multimer && recycle > 0 && this.recycleEarlyStopTolerance >= 0
          && stopAfterRecycle === Number.POSITIVE_INFINITY) {
          const rms = multimerRecycleDistanceRms(
            previousConvergencePositions!, structure.atom37, features.seqMask,
          );
          if (rms <= this.recycleEarlyStopTolerance) stopAfterRecycle = recycle + 1;
        }
        previousConvergencePositions = structure.atom37;
        if (recycle >= stopAfterRecycle) {
          featureIterator.return?.();
          break;
        }
        recycle += 1;
        featureStep = featureIterator.next();
      }
      if (finalDetails === undefined) throw new Error("monomer prediction produced no recycle result");
      const finalReadbackEncoder = this.device.createCommandEncoder({ label: "monomer.final-pair-readback" });
      const finalPairReadback = execution.createReadback(
        "monomer.final-pair-readback", previousPair, finalReadbackEncoder,
      );
      this.device.pushErrorScope("validation");
      await submit(finalReadbackEncoder, "final pair readback");
      const finalPair = await execution.mapFloat32(finalPairReadback);
      releaseTensor(finalPairReadback);
      const finalResult: MonomerRecycleResult = { ...finalDetails, pair: finalPair };
      const mainMemory = execution.snapshot();
      return {
        recycles: results, final: finalResult, elapsedMilliseconds: performance.now() - start,
        memory: {
          ...mainMemory,
          mainPeakResidentBytes: mainMemory.peakResidentBytes,
          structureCorePeakResidentBytes,
          confidencePeakResidentBytes,
          combinedPeakResidentBytes: Math.max(combinedPeakResidentBytes, mainMemory.peakResidentBytes),
        },
      };
    } finally {
      execution.release();
    }
  }
}
