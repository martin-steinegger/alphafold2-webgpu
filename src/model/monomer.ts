import { CLUSTERED_MSA_CHANNELS } from "../input/msa-features.js";
import {
  ConfidenceHeadsGpu,
  type ConfidenceSummaryResult,
  type PredictedAlignedErrorWeights,
  type PredictedLddtWeights,
  type ReducedConfidenceResult,
} from "../heads/confidence.js";
import { encodeInputEmbedder, type InputEmbedderWeights } from "../evoformer/input-embedder.js";
import type { TriangleWholeStorage } from "../triangle/shaders.js";
import { type ActivationStorage, storageWords, unpackHalfWords } from "../runtime/storage.js";
import {
  encodeEvoformerBlock, encodeExtraMsaBlock, type EvoformerBlockWeights, type ExtraMsaBlockWeights,
} from "../evoformer/block.js";
import {
  QueryOnlyTemplateGpu, queryOnlyTemplateConstant, type QueryOnlyTemplateWeights,
} from "../evoformer/template.js";
import { MultimerMockTemplateGpu, type MultimerMockTemplateWeights } from "../evoformer/multimer-template.js";
import { WebGpuExecution, type GpuTensor, type GpuTimestampEntry } from "../runtime/execution.js";
import { StructureModuleGpu, type StructureModuleResult, type StructureModuleWeights } from "../structure/module.js";
import type { ResidueGeometryTables } from "../structure/geometry.js";
import {
  iterateA3mFeatures, type A3mFeatureOptions, type RecycleFeatureSource,
} from "../input/a3m-features.js";
import type { QueryOnlyFeatureTables } from "../input/query-only-features.js";
import { TRANSITION_CHUNK_TARGET_BYTES } from "../evoformer/transition.js";
import { COMPACT_GPU_POOL_BYTES, type AllocationShare, type AllocationSnapshot } from "../runtime/allocator.js";
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

/**
 * The exact storages the differential tests compare the model against.
 *
 * Predictions run packed: measurement on real alignments put the difference
 * below the confidence's own resolution, and the memory it saves is what
 * decides whether a chain runs at all in a browser. The exact path stays
 * reachable so the kernels can still be checked against AlphaFold's own
 * tensors, which are f32.
 */
export const EXACT_STORAGE = {
  triangleWholeStorage: "f32", msaStorage: "f32", pairStorage: "f32",
} as const;

/** Stands in for the pair when a prediction did not ask to read it back. */
const EMPTY_PAIR = new Float32Array(0);

export interface MonomerRecycleResult {
  readonly msaFirstRow: Float32Array;
  /** The final pair, empty unless the model was built with `returnFinalPair`. */
  readonly pair: Float32Array;
  readonly structure: StructureModuleResult; readonly confidence: PredictionConfidenceResult;
  readonly elapsedMilliseconds: number;
  readonly trunkSubmissions: MonomerTrunkSubmissionCounts;
  readonly gpuProfile?: MonomerRecycleGpuProfile;
}

/** Full per-recycle data used transiently before the final pair readback. */
export type MonomerRecycleDetails = Omit<MonomerRecycleResult, "pair" | "confidence"> & {
  readonly confidence: ReducedConfidenceResult;
  /** Wall time of the monomer's per-recycle template update, when it ran. */
  readonly templateMilliseconds?: number;
};

type MonomerFinalDetails = Omit<MonomerRecycleResult, "pair">;

/** Evoformer command buffers the host may run ahead of the GPU. */
const BLOCKS_IN_FLIGHT = 4;

/**
 * Pair size from which the monomer's template update is recomputed per recycle
 * rather than kept on the host: 32 MiB, about 256 residues. Below it the copy
 * is small and the template stack is a sizeable share of a short recycle.
 */
const TEMPLATE_RECOMPUTE_BYTES = 32 * 1024 ** 2;

export interface MonomerRecycleSummary {
  readonly confidence: Pick<ConfidenceSummaryResult, "meanPlddt" | "ptm">;
  readonly elapsedMilliseconds: number;
  /** Wall time of the monomer's per-recycle template update, when it ran. */
  readonly templateMilliseconds?: number;
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
  /** What was live when the structure module reached its own peak. */
  readonly structurePeakComposition?: readonly AllocationShare[];
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
  /** Storage of the triangle multiplication's whole projection; `f16` halves it inexactly. */
  readonly triangleWholeStorage?: TriangleWholeStorage;
  /** Storage of the MSA activations; `f16` halves them inexactly. Monomer only. */
  readonly msaStorage?: ActivationStorage;
  /**
   * Storage of the pair; `f16` halves it inexactly. Monomer only.
   *
   * The pair is one of the three tensors that set the trunk's peak, beside the
   * MSA activations and the triangle multiplication's whole projection.
   */
  readonly pairStorage?: ActivationStorage;
  /** Caps reusable scratch retained between blocks; compact mode uses the bounded shared default. */
  readonly maxPooledBytes?: number;
  /** Internal model architecture selector used by AlphaFoldMultimerGpu. */
  readonly multimer?: boolean;
  /** Multimer CA-distance RMS threshold; negative disables early stopping. */
  readonly recycleEarlyStopTolerance?: number;
  /**
   * Reads the final pair representation back to the host.
   *
   * Only differential testing wants it: at 1000 residues the pair is 488 MiB,
   * and returning it costs that much again in a mapped staging buffer and once
   * more in the host copy, at the point where the run is otherwise done and
   * the device is at its fullest. Predictions leave it off and get an empty
   * array in `final.pair`.
   */
  readonly returnFinalPair?: boolean;
  /**
   * Collapses the query-only template module to the constant it computes.
   *
   * Off, the module runs on the real chain, which is what the differential
   * tests compare against.
   */
  readonly collapseQueryOnlyTemplate?: boolean;
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
    ...(result.templateMilliseconds === undefined ? {} : { templateMilliseconds: result.templateMilliseconds }),
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
  readonly triangleWholeStorage: TriangleWholeStorage;
  readonly msaStorage: ActivationStorage;
  readonly pairStorage: ActivationStorage;
  readonly multimer: boolean;
  readonly recycleEarlyStopTolerance: number;
  readonly returnFinalPair: boolean;
  readonly collapseQueryOnlyTemplate: boolean;
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
    this.returnFinalPair = options.returnFinalPair ?? false;
    this.collapseQueryOnlyTemplate = options.collapseQueryOnlyTemplate ?? true;
    this.triangleWholeStorage = options.triangleWholeStorage ?? "f16";
    this.msaStorage = options.msaStorage ?? "f16";
    this.pairStorage = options.pairStorage ?? "f16";
    if (this.triangleWholeStorage !== "f32" && this.triangleWholeStorage !== "f16") {
      throw new RangeError("triangle whole storage must be f32 or f16");
    }
    if (this.msaStorage !== "f32" && this.msaStorage !== "f16") {
      throw new RangeError("MSA storage must be f32 or f16");
    }
    if (this.pairStorage !== "f32" && this.pairStorage !== "f16") {
      throw new RangeError("pair storage must be f32 or f16");
    }


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
    // The monomer's template update depends only on the length and mask. For
    // long chains it is recomputed on the GPU each recycle and added into the
    // resident pair, so no pair-sized copy of it lives on the host or the
    // device between uses; for short chains, where that copy is small and the
    // template stack is a sizeable share of a recycle, it is computed once and
    // uploaded per recycle instead.
    const templateWeights = this.multimer ? undefined : (weights as MonomerModelWeights).template;
    if (!this.multimer && templateWeights === undefined) {
      throw new Error("AlphaFold monomer weights require a template module");
    }
    // With no template search the module's update is one vector repeated over
    // every pair, so it is computed once on a handful of residues and folded
    // into the pair projection's bias, which the embedder already adds to
    // every pair element. That removes the template stack from the trunk: no
    // per-recycle pass over the pair, and none of the hundreds of MiB its own
    // activations take at long lengths. It holds only for an unpadded chain,
    // where the pair mask is everywhere one; anything else runs the module.
    const uniformPairMask = pairMask.every((value) => value === 1);
    let embeddingWeights = weights.embedding;
    let templateConstantMilliseconds: number | undefined;
    let templateConstantApplied = false;
    if (this.collapseQueryOnlyTemplate && templateWeights !== undefined && uniformPairMask) {
      const started = performance.now();
      const constant = await queryOnlyTemplateConstant(this.device, templateWeights);
      if (constant !== undefined) {
        const leftSingleBias = Float32Array.from(weights.embedding.leftSingleBias);
        if (leftSingleBias.length !== constant.length) {
          throw new RangeError("template constant and pair projection bias disagree on channels");
        }
        for (let channel = 0; channel < leftSingleBias.length; channel += 1) {
          leftSingleBias[channel] = leftSingleBias[channel]! + constant[channel]!;
        }
        embeddingWeights = { ...weights.embedding, leftSingleBias };
        templateConstantApplied = true;
        templateConstantMilliseconds = performance.now() - started;
      }
    }
    const templateModule = templateWeights === undefined || templateConstantApplied
      ? undefined : new QueryOnlyTemplateGpu(this.device);

    const recomputeTemplate = length * length * 128 * 4 >= TEMPLATE_RECOMPUTE_BYTES;
    const templateUpdateValue = templateModule !== undefined && templateWeights !== undefined && !recomputeTemplate
      ? (await templateModule.run({
        length, templateChannels: 64, pairChannels: 128, pairMask, weights: templateWeights,
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
    let structurePeakComposition: readonly AllocationShare[] | undefined;
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
    // Evoformer blocks are submitted ahead of the GPU: a block's command buffer
    // goes to the queue as soon as it is encoded, and the host encodes the next
    // one while the GPU runs it. Waiting for each block's validation result
    // before encoding the next left the GPU idle for a host round trip per
    // block, which at short lengths is a sizeable share of the block itself.
    // A rolling window bounds how far ahead the host runs, which also bounds the
    // upload staging the implementation holds for the in-flight weights.
    // Validation errors are collected and settled at the window's pace and
    // before anything is read back; queue ordering keeps scratch reuse safe.
    const pendingErrors: { readonly label: string; readonly error: Promise<GPUError | null> }[] = [];
    const inFlight: Promise<undefined>[] = [];
    const settleErrors = async (): Promise<void> => {
      const pending = pendingErrors.splice(0);
      for (const { label, error } of pending) {
        const result = await error;
        if (result !== null) throw new Error(`WebGPU ${label} failed: ${result.message}`);
      }
    };
    const submitAhead = async (encoder: GPUCommandEncoder, label: string): Promise<void> => {
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      pendingErrors.push({ label, error: this.device.popErrorScope() });
      inFlight.push(this.device.queue.onSubmittedWorkDone());
      if (inFlight.length > BLOCKS_IN_FLIGHT) await inFlight.shift();
      if (pendingErrors.length > BLOCKS_IN_FLIGHT) await settleErrors();
    };
    const releaseTensor = (tensor: GpuTensor): void => {
      tensor.allocation.release();
      execution.allocator.trimPooled();
    };
    try {
      const pairMaskTensor = execution.upload("monomer.pair-mask", pairMask);
      // The first recycle reads zeros. A pooled buffer carries whatever the
      // last allocation left in it, so the zeros are written on the device:
      // uploading them would cost a pair-sized host array (488 MiB at 1000
      // residues) and a copy of it, to send across what the GPU can clear for
      // nothing.
      let previousMsa = execution.allocate("monomer.recycle-msa-zero", length * 256,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      let previousPair = execution.allocate("monomer.recycle-pair-zero",
        storageWords(length * length * 128, this.pairStorage),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      let previousPositions = execution.allocate("monomer.recycle-positions-zero", length * 37 * 3,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
      const clearEncoder = this.device.createCommandEncoder({ label: "monomer.recycle-zero" });
      for (const tensor of [previousMsa, previousPair, previousPositions]) {
        clearEncoder.clearBuffer(tensor.allocation.buffer, 0, tensor.allocation.byteLength);
      }
      this.device.pushErrorScope("validation");
      await submit(clearEncoder, "recycle zero");
      let stopAfterRecycle = Number.POSITIVE_INFINITY;
      let previousConvergencePositions: Float32Array | undefined;

      let recycle = 0;
      while (!featureStep.done) {
        const features = featureStep.value;
        if (features.aatype.length !== length) throw new RangeError("all recycle feature lengths must match");
        if (this.multimer && (features.targetChannels !== 21
          || features.msaFeatureChannels !== CLUSTERED_MSA_CHANNELS
          || features.chainRelative === undefined)) {
          throw new RangeError("Multimer-v3 requires 21 target channels, compact MSA features, and chain identifiers");
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
          msaChannels: 256, pairChannels: 128, extraMsaChannels: 64, weights: embeddingWeights,
          msaStorage: this.msaStorage, pairStorage: this.pairStorage,
          ...(features.chainRelative === undefined ? {} : { chainRelative: features.chainRelative }),
        }, previousMsa, previousPair, previousPositions);
        // The clustered MSA is embedded only after the extra stack, which never
        // reads it, so the largest tensor of the prediction stays out of that
        // stack's peak. Multimer's template rows, which are merged into it, are
        // kept aside until then.
        await submit(embeddingEncoder, `embedding recycle ${recycle}`);
        // The new pair was written over `previousPair`, which therefore stays live.
        for (const temporary of embedding.temporaries) releaseTensor(temporary);
        releaseTensor(previousPositions);
        let templateMilliseconds: number | undefined = templateConstantMilliseconds;
        templateConstantMilliseconds = undefined;
        if (templateUpdateValue !== undefined) {
          const templateStart = performance.now();
          const templateUpdate = execution.upload(`monomer.template-update-${recycle}`, templateUpdateValue);
          const templateEncoder = this.device.createCommandEncoder({ label: `monomer.template-residual-${recycle}` });
          this.device.pushErrorScope("validation");
          await execution.addInPlace(templateEncoder, embedding.pairWithoutTemplates, templateUpdate,
            `monomer.template-residual-${recycle}`, this.pairStorage);
          await submit(templateEncoder, `template residual recycle ${recycle}`);
          releaseTensor(templateUpdate);
          templateMilliseconds = performance.now() - templateStart;
        } else if (templateModule !== undefined && templateWeights !== undefined) {
          const templateStart = performance.now();
          const templateInput = {
            length, templateChannels: 64, pairChannels: 128, pairMask, weights: templateWeights,
          };
          if (this.pairStorage === "f32") {
            await templateModule.run(templateInput,
              { execution, pair: embedding.pairWithoutTemplates, pairMask: pairMaskTensor });
          } else {
            // The module writes f32, so a packed pair takes its update through
            // the packed residual add rather than letting it write the pair.
            const update = await templateModule.run(templateInput);
            const templateUpdate = execution.upload(`monomer.template-update-${recycle}`, update.pairUpdate);
            const templateEncoder = this.device.createCommandEncoder(
              { label: `monomer.template-residual-${recycle}` });
            this.device.pushErrorScope("validation");
            await execution.addInPlace(templateEncoder, embedding.pairWithoutTemplates, templateUpdate,
              `monomer.template-residual-${recycle}`, this.pairStorage);
            await submit(templateEncoder, `template residual recycle ${recycle}`);
            releaseTensor(templateUpdate);
          }
          templateMilliseconds = performance.now() - templateStart;
          // The template stack's scratch shapes recur only next recycle.
          execution.allocator.destroyPooled();
        }
        const releaseMsaInputs = (): void => {
          for (const temporary of embedding.msaTemporaries) releaseTensor(temporary);
          releaseTensor(previousMsa);
        };

        let mainMsaMask = msaMask;
        let mainSequences = features.msaSequences;
        let multimerMainMsa: GpuTensor | undefined;
        let multimerMainMsaMask: GpuTensor | undefined;
        let templateRows: GpuTensor | undefined;
        let templateSubmissions = 0;
        if (this.multimer) {
          // The template module's pair update is needed before the extra stack;
          // its MSA rows are not, so only they are kept (a few rows) and the
          // rest of the module's tensors retire at once.
          const multimerWeights = weights as MultimerCompatibleModelWeights;
          mainSequences += multimerWeights.multimerTemplate.templateRows;
          templateRows = execution.allocate(`multimer.template-msa-rows-${recycle}`,
            storageWords(multimerWeights.multimerTemplate.templateRows * length * 256, this.msaStorage),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
          const templateCheckpoint = execution.checkpoint();
          const template = await new MultimerMockTemplateGpu(this.device).run(
            new Float32Array(0), new Float32Array(0), length, multimerWeights.multimerTemplate, execution,
            { pair: embedding.pairWithoutTemplates, pairMask: pairMaskTensor, pairStorage: this.pairStorage },
          );
          const templatePair = template.pairUpdateTensor!;
          const templateMsa = template.msaRowsTensor!;
          const templateEncoder = this.device.createCommandEncoder({ label: `multimer.template-merge-${recycle}` });
          this.device.pushErrorScope("validation");
          await execution.addInPlace(templateEncoder, embedding.pairWithoutTemplates, templatePair,
            `multimer.template-pair-residual-${recycle}`, this.pairStorage);
          if (this.msaStorage === "f32") {
            execution.endComputePass(templateEncoder);
            templateEncoder.copyBufferToBuffer(templateMsa.allocation.buffer, 0,
              templateRows.allocation.buffer, 0, templateMsa.elements * 4);
          } else {
            // The rows join a packed MSA, so they are packed on the way.
            await execution.packHalves(templateEncoder, templateMsa, templateRows,
              `multimer.template-msa-pack-${recycle}`);
          }
          await submit(templateEncoder, `Multimer template pair update recycle ${recycle}`);
          templateSubmissions = template.submissions + 1;
          execution.releaseSince(templateCheckpoint);
        }

        const extraShape = {
          sequences: features.extraSequences, length, cM: 64, cZ: 128,
          cOuter: weights.extraStack[0]!.outerProductMean.leftBias.length,
          triangleHidden: weights.extraStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
          triangleWholeStorage: this.triangleWholeStorage, msaStorage: this.msaStorage,
          pairStorage: this.pairStorage,
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
          if (profiling) await submit(encoder, `extra-MSA recycle ${recycle} block ${block}`);
          else await submitAhead(encoder, `extra-MSA recycle ${recycle} block ${block}`);
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
        await settleErrors();
        releaseTensor(embedding.extraMsa); releaseTensor(extraMsaMask);
        // The extra stack's retired scratch would otherwise sit in the pool
        // beside the main stack's while the clustered MSA comes to life; the
        // main stack recreates the shapes it shares once.
        execution.allocator.destroyPooled();
        const msaEncoder = this.device.createCommandEncoder({ label: `monomer.msa-embedding-${recycle}` });
        this.device.pushErrorScope("validation");
        const clusteredMsa = embedding.encodeMsa(msaEncoder);
        await submit(msaEncoder, `MSA embedding recycle ${recycle}`);
        releaseMsaInputs();
        // The embedder's inputs retired just now and nothing in the trunk fits them.
        execution.allocator.destroyPooled();
        let mainMsa = clusteredMsa;
        if (templateRows !== undefined) {
          // Multimer's main-stack MSA is the clustered rows followed by the
          // template rows kept from before the extra stack.
          const multimerWeights = weights as MultimerCompatibleModelWeights;
          multimerMainMsa = execution.allocate(`multimer.main-msa-${recycle}`,
            storageWords(mainSequences * length * 256, this.msaStorage),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
          const combinedMask = new Float32Array(
            features.msaMask.length + multimerWeights.multimerTemplate.templateRows * length,
          );
          combinedMask.set(features.msaMask);
          multimerMainMsaMask = execution.upload(`multimer.main-msa-mask-${recycle}`, combinedMask);
          const mergeEncoder = this.device.createCommandEncoder({ label: `multimer.msa-merge-${recycle}` });
          this.device.pushErrorScope("validation");
          mergeEncoder.copyBufferToBuffer(clusteredMsa.allocation.buffer, 0,
            multimerMainMsa.allocation.buffer, 0, clusteredMsa.elements * 4);
          mergeEncoder.copyBufferToBuffer(templateRows.allocation.buffer, 0,
            multimerMainMsa.allocation.buffer, clusteredMsa.elements * 4, templateRows.elements * 4);
          await submit(mergeEncoder, `Multimer MSA merge recycle ${recycle}`);
          releaseTensor(clusteredMsa); releaseTensor(templateRows);
          mainMsaMask = multimerMainMsaMask;
          mainMsa = multimerMainMsa;
        }

        const mainDescriptor = {
          msa: new Float32Array(0), pair: new Float32Array(0), msaMask: new Float32Array(0),
          pairMask: new Float32Array(0), sequences: mainSequences, length, cM: 256, cZ: 128,
          cOuter: weights.mainStack[0]!.outerProductMean.leftBias.length,
          triangleHidden: weights.mainStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
          triangleWholeStorage: this.triangleWholeStorage, msaStorage: this.msaStorage,
          pairStorage: this.pairStorage,
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
          if (profiling) await submit(encoder, `main Evoformer recycle ${recycle} block ${block}`);
          else await submitAhead(encoder, `main Evoformer recycle ${recycle} block ${block}`);
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
        await settleErrors();
        const readbackEncoder = this.device.createCommandEncoder({ label: `monomer.readback-${recycle}` });
        const firstRowWords = storageWords(length * 256, this.msaStorage);
        const msaFirstRowTensor = execution.allocate(
          `monomer.msa-first-row-readback-${recycle}`, firstRowWords,
          GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        );
        execution.endComputePass(readbackEncoder);
        readbackEncoder.copyBufferToBuffer(
          mainMsa.allocation.buffer, 0, msaFirstRowTensor.allocation.buffer, 0, firstRowWords * 4,
        );
        // The recycled first row feeds the next embedder as f32. Packed storage
        // goes through the host, where the readback is unpacked anyway.
        const nextPreviousMsaCopy = this.msaStorage === "f32" ? execution.allocate(
          `monomer.recycle-msa-${recycle}`, length * 256, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        ) : undefined;
        if (nextPreviousMsaCopy !== undefined) {
          readbackEncoder.copyBufferToBuffer(
            mainMsa.allocation.buffer, 0, nextPreviousMsaCopy.allocation.buffer, 0, length * 256 * 4,
          );
        }
        this.device.pushErrorScope("validation");
        await submit(readbackEncoder, `readback recycle ${recycle}`);
        const firstRowMapped = await execution.mapFloat32(msaFirstRowTensor);
        const msaFirstRow = this.msaStorage === "f32" ? firstRowMapped
          : unpackHalfWords(new Uint32Array(firstRowMapped.buffer, firstRowMapped.byteOffset, firstRowWords), length * 256);
        const nextPreviousMsa = nextPreviousMsaCopy
          ?? execution.upload(`monomer.recycle-msa-${recycle}`, msaFirstRow);
        releaseTensor(msaFirstRowTensor); releaseTensor(msaMask);
        if (multimerMainMsa !== undefined) releaseTensor(multimerMainMsa);
        if (multimerMainMsaMask !== undefined) releaseTensor(multimerMainMsaMask);
        releaseTensor(clusteredMsa);
        // The trunk is complete and read back, so nothing references its
        // retired scratch. Dropping the pool here keeps the resident footprint
        // at the live tensors while the structure module and confidence heads
        // allocate their own working sets, and between recycles; the next
        // recycle recreates its handful of large buffers once.
        execution.allocator.destroyPooled();

        // The structure module and the confidence heads read the trunk's pair
        // where it lies, packed or not, so no expanded copy is ever made.
        const headsPair = embedding.pairWithoutTemplates;
        const structure = await new StructureModuleGpu(this.device).run({
          msaFirstRow, pair: new Float32Array(0), mask: features.seqMask, aatype: features.aatype,
          pairBuffer: headsPair.allocation.buffer,
          ...(this.pairStorage === "f32" ? {} : { pairStorage: this.pairStorage }),
          atom37ToAtom14: features.atom37ToAtom14, atom37Mask: features.atom37Mask,
          length, weights: weights.structure, geometry: weights.geometry,
          ...(this.multimer ? { multimer: true } : {}),
        });
        const trunkResidentBytes = execution.snapshot().residentBytes;
        const structurePeak = structure.memory?.peakResidentBytes ?? 0;
        if (structurePeak > structureCorePeakResidentBytes) {
          structurePeakComposition = structure.memory?.peakComposition;
        }
        structureCorePeakResidentBytes = Math.max(structureCorePeakResidentBytes, structurePeak);
        combinedPeakResidentBytes = Math.max(
          combinedPeakResidentBytes, trunkResidentBytes + structurePeak,
        );
        const confidence = await new ConfidenceHeadsGpu(this.device).runReduced(
          structure.finalRepresentation, new Float32Array(0), length, weights.lddt, weights.pae, paeBreaks,
          { pairBuffer: headsPair.allocation.buffer,
            ...(this.pairStorage === "f32" ? {} : { pairStorage: this.pairStorage }) },
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
          ...(templateMilliseconds === undefined ? {} : { templateMilliseconds }),
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
      let finalPair: Float32Array = EMPTY_PAIR;
      if (this.returnFinalPair) {
        const finalReadbackEncoder = this.device.createCommandEncoder({ label: "monomer.final-pair-readback" });
        const finalPairReadback = execution.createReadback(
          "monomer.final-pair-readback", previousPair, finalReadbackEncoder,
        );
        this.device.pushErrorScope("validation");
        await submit(finalReadbackEncoder, "final pair readback");
        const mapped = await execution.mapFloat32(finalPairReadback);
        finalPair = this.pairStorage === "f32" ? mapped
          : unpackHalfWords(new Uint32Array(mapped.buffer, mapped.byteOffset, mapped.length),
            length * length * 128);
        releaseTensor(finalPairReadback);
      }
      const finalResult: MonomerRecycleResult = { ...finalDetails, pair: finalPair };
      const mainMemory = execution.snapshot();
      return {
        recycles: results, final: finalResult, elapsedMilliseconds: performance.now() - start,
        memory: {
          ...mainMemory,
          mainPeakResidentBytes: mainMemory.peakResidentBytes,
          structureCorePeakResidentBytes,
          ...(structurePeakComposition === undefined ? {} : { structurePeakComposition }),
          confidencePeakResidentBytes,
          combinedPeakResidentBytes: Math.max(combinedPeakResidentBytes, mainMemory.peakResidentBytes),
        },
      };
    } finally {
      execution.release();
    }
  }
}
