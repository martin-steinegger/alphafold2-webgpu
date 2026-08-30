import { ConfidenceHeadsGpu, type ConfidenceResult, type PredictedAlignedErrorWeights, type PredictedLddtWeights } from "../heads/confidence.js";
import { encodeInputEmbedder, type InputEmbedderWeights } from "../evoformer/input-embedder.js";
import {
  encodeEvoformerBlock, encodeExtraMsaBlock, type EvoformerBlockWeights, type ExtraMsaBlockWeights,
} from "../evoformer/block.js";
import { QueryOnlyTemplateGpu, type QueryOnlyTemplateWeights } from "../evoformer/template.js";
import { WebGpuExecution, type GpuTensor, type GpuTimestampEntry } from "../runtime/execution.js";
import { StructureModuleGpu, type StructureModuleResult, type StructureModuleWeights } from "../structure/module.js";
import type { ResidueGeometryTables } from "../structure/geometry.js";
import { makeA3mFeatures, type A3mFeatureOptions } from "../input/a3m-features.js";
import type { QueryOnlyFeatureTables } from "../input/query-only-features.js";
import { TRANSITION_CHUNK_TARGET_BYTES } from "../evoformer/transition.js";
import type { AllocationSnapshot } from "../runtime/allocator.js";

export interface MonomerRecycleFeatures {
  readonly targetFeatures: Float32Array; readonly msaFeatures: Float32Array; readonly msaMask: Float32Array;
  readonly extraMsa: Float32Array; readonly extraHasDeletion: Float32Array; readonly extraDeletionValue: Float32Array;
  readonly extraMsaMask: Float32Array; readonly residueIndex: Float32Array; readonly aatype: Float32Array;
  readonly seqMask: Float32Array; readonly atom37ToAtom14: Float32Array; readonly atom37Mask: Float32Array;
  readonly msaSequences: number; readonly extraSequences: number;
  readonly targetChannels: number; readonly msaFeatureChannels: number;
}

export interface MonomerModelWeights {
  readonly embedding: InputEmbedderWeights; readonly template: QueryOnlyTemplateWeights;
  readonly extraStack: readonly ExtraMsaBlockWeights[]; readonly mainStack: readonly EvoformerBlockWeights[];
  readonly structure: StructureModuleWeights; readonly lddt: PredictedLddtWeights;
  readonly pae: PredictedAlignedErrorWeights; readonly geometry: ResidueGeometryTables;
}

export interface MonomerRecycleResult {
  readonly msaFirstRow: Float32Array; readonly pair: Float32Array;
  readonly structure: StructureModuleResult; readonly confidence: ConfidenceResult;
  readonly elapsedMilliseconds: number;
  readonly trunkSubmissions: MonomerTrunkSubmissionCounts;
  readonly gpuProfile?: MonomerRecycleGpuProfile;
}

export interface MonomerPrediction {
  readonly recycles: readonly MonomerRecycleResult[]; readonly final: MonomerRecycleResult;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

export interface MonomerTrunkSubmissionCounts {
  readonly embedding: number; readonly extraMsa: number; readonly mainEvoformer: number;
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

export type MonomerRecycleCallback = (result: MonomerRecycleResult, recycle: number) => void;

/** Full monomer model for clustered MSA/A3M inputs, with all learned operations dispatched through WebGPU. */
export class AlphaFoldMonomerGpu {
  readonly device: GPUDevice;
  readonly profile: boolean;
  readonly profileRecycle: number;
  readonly profileExtraMsaBlock: number;
  readonly profileMainEvoformerBlock: number;
  readonly compactTransitions: boolean;
  constructor(device: GPUDevice, options: MonomerGpuOptions = {}) {
    this.device = device;
    this.profile = options.profile ?? false;
    this.profileRecycle = options.profileRecycle ?? 0;
    this.profileExtraMsaBlock = options.profileExtraMsaBlock ?? 0;
    this.profileMainEvoformerBlock = options.profileMainEvoformerBlock ?? 0;
    this.compactTransitions = options.compactTransitions ?? false;
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
    return this.predict(makeA3mFeatures(a3mText, featureTables, options), weights, paeBreaks, onRecycle);
  }
  async predict(featuresByRecycle: readonly MonomerRecycleFeatures[], weights: MonomerModelWeights,
    paeBreaks?: Float32Array, onRecycle?: MonomerRecycleCallback): Promise<MonomerPrediction> {
    if (featuresByRecycle.length === 0) throw new RangeError("at least one feature set is required");
    const length = featuresByRecycle[0]!.aatype.length;
    const pairMask = new Float32Array(length * length);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      pairMask[i * length + j] = featuresByRecycle[0]!.seqMask[i]! * featuresByRecycle[0]!.seqMask[j]!;
    }
    const template = await new QueryOnlyTemplateGpu(this.device).run({
      length, templateChannels: 64, pairChannels: 128, pairMask, weights: weights.template,
    });
    if (weights.extraStack.length === 0 || weights.mainStack.length === 0) {
      throw new RangeError("AlphaFold monomer requires non-empty extra and main Evoformer stacks");
    }
    if (this.profile && (this.profileRecycle >= featuresByRecycle.length
      || this.profileExtraMsaBlock >= weights.extraStack.length
      || this.profileMainEvoformerBlock >= weights.mainStack.length)) {
      throw new RangeError("requested monomer GPU profile block or recycle is out of range");
    }
    const execution = new WebGpuExecution(this.device, this.compactTransitions
      ? { transitionBufferLimit: TRANSITION_CHUNK_TARGET_BYTES } : {});
    const results: MonomerRecycleResult[] = [];
    const start = performance.now();
    const submit = async (encoder: GPUCommandEncoder, label: string): Promise<void> => {
      execution.endComputePass(encoder);
      this.device.queue.submit([encoder.finish()]);
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU ${label} failed: ${error.message}`);
    };
    const releaseTensor = (tensor: GpuTensor): void => tensor.allocation.release();
    try {
      const templateUpdate = execution.upload("monomer.template-update", template.pairUpdate);
      const pairMaskTensor = execution.upload("monomer.pair-mask", pairMask);
      let previousMsa = execution.upload("monomer.recycle-msa-zero", new Float32Array(length * 256));
      let previousPair = execution.upload("monomer.recycle-pair-zero", new Float32Array(length * length * 128));
      let previousPositions = execution.upload(
        "monomer.recycle-positions-zero", new Float32Array(length * 37 * 3),
      );

      for (let recycle = 0; recycle < featuresByRecycle.length; recycle += 1) {
        const features = featuresByRecycle[recycle]!;
        if (features.aatype.length !== length) throw new RangeError("all recycle feature lengths must match");
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
        }, previousMsa, previousPair, previousPositions);
        await execution.addInPlace(
          embeddingEncoder, embedding.pairWithoutTemplates, templateUpdate, `monomer.template-residual-${recycle}`,
        );
        await submit(embeddingEncoder, `embedding recycle ${recycle}`);
        for (const temporary of embedding.temporaries) releaseTensor(temporary);
        releaseTensor(previousMsa); releaseTensor(previousPair); releaseTensor(previousPositions);

        const extraShape = {
          sequences: features.extraSequences, length, cM: 64, cZ: 128,
          cOuter: weights.extraStack[0]!.outerProductMean.leftBias.length,
          triangleHidden: weights.extraStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
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

        const mainDescriptor = {
          msa: new Float32Array(0), pair: new Float32Array(0), msaMask: new Float32Array(0),
          pairMask: new Float32Array(0), sequences: features.msaSequences, length, cM: 256, cZ: 128,
          cOuter: weights.mainStack[0]!.outerProductMean.leftBias.length,
          triangleHidden: weights.mainStack[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
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
          }, embedding.msa, embedding.pairWithoutTemplates, msaMask, pairMaskTensor);
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
          embedding.msa.allocation.buffer, 0, msaFirstRowTensor.allocation.buffer, 0, length * 256 * 4,
        );
        const pairReadback = execution.createReadback(
          `monomer.pair-readback-${recycle}`, embedding.pairWithoutTemplates, readbackEncoder,
        );
        this.device.pushErrorScope("validation");
        await submit(readbackEncoder, `readback recycle ${recycle}`);
        const [msaFirstRow, pair] = await Promise.all([
          execution.mapFloat32(msaFirstRowTensor), execution.mapFloat32(pairReadback),
        ]);
        releaseTensor(msaFirstRowTensor); releaseTensor(pairReadback); releaseTensor(msaMask);

        const structure = await new StructureModuleGpu(this.device).run({
          msaFirstRow, pair, mask: features.seqMask, aatype: features.aatype,
          atom37ToAtom14: features.atom37ToAtom14, atom37Mask: features.atom37Mask,
          length, weights: weights.structure, geometry: weights.geometry,
        });
        const confidence = await new ConfidenceHeadsGpu(this.device).run(
          structure.finalRepresentation, pair, length, weights.lddt, weights.pae, paeBreaks,
        );
        const trunkSubmissions = {
          embedding: 1, extraMsa: extraSubmissions, mainEvoformer: mainSubmissions, readback: 1,
          total: 2 + extraSubmissions + mainSubmissions,
        };
        const gpuProfile = extraProfile === undefined || mainProfile === undefined
          ? undefined : { extraMsa: extraProfile, mainEvoformer: mainProfile };
        const recycleResult = { msaFirstRow, pair, structure, confidence,
          elapsedMilliseconds: performance.now() - recycleStart, trunkSubmissions,
          ...(gpuProfile === undefined ? {} : { gpuProfile }),
        };
        results.push(recycleResult);
        onRecycle?.(recycleResult, recycle);
        previousMsa = embedding.msa;
        previousPair = embedding.pairWithoutTemplates;
        previousPositions = execution.upload(`monomer.recycle-positions-${recycle}`, structure.atom37);
      }
      return {
        recycles: results, final: results[results.length - 1]!, elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
      };
    } finally {
      execution.release();
    }
  }
}
