import {
  encodeEvoformerPairBlock,
  encodeExtraMsaBlock,
  encodeEvoformerBlock,
  type EvoformerBlockInput,
  type EvoformerBlockWeights,
  type EvoformerPairBlockWeights,
  type ExtraMsaBlockWeights,
} from "./block.js";
import { WebGpuExecution, type GpuTimestampEntry } from "../runtime/execution.js";
import type { AllocationSnapshot } from "../runtime/allocator.js";

export interface EvoformerStackInput extends Omit<EvoformerBlockInput, "weights"> {
  readonly blockWeights: readonly EvoformerBlockWeights[];
  /** Optional zero-based block to profile with GPU timestamp queries. */
  readonly profileBlock?: number;
  /** Number of submitted blocks allowed in flight before temporary buffers are reclaimed. */
  readonly submissionWindow?: number;
}

export interface EvoformerStackResult {
  readonly msa: Float32Array;
  readonly pair: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
  readonly blocks: number;
  readonly timestampProfile?: readonly GpuTimestampEntry[];
}

export interface ExtraMsaPairStackInput {
  readonly msa: Float32Array;
  readonly pair: Float32Array;
  readonly msaMask: Float32Array;
  readonly pairMask: Float32Array;
  readonly sequences: number;
  readonly length: number;
  readonly cM: number;
  readonly cZ: number;
  readonly cOuter: number;
  readonly triangleHidden: number;
  readonly blockWeights: readonly EvoformerPairBlockWeights[];
}

export interface ExtraMsaStackInput extends Omit<ExtraMsaPairStackInput, "blockWeights"> {
  readonly blockWeights: readonly ExtraMsaBlockWeights[];
}

export class EvoformerStackGpu {
  readonly device: GPUDevice;

  constructor(device: GPUDevice) { this.device = device; }

  async run(input: EvoformerStackInput): Promise<EvoformerStackResult> {
    if (input.blockWeights.length === 0) throw new RangeError("Evoformer stack requires at least one block");
    const execution = new WebGpuExecution(this.device);
    try {
      const msaElements = input.sequences * input.length * input.cM;
      const pairElements = input.length * input.length * input.cZ;
      if (input.msa.length !== msaElements || input.pair.length !== pairElements) {
        throw new RangeError("Evoformer stack activation shape mismatch");
      }
      const msa = execution.upload("stack.msa", input.msa, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pair = execution.upload("stack.pair", input.pair, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaMask = execution.upload("stack.msa-mask", input.msaMask);
      const pairMask = execution.upload("stack.pair-mask", input.pairMask);
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();
      let timestampProfile: readonly GpuTimestampEntry[] | undefined;
      const requestedWindow = input.submissionWindow ?? input.blockWeights.length;
      if (!Number.isSafeInteger(requestedWindow) || requestedWindow < 1) {
        throw new RangeError("submissionWindow must be a positive safe integer");
      }
      const submissionWindow = input.profileBlock === undefined ? requestedWindow : 1;

      for (let block = 0; block < input.blockWeights.length; block += 1) {
        const encoder = this.device.createCommandEncoder({ label: `evoformer-stack.block-${block}` });
        const profiling = input.profileBlock === block;
        if (profiling) execution.beginTimestampProfile();
        this.device.pushErrorScope("validation");
        await encodeEvoformerBlock(execution, encoder, {
          ...input,
          weights: input.blockWeights[block]!,
        }, msa, pair, msaMask, pairMask);
        execution.endComputePass(encoder);
        const pendingProfile = profiling ? execution.finishTimestampProfile(encoder) : undefined;
        this.device.queue.submit([encoder.finish()]);
        const validationError = await this.device.popErrorScope();
        if (validationError !== null) {
          throw new Error(`WebGPU block ${block} validation failed: ${validationError.message}`);
        }
        const endOfWindow = (block + 1) % submissionWindow === 0 || block + 1 === input.blockWeights.length;
        if (pendingProfile !== undefined) {
          await this.device.queue.onSubmittedWorkDone();
          timestampProfile = await execution.readTimestampProfile(pendingProfile);
          execution.releaseSince(persistentCheckpoint);
        } else {
          // Pooling makes these buffers available to the next encoded block.
          // Queue ordering ensures its commands execute only after this block.
          execution.releaseSince(persistentCheckpoint);
          if (endOfWindow) await this.device.queue.onSubmittedWorkDone();
        }
      }

      const encoder = this.device.createCommandEncoder({ label: "evoformer-stack.readback" });
      const msaReadback = execution.createReadback("stack.msa-readback", msa, encoder);
      const pairReadback = execution.createReadback("stack.pair-readback", pair, encoder);
      this.device.queue.submit([encoder.finish()]);
      const [msaOutput, pairOutput] = await Promise.all([
        execution.mapFloat32(msaReadback), execution.mapFloat32(pairReadback),
      ]);
      return {
        msa: msaOutput,
        pair: pairOutput,
        elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot(),
        blocks: input.blockWeights.length,
        ...(timestampProfile === undefined ? {} : { timestampProfile }),
      };
    } finally {
      execution.release();
    }
  }
}

export class ExtraMsaPairStackGpu {
  readonly device: GPUDevice;

  constructor(device: GPUDevice) { this.device = device; }

  async run(input: ExtraMsaPairStackInput): Promise<{
    readonly pair: Float32Array;
    readonly elapsedMilliseconds: number;
    readonly memory: AllocationSnapshot;
  }> {
    const execution = new WebGpuExecution(this.device);
    try {
      const msa = execution.upload("extra-stack.msa", input.msa);
      const pair = execution.upload(
        "extra-stack.pair", input.pair, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      );
      const msaMask = execution.upload("extra-stack.msa-mask", input.msaMask);
      const pairMask = execution.upload("extra-stack.pair-mask", input.pairMask);
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();
      for (let block = 0; block < input.blockWeights.length; block += 1) {
        const encoder = this.device.createCommandEncoder({ label: `extra-msa-pair-stack.block-${block}` });
        this.device.pushErrorScope("validation");
        await encodeEvoformerPairBlock(
          execution, encoder, input, input.blockWeights[block]!, msa, pair, msaMask, pairMask,
        );
        execution.endComputePass(encoder);
        this.device.queue.submit([encoder.finish()]);
        const validationError = await this.device.popErrorScope();
        if (validationError !== null) {
          throw new Error(`WebGPU extra-MSA block ${block} validation failed: ${validationError.message}`);
        }
        // Commands are queue ordered, so the next block may alias these
        // pooled scratch buffers without a host-side wait.
        execution.releaseSince(persistentCheckpoint);
      }
      const encoder = this.device.createCommandEncoder({ label: "extra-msa-pair-stack.readback" });
      const readback = execution.createReadback("extra-stack.pair-readback", pair, encoder);
      this.device.queue.submit([encoder.finish()]);
      const output = await execution.mapFloat32(readback);
      return { pair: output, elapsedMilliseconds: performance.now() - start, memory: execution.snapshot() };
    } finally {
      execution.release();
    }
  }
}

export class ExtraMsaStackGpu {
  readonly device: GPUDevice;
  constructor(device: GPUDevice) { this.device = device; }
  async run(input: ExtraMsaStackInput): Promise<{
    readonly msa: Float32Array;
    readonly pair: Float32Array;
    readonly elapsedMilliseconds: number;
    readonly memory: AllocationSnapshot;
  }> {
    const execution = new WebGpuExecution(this.device);
    try {
      const msa = execution.upload("extra-full-stack.msa", input.msa, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const pair = execution.upload("extra-full-stack.pair", input.pair, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
      const msaMask = execution.upload("extra-full-stack.msa-mask", input.msaMask);
      const pairMask = execution.upload("extra-full-stack.pair-mask", input.pairMask);
      const persistentCheckpoint = execution.checkpoint();
      const start = performance.now();
      for (let block = 0; block < input.blockWeights.length; block += 1) {
        const encoder = this.device.createCommandEncoder({ label: `extra-msa-stack.block-${block}` });
        this.device.pushErrorScope("validation");
        await encodeExtraMsaBlock(execution, encoder, input, input.blockWeights[block]!, msa, pair, msaMask, pairMask);
        execution.endComputePass(encoder);
        this.device.queue.submit([encoder.finish()]);
        const validationError = await this.device.popErrorScope();
        if (validationError !== null) throw new Error(`WebGPU extra-MSA block ${block} failed: ${validationError.message}`);
        execution.releaseSince(persistentCheckpoint);
      }
      const encoder = this.device.createCommandEncoder({ label: "extra-full-stack.readback" });
      const msaReadback = execution.createReadback("extra-full-stack.msa-readback", msa, encoder);
      const pairReadback = execution.createReadback("extra-full-stack.pair-readback", pair, encoder);
      this.device.queue.submit([encoder.finish()]);
      const [msaOutput, pairOutput] = await Promise.all([
        execution.mapFloat32(msaReadback), execution.mapFloat32(pairReadback),
      ]);
      return { msa: msaOutput, pair: pairOutput, elapsedMilliseconds: performance.now() - start,
        memory: execution.snapshot() };
    } finally { execution.release(); }
  }
}
