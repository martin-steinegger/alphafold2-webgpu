import type { ActivationStorage } from "../runtime/storage.js";
import {
  GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot,
} from "../runtime/allocator.js";
import {
  InvariantPointAttentionGpu,
  type InvariantPointAttentionWeights,
} from "./ipa.js";
import {
  StructurePostAttentionGpu,
  type StructurePostAttentionWeights,
} from "./iteration.js";

export interface StructureCoreInput {
  readonly activations: Float32Array;
  readonly pair: Float32Array;
  readonly pairBuffer?: GPUBuffer;
  /** Storage of that buffer; `f16` means packed half-precision words. */
  readonly pairStorage?: ActivationStorage;
  /** Bytes one binding may cover of the pair; tests lower it. */
  readonly bindingLimitBytes?: number;
  readonly mask: Float32Array;
  readonly affine: Float32Array;
  readonly length: number;
  readonly channels: number;
  readonly pairChannels: number;
  readonly iterations?: number;
  readonly multimer?: boolean;
  readonly ipaWeights: InvariantPointAttentionWeights;
  readonly postAttentionWeights: StructurePostAttentionWeights;
}

export interface StructureCoreResult {
  readonly activations: Float32Array;
  readonly affine: Float32Array;
  readonly elapsedMilliseconds: number;
  readonly memory: AllocationSnapshot;
}

export class StructureCoreGpu {
  readonly device: GPUDevice;
  constructor(device: GPUDevice) { this.device = device; }

  async run(input: StructureCoreInput): Promise<StructureCoreResult> {
    let activations = input.activations;
    let affine = input.affine;
    const start = performance.now();
    const iterations = input.iterations ?? 8;
    if (!Number.isSafeInteger(iterations) || iterations < 0) {
      throw new RangeError("structure iterations must be a non-negative integer");
    }
    if (iterations === 0) {
      const allocator = new GpuBufferAllocator(this.device);
      return { activations, affine, elapsedMilliseconds: performance.now() - start, memory: allocator.snapshot() };
    }
    // IPA and the post-attention transition have complementary lifetimes.
    // Sharing a pool lets later dispatches reuse buffers after their final
    // encoded use instead of retaining eight complete iteration workspaces.
    const allocator = new GpuBufferAllocator(this.device, true);
    const ipa = new InvariantPointAttentionGpu(this.device, allocator);
    const post = new StructurePostAttentionGpu(this.device, allocator);
    const geometry = {
      activations, affine,
      pair: input.pair,
      ...(input.pairBuffer === undefined ? {} : { pairBuffer: input.pairBuffer }),
      ...(input.pairStorage === undefined ? {} : { pairStorage: input.pairStorage }),
      ...(input.bindingLimitBytes === undefined ? {} : { bindingLimitBytes: input.bindingLimitBytes }),
      mask: input.mask,
      length: input.length,
      channels: input.channels,
      pairChannels: input.pairChannels,
      heads: 12,
      scalarQk: 16,
      scalarV: 16,
      pointQk: 4,
      pointV: 8,
      ...(input.multimer === undefined ? {} : { multimer: input.multimer }),
      weights: input.ipaWeights,
    } as const;
    const postDescriptor = {
      activations, attentionUpdate: activations, affine,
      length: input.length, channels: input.channels, weights: input.postAttentionWeights,
    } as const;

    // Every iteration reads the previous one's activations and frame, so the
    // whole loop is encoded into one command buffer with the intermediates left
    // on the device. Reading them back per iteration cost two submissions, two
    // fences and two mapped readbacks each time, which dominated the structure
    // module at short chain lengths.
    const prepared = await ipa.prepare(geometry);
    const preparedPost = await post.prepare(postDescriptor);
    const scratch: AllocatedGpuBuffer[] = [];
    try {
      let activationTensor = ipa.allocator.upload("structure-core.activations", activations,
        GPUBufferUsage.STORAGE);
      let affineTensor = ipa.allocator.upload("structure-core.affine", affine, GPUBufferUsage.STORAGE);
      scratch.push(activationTensor, affineTensor);
      const encoder = this.device.createCommandEncoder({ label: "structure-core" });
      this.device.pushErrorScope("validation");
      const pass = encoder.beginComputePass({ label: "structure-core" });
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        const previousActivation = activationTensor;
        const previousAffine = affineTensor;
        const attended = ipa.encode(pass, geometry, prepared, activationTensor, affineTensor);
        const updated = post.encode(pass, input, preparedPost, activationTensor, attended.output, affineTensor);
        scratch.push(...attended.scratch, attended.output, ...updated.scratch);
        // These buffers have all had their final uses encoded. Returning them
        // to the shared pool is safe even before submission because later
        // dispatches in this pass execute in order.
        for (const buffer of attended.scratch) buffer.release();
        for (const buffer of updated.scratch) buffer.release();
        attended.output.release();
        previousActivation.release();
        previousAffine.release();
        activationTensor = updated.activations;
        affineTensor = updated.affine;
        scratch.push(activationTensor, affineTensor);
      }
      pass.end();
      const activationReadback = ipa.allocator.allocate("structure-core.activation-readback",
        activationTensor.byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      const affineReadback = ipa.allocator.allocate("structure-core.affine-readback",
        affineTensor.byteLength, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
      scratch.push(activationReadback, affineReadback);
      encoder.copyBufferToBuffer(activationTensor.buffer, 0, activationReadback.buffer, 0,
        activationTensor.byteLength);
      encoder.copyBufferToBuffer(affineTensor.buffer, 0, affineReadback.buffer, 0, affineTensor.byteLength);
      this.device.queue.submit([encoder.finish()]);
      allocator.trimPooled();
      const error = await this.device.popErrorScope();
      if (error !== null) throw new Error(`WebGPU structure core failed: ${error.message}`);
      await Promise.all([
        activationReadback.buffer.mapAsync(GPUMapMode.READ),
        affineReadback.buffer.mapAsync(GPUMapMode.READ),
      ]);
      activations = new Float32Array(activationReadback.buffer.getMappedRange().slice(0));
      affine = new Float32Array(affineReadback.buffer.getMappedRange().slice(0));
      activationReadback.buffer.unmap();
      affineReadback.buffer.unmap();
    } finally {
      for (let index = scratch.length - 1; index >= 0; index -= 1) scratch[index]!.release();
      preparedPost.release();
      prepared.release();
      allocator.destroyPooled();
    }
    return { activations, affine, elapsedMilliseconds: performance.now() - start, memory: allocator.snapshot() };
  }
}
