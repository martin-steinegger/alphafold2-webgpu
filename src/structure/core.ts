import type { AllocatedGpuBuffer } from "../runtime/allocator.js";
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
    if (iterations === 0) return { activations, affine, elapsedMilliseconds: performance.now() - start };
    const ipa = new InvariantPointAttentionGpu(this.device);
    const post = new StructurePostAttentionGpu(this.device);
    const geometry = {
      activations, affine,
      pair: input.pair,
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
        const attended = ipa.encode(pass, geometry, prepared, activationTensor, affineTensor);
        const updated = post.encode(pass, input, preparedPost, activationTensor, attended.output, affineTensor);
        scratch.push(...attended.scratch, attended.output, ...updated.scratch);
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
    }
    return { activations, affine, elapsedMilliseconds: performance.now() - start };
  }
}
