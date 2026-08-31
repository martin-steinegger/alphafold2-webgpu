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
      weights: input.ipaWeights,
    } as const;
    const prepared = await ipa.prepare(geometry);
    try {
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        const attention = await ipa.run({
          ...geometry, activations, affine, prepared,
        });
        const update = await post.run({
          activations,
          attentionUpdate: attention.output,
          affine,
          length: input.length,
          channels: input.channels,
          weights: input.postAttentionWeights,
        });
        activations = update.activations;
        affine = update.affine;
      }
    } finally {
      prepared.release();
    }
    return { activations, affine, elapsedMilliseconds: performance.now() - start };
  }
}
