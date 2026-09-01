import { AtomGeometryGpu, type ResidueGeometryTables } from "./geometry.js";
import { StructureCoreGpu } from "./core.js";
import { StructureInitializeGpu, type StructureInitializeWeights } from "./initialize.js";
import type { InvariantPointAttentionWeights } from "./ipa.js";
import type { StructurePostAttentionWeights } from "./iteration.js";
import { SidechainAnglesGpu, type SidechainWeights } from "./sidechain.js";
import type { AllocationSnapshot } from "../runtime/allocator.js";

export interface StructureModuleWeights {
  readonly initialize: StructureInitializeWeights;
  readonly ipa: InvariantPointAttentionWeights;
  readonly postAttention: StructurePostAttentionWeights;
  readonly sidechain: SidechainWeights;
}

export interface StructureModuleInput {
  readonly msaFirstRow: Float32Array;
  readonly pair: Float32Array;
  /** Pair activation already resident on this device. */
  readonly pairBuffer?: GPUBuffer;
  readonly mask: Float32Array;
  readonly aatype: Float32Array;
  readonly atom37ToAtom14: Float32Array;
  readonly atom37Mask: Float32Array;
  readonly length: number;
  readonly msaChannels?: number;
  readonly structureChannels?: number;
  readonly pairChannels?: number;
  readonly weights: StructureModuleWeights;
  readonly geometry: ResidueGeometryTables;
  readonly multimer?: boolean;
}

export interface StructureModuleResult {
  readonly atom14: Float32Array;
  readonly atom37: Float32Array;
  readonly atom37Mask: Float32Array;
  readonly finalRepresentation: Float32Array;
  readonly affine: Float32Array;
  readonly angles: Float32Array;
  readonly unnormalizedAngles: Float32Array;
  readonly elapsedMilliseconds: number;
  /** Peak for the dominant eight-iteration structure core allocator. */
  readonly memory?: AllocationSnapshot;
}

/** Complete eight-iteration AlphaFold structure module. All learned operations and atom geometry execute in WGSL. */
export class StructureModuleGpu {
  readonly device: GPUDevice;
  constructor(device: GPUDevice) { this.device = device; }

  async run(input: StructureModuleInput): Promise<StructureModuleResult> {
    const start = performance.now();
    const msaChannels = input.msaChannels ?? 256;
    const structureChannels = input.structureChannels ?? 384;
    const pairChannels = input.pairChannels ?? 128;
    const initialized = await new StructureInitializeGpu(this.device).run(
      input.msaFirstRow, input.length, msaChannels, structureChannels, input.weights.initialize,
    );
    const core = await new StructureCoreGpu(this.device).run({
      activations: initialized.activations,
      pair: input.pair,
      ...(input.pairBuffer === undefined ? {} : { pairBuffer: input.pairBuffer }),
      mask: input.mask,
      affine: initialized.affine,
      length: input.length,
      channels: structureChannels,
      pairChannels,
      ipaWeights: input.weights.ipa,
      postAttentionWeights: input.weights.postAttention,
      ...(input.multimer === undefined ? {} : { multimer: input.multimer }),
    });
    const sidechain = await new SidechainAnglesGpu(this.device).run(
      core.activations, initialized.initialRepresentation, input.length, structureChannels, 128, input.weights.sidechain,
    );
    const geometry = await new AtomGeometryGpu(this.device).run({
      affine: core.affine,
      angles: sidechain.angles,
      aatype: input.aatype,
      atom37ToAtom14: input.atom37ToAtom14,
      atom37Mask: input.atom37Mask,
      length: input.length,
      positionScale: input.multimer === true ? 20 : 10,
      tables: input.geometry,
    });
    return {
      atom14: geometry.atom14,
      atom37: geometry.atom37,
      atom37Mask: input.atom37Mask,
      finalRepresentation: core.activations,
      affine: core.affine,
      angles: sidechain.angles,
      unnormalizedAngles: sidechain.unnormalizedAngles,
      elapsedMilliseconds: performance.now() - start,
      memory: core.memory,
    };
  }
}
