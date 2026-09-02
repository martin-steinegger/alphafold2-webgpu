import { COMPACT_GPU_POOL_BYTES } from "./allocator.js";
import {
  OUTER_PRODUCT_BLOCK_LIMIT_BYTES, outerProductMeanNormalizeWindow, outerProductMeanRowBlock,
} from "../evoformer/outer-product-mean.js";
import { ATTENTION_WINDOW_TARGET_BYTES, attentionBatchWindow } from "../evoformer/attention.js";
import { triangleBlockRows } from "../evoformer/block.js";
import type { TriangleWholeStorage } from "../triangle/shaders.js";
import type { ActivationStorage } from "./storage.js";
import { TRANSITION_CHUNK_TARGET_BYTES, transitionChunkRows } from "../evoformer/transition.js";
import { recordSubgroupRange } from "./subgroups.js";

const WEBGPU_BASE_MAX_BUFFER_SIZE = 256 * 1024 * 1024;
const WEBGPU_BASE_MAX_STORAGE_BINDING_SIZE = 128 * 1024 * 1024;

function nextLimitTier(value: number, baseline: number): number {
  let tier = baseline;
  while (tier < value) tier *= 2;
  return tier;
}

export interface AlphaFoldDeviceRequirements {
  readonly maxBufferSize: number;
  readonly maxStorageBufferBindingSize: number;
}

export interface AlphaFoldDevicePlan {
  readonly requirements: AlphaFoldDeviceRequirements;
  readonly transitionMode: "full" | "chunked";
  readonly memory: MonomerMemoryEstimate;
}

export interface MonomerMemoryEstimate {
  readonly persistentBytes: number;
  readonly scratchBytes: number;
  readonly estimatedPeakBytes: number;
}

export interface MonomerRowSuggestion {
  readonly msaSequences: number;
  readonly extraSequences: number;
  readonly estimatedPeakBytes: number;
}

function checkedBytes(label: string, ...factors: number[]): number {
  const value = factors.reduce((product, factor) => product * factor, 1);
  if (!Number.isSafeInteger(value) || value < 0) throw new RangeError(`${label} exceeds JavaScript precision`);
  return value;
}

/**
 * Conservative aggregate allocation estimate for one monomer recycle.
 *
 * WebGPU intentionally does not expose free VRAM. This estimate therefore
 * models simultaneously resident activations and the largest operator scratch
 * set; it is a preflight guard, not a claim about physical memory availability.
 */
export interface MonomerMemoryOptions {
  /** Storage of the triangle multiplication's whole projection; `f16` halves it. */
  readonly triangleWholeStorage?: TriangleWholeStorage;
  /** Storage of the MSA activations; `f16` halves them. */
  readonly msaStorage?: ActivationStorage;
}

export function estimateMonomerMemory(
  length: number,
  msaSequences: number,
  extraSequences: number,
  transitionMode: "full" | "chunked",
  options: MonomerMemoryOptions = {},
): MonomerMemoryEstimate {
  if (![length, msaSequences, extraSequences]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("monomer memory dimensions must be positive safe integers");
  }
  void transitionMode;
  const bytes = Float32Array.BYTES_PER_ELEMENT;
  const pair = checkedBytes("pair representation", length, length, 128, bytes);
  const activationBytes = options.msaStorage === "f16" ? 2 : bytes;
  const msa = checkedBytes("MSA representation", msaSequences, length, 256, activationBytes);
  const extra = checkedBytes("extra-MSA representation", extraSequences, length, 64, activationBytes);
  const masks = checkedBytes("model masks", length, msaSequences + extraSequences + length, bytes);
  const positions = checkedBytes("atom positions", length, 37, 3, bytes);

  const msaFeatures = checkedBytes("MSA features", msaSequences, length, 49, bytes);

  // Only the pair and the masks are live in every phase. The clustered MSA is
  // embedded after the extra stack and released after the main stack, the
  // extra MSA is released before the clustered one exists, and the template
  // update is uploaded per recycle and released with the embedder's command
  // buffer, so the trunk carries the pair plus the larger alignment.
  const persistentBytes = checkedBytes("persistent activation estimate", 1,
    pair + Math.max(msa, extra) + masks + 2 * positions);
  // The embedder writes the new pair over the recycled one and adds the
  // template update while the extra alignment and the MSA features are live.
  const embedderBytes = 2 * pair + extra + msaFeatures + masks + 2 * positions;

  // Every operation bounds its own scratch against an explicit budget, so the
  // peak is the largest single operation's working set, not a sum over the
  // block. Each term below mirrors what that operation allocates.
  const attentionScratch = (batch: number, queries: number, channels: number, tensors: number): number =>
    attentionBatchWindow(batch, queries, channels) * queries * channels * bytes * tensors;
  const transitionScratch = (rows: number, channels: number, hidden: number): number => {
    const chunk = transitionChunkRows(rows, channels, hidden, TRANSITION_CHUNK_TARGET_BYTES);
    return chunk * (channels + hidden) * bytes;
  };
  const outerProductScratch = (sequences: number, channels: number, outer: number): number =>
    outerProductMeanRowBlock(length, outer) * length * outer * outer * bytes
    + outerProductMeanNormalizeWindow(sequences * length, channels) * channels * bytes
    + 2 * sequences * length * outer * bytes
    + length * length * bytes;
  // The hidden width matches the pair width in every released AlphaFold model.
  const triangleBlock = triangleBlockRows(length, 128, 128) * length * 128 * bytes;

  // Attention over the pair adds a bias of one value per head and query pair,
  // built from the pair one row window at a time.
  const pairBias = 8 * length * length * bytes + Math.min(pair, ATTENTION_WINDOW_TARGET_BYTES);
  const operatorScratch = Math.max(
    // Row and column attention over the clustered and extra alignments: the
    // normalized input, query, key, value and gate, one batch window each.
    attentionScratch(msaSequences, length, 256, 5) + pairBias,
    attentionScratch(length, msaSequences, 256, 5),
    attentionScratch(extraSequences, length, 64, 5) + pairBias,
    attentionScratch(length, length, 128, 5) + pairBias,
    // Triangle multiplication keeps one projection whole and streams the
    // other projection, the contraction and the output gate in blocks.
    (options.triangleWholeStorage === "f16" ? pair / 2 : pair) + 3 * triangleBlock,
    transitionScratch(msaSequences * length, 256, 1024),
    transitionScratch(extraSequences * length, 64, 256),
    transitionScratch(length * length, 128, 512),
    outerProductScratch(msaSequences, 256, 32),
    outerProductScratch(extraSequences, 64, 32),
  );
  // Readbacks, uniforms and allocation padding, none of which scale with the
  // shape, plus headroom for the operator this model does not enumerate.
  const scratchBytes = Math.ceil(operatorScratch * 1.15) + 16 * 1024 ** 2;
  const estimatedPeakBytes = Math.max(persistentBytes + scratchBytes, embedderBytes + 16 * 1024 ** 2);
  if (![persistentBytes, scratchBytes, estimatedPeakBytes].every(Number.isSafeInteger)) {
    throw new RangeError("monomer aggregate memory estimate exceeds JavaScript precision");
  }
  return { persistentBytes, scratchBytes, estimatedPeakBytes };
}

/** Reduces extra rows first, then clustered rows, to fit an explicit budget. */
export function suggestMonomerRows(
  length: number,
  msaSequences: number,
  extraSequences: number,
  transitionMode: "full" | "chunked",
  budgetBytes: number,
): MonomerRowSuggestion | undefined {
  if (!Number.isSafeInteger(budgetBytes) || budgetBytes <= 0) throw new RangeError("memory budget must be positive");
  const fits = (msa: number, extra: number): boolean =>
    estimateMonomerMemory(length, msa, extra, transitionMode).estimatedPeakBytes <= budgetBytes;
  if (fits(msaSequences, extraSequences)) {
    return { msaSequences, extraSequences,
      estimatedPeakBytes: estimateMonomerMemory(length, msaSequences, extraSequences, transitionMode).estimatedPeakBytes };
  }
  const maximize = (maximum: number, predicate: (value: number) => boolean): number => {
    let low = 1; let high = maximum; let best = 0;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      if (predicate(middle)) { best = middle; low = middle + 1; } else high = middle - 1;
    }
    return best;
  };
  const extra = maximize(extraSequences, (value) => fits(msaSequences, value));
  if (extra > 0) {
    return { msaSequences, extraSequences: extra,
      estimatedPeakBytes: estimateMonomerMemory(length, msaSequences, extra, transitionMode).estimatedPeakBytes };
  }
  const msa = maximize(msaSequences, (value) => fits(value, 1));
  if (msa === 0) return undefined;
  return { msaSequences: msa, extraSequences: 1,
    estimatedPeakBytes: estimateMonomerMemory(length, msa, 1, transitionMode).estimatedPeakBytes };
}

/** Largest persistent monomer tensor; temporary transition tensors are processed in bounded chunks. */
export function monomerDeviceRequirements(
  length: number,
  msaSequences: number,
  extraSequences: number,
): AlphaFoldDeviceRequirements {
  if (![length, msaSequences, extraSequences]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("monomer device dimensions must be positive safe integers");
  }
  const bytes = Float32Array.BYTES_PER_ELEMENT;
  // The persistent activations, plus the largest scratch tensor, which every
  // operation now bounds against its own budget rather than letting it grow
  // with the shape: the outer-product contraction, the transition window and
  // the attention window are all capped.
  const largestTensor = Math.max(
    msaSequences * length * 256 * bytes,
    extraSequences * length * 64 * bytes,
    length * length * 128 * bytes,
    OUTER_PRODUCT_BLOCK_LIMIT_BYTES,
  );
  if (!Number.isSafeInteger(largestTensor)) throw new RangeError("monomer tensor size exceeds JavaScript precision");
  return {
    maxBufferSize: Math.max(WEBGPU_BASE_MAX_BUFFER_SIZE, largestTensor),
    maxStorageBufferBindingSize: Math.max(WEBGPU_BASE_MAX_STORAGE_BINDING_SIZE, largestTensor),
  };
}

/** Chooses full transitions when supported, with a bounded-window fallback for constrained adapters. */
export function planMonomerDevice(
  adapter: GPUAdapter,
  length: number,
  msaSequences: number,
  extraSequences: number,
  memoryBudgetBytes?: number,
  preferCompact = false,
): AlphaFoldDevicePlan {
  if (memoryBudgetBytes !== undefined
    && (!Number.isSafeInteger(memoryBudgetBytes) || memoryBudgetBytes <= 0)) {
    throw new RangeError("memory budget must be a positive safe integer");
  }
  const compact = monomerDeviceRequirements(length, msaSequences, extraSequences);
  const bytes = Float32Array.BYTES_PER_ELEMENT;
  const largestFullTransition = Math.max(
    msaSequences * length * 1024 * bytes,
    extraSequences * length * 256 * bytes,
    length * length * 512 * bytes,
  );
  if (!Number.isSafeInteger(largestFullTransition)) {
    throw new RangeError("monomer transition size exceeds JavaScript precision");
  }
  const fast = {
    maxBufferSize: Math.max(compact.maxBufferSize, largestFullTransition),
    maxStorageBufferBindingSize: Math.max(compact.maxStorageBufferBindingSize, largestFullTransition),
  };
  const fullMemory = estimateMonomerMemory(length, msaSequences, extraSequences, "full");
  if (!preferCompact
    && fast.maxBufferSize <= adapter.limits.maxBufferSize
    && fast.maxStorageBufferBindingSize <= adapter.limits.maxStorageBufferBindingSize
    && (memoryBudgetBytes === undefined || fullMemory.estimatedPeakBytes <= memoryBudgetBytes)) {
    return { requirements: fast, transitionMode: "full",
      memory: fullMemory };
  }
  return { requirements: compact, transitionMode: "chunked",
    memory: estimateMonomerMemory(length, msaSequences, extraSequences, "chunked") };
}

/** Requests optional WebGPU features and only the buffer limits required by the selected shape. */
export async function requestAlphaFoldDevice(
  adapter: GPUAdapter,
  requirements: AlphaFoldDeviceRequirements = {
    maxBufferSize: WEBGPU_BASE_MAX_BUFFER_SIZE,
    maxStorageBufferBindingSize: WEBGPU_BASE_MAX_STORAGE_BINDING_SIZE,
  },
): Promise<GPUDevice> {
  // subgroup-size-control is shipping ahead of the current @webgpu/types union.
  const optional = ["subgroups", "subgroup-size-control", "timestamp-query"] as const;
  const requiredFeatures = optional.filter(
    (feature) => adapter.features.has(feature as GPUFeatureName),
  ) as GPUFeatureName[];
  const requiredBufferSize = requirements.maxBufferSize;
  const requiredStorageBufferBindingSize = requirements.maxStorageBufferBindingSize;
  if (![requiredBufferSize, requiredStorageBufferBindingSize]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("requested WebGPU buffer limits must be positive safe integers");
  }
  if (requiredStorageBufferBindingSize > requiredBufferSize) {
    throw new RangeError("storage binding requirement cannot exceed the buffer-size requirement");
  }
  if (requiredBufferSize > adapter.limits.maxBufferSize
    || requiredStorageBufferBindingSize > adapter.limits.maxStorageBufferBindingSize) {
    const mib = (value: number): string => `${(value / 1024 ** 2).toFixed(0)} MiB`;
    throw new RangeError(`This input requires a ${mib(requiredStorageBufferBindingSize)} storage binding and `
      + `${mib(requiredBufferSize)} buffer, but this adapter exposes `
      + `${mib(adapter.limits.maxStorageBufferBindingSize)} and ${mib(adapter.limits.maxBufferSize)}.`);
  }
  const maxStorageBufferBindingSize = Math.min(
    adapter.limits.maxStorageBufferBindingSize,
    nextLimitTier(requiredStorageBufferBindingSize, WEBGPU_BASE_MAX_STORAGE_BINDING_SIZE),
  );
  const maxBufferSize = Math.min(
    adapter.limits.maxBufferSize,
    nextLimitTier(Math.max(requiredBufferSize, maxStorageBufferBindingSize), WEBGPU_BASE_MAX_BUFFER_SIZE),
  );
  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits: { maxBufferSize, maxStorageBufferBindingSize },
  });
  recordSubgroupRange(device, adapter);
  return device;
}
