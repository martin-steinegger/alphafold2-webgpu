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
export function estimateMonomerMemory(
  length: number,
  msaSequences: number,
  extraSequences: number,
  transitionMode: "full" | "chunked",
): MonomerMemoryEstimate {
  if (![length, msaSequences, extraSequences]
    .every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("monomer memory dimensions must be positive safe integers");
  }
  const bytes = Float32Array.BYTES_PER_ELEMENT;
  const pair = checkedBytes("pair representation", length, length, 128, bytes);
  const msa = checkedBytes("MSA representation", msaSequences, length, 256, bytes);
  const extra = checkedBytes("extra-MSA representation", extraSequences, length, 64, bytes);
  const masks = checkedBytes("model masks", length, msaSequences + extraSequences + length, bytes);
  const positions = checkedBytes("atom positions", length, 37, 3, bytes);

  // Previous/current recycle representations coexist during embedding. The
  // multipliers include residual outputs and pooled buffers retained for reuse.
  const persistentBytes = checkedBytes("persistent activation estimate", 1,
    4 * pair + 4 * msa + 3 * extra + masks + 2 * positions);
  const transitionRows = transitionMode === "full"
    ? Math.max(
      checkedBytes("main transition", msaSequences, length, 1024, bytes),
      checkedBytes("extra transition", extraSequences, length, 256, bytes),
      checkedBytes("pair transition", length, length, 512, bytes),
    )
    : 96 * 1024 ** 2;
  const transitionNormalized = transitionMode === "full"
    ? Math.max(msa, extra, pair)
    : Math.ceil(transitionRows / 4);
  const attentionScratch = Math.max(5 * msa, 5 * extra, 6 * pair);
  const outerTileSequences = Math.min(32, Math.max(msaSequences, extraSequences));
  const outerProductScratch = checkedBytes(
    "outer-product scratch", outerTileSequences, length, 32, 128, bytes,
  ) + checkedBytes("outer-product projections", 2, Math.max(msaSequences, extraSequences), length, 32, bytes)
    + pair;
  const operatorScratch = Math.max(
    transitionRows + transitionNormalized,
    attentionScratch,
    outerProductScratch,
  );
  // Pipeline parameters, readbacks, alignment padding, and implementation
  // bookkeeping are deliberately represented by a fixed safety allowance.
  const scratchBytes = operatorScratch + pair + msaSequences * length * 256 * bytes + 64 * 1024 ** 2;
  // Exact-sized pooling retains buffers needed by successive block shapes.
  // GB10 full-model qualification measured a 2.31x resident/logical-model
  // ratio at L=59, so use 2.5x as a conservative implementation allowance.
  const estimatedPeakBytes = Math.ceil((persistentBytes + scratchBytes) * 2.5);
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
  const outerProductTileSequences = Math.min(32, Math.max(msaSequences, extraSequences));
  const largestTensor = Math.max(
    msaSequences * length * 256 * bytes,
    extraSequences * length * 64 * bytes,
    length * length * 128 * bytes,
    outerProductTileSequences * length * 32 * 128 * bytes,
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
  if (fast.maxBufferSize <= adapter.limits.maxBufferSize
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
  return adapter.requestDevice({
    requiredFeatures,
    requiredLimits: { maxBufferSize, maxStorageBufferBindingSize },
  });
}
