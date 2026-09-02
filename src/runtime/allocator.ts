/** One label's share of the live working set. */
export interface AllocationShare {
  readonly label: string;
  readonly bytes: number;
  readonly count: number;
}

export interface AllocationSnapshot {
  readonly currentBytes: number;
  readonly peakBytes: number;
  /**
   * What was live when the peak was reached, largest first.
   *
   * The peak is what a device has to fit, so knowing which tensors made it is
   * the difference between guessing at a reduction and choosing one.
   */
  readonly peakComposition: readonly AllocationShare[];
  /** Bytes held by live and pooled GPUBuffer objects. */
  readonly residentBytes: number;
  /** Maximum physical GPUBuffer storage retained by this allocator. */
  readonly peakResidentBytes: number;
  readonly pooledBytes: number;
  readonly allocationCount: number;
  readonly bufferCount: number;
}

// Retains more of the repeated AF2 block working set now that compact best-fit
// reuse lowers its measured resident peak, while keeping idle unified-memory
// scratch under an explicit safety bound.
export const COMPACT_GPU_POOL_BYTES = 864 * 1024 ** 2;

/**
 * A retired buffer that no request has taken over within this many submitted
 * command buffers is destroyed. Every Evoformer block submits at least once
 * and cycles through the same scratch shapes, so the scratch of the running
 * phase is always younger than this, while one-shot inputs such as the
 * embedder's features or a retired alignment do not linger through the trunk.
 */
export const POOL_IDLE_GENERATIONS = 4;

/**
 * Requests of at least this size are backed by a whole number of mebibytes.
 * Operations size their scratch from byte budgets and land a few percent
 * apart (an attention window at 15.8 MiB, a transition chunk at 16.0), and
 * a request can only reuse a buffer at least as large, so without a common
 * granularity every operation kept its own set of nearly equal chunks.
 */
export const POOL_GRANULARITY_BYTES = 1024 ** 2;

/**
 * Resident GPU memory accounting shared by every allocator on a device, so one
 * budget covers the trunk, the template module, the structure module and the
 * confidence heads together. On unified-memory machines an allocation that the
 * driver accepts can still push the system into paging; refusing it here turns
 * a frozen machine into an error naming the tensor.
 */
interface DeviceMemoryTally { resident: number; budget: number; }
const deviceTallies = new WeakMap<GPUDevice, DeviceMemoryTally>();
function deviceTally(device: GPUDevice): DeviceMemoryTally {
  let tally = deviceTallies.get(device);
  if (tally === undefined) { tally = { resident: 0, budget: Number.POSITIVE_INFINITY }; deviceTallies.set(device, tally); }
  return tally;
}

/** Cap the resident GPU bytes every allocator on `device` may hold together; `undefined` removes the cap. */
export function setGpuMemoryBudget(device: GPUDevice, bytes: number | undefined): void {
  if (bytes !== undefined && (!Number.isSafeInteger(bytes) || bytes <= 0)) {
    throw new RangeError("GPU memory budget must be a positive safe integer");
  }
  deviceTally(device).budget = bytes ?? Number.POSITIVE_INFINITY;
}

/** Resident GPU bytes currently held by all allocators on `device`. */
export function gpuMemoryResident(device: GPUDevice): number { return deviceTally(device).resident; }

export class GpuMemoryBudgetError extends Error {
  readonly label: string;
  readonly requestedBytes: number;
  readonly residentBytes: number;
  readonly budgetBytes: number;
  constructor(label: string, requestedBytes: number, residentBytes: number, budgetBytes: number) {
    const mib = (value: number): string => (value / 1024 ** 2).toFixed(0);
    super(`GPU memory budget exceeded: allocating ${mib(requestedBytes)} MiB for ${label} would bring resident `
      + `GPU memory to ${mib(residentBytes + requestedBytes)} MiB, over the ${mib(budgetBytes)} MiB budget. `
      + "Reduce the MSA rows, the sequence length, or the number of chains.");
    this.name = "GpuMemoryBudgetError";
    this.label = label; this.requestedBytes = requestedBytes;
    this.residentBytes = residentBytes; this.budgetBytes = budgetBytes;
  }
}

interface PooledGpuBuffer {
  readonly buffer: GPUBuffer;
  /** Physical size of the reusable GPUBuffer. */
  readonly byteLength: number;
  readonly usage: GPUBufferUsageFlags;
  readonly key: string;
  /** Submission generation in which the buffer was retired. */
  readonly generation: number;
}

export interface AllocateOptions {
  /**
   * Only reuse buffers retired before the last submitted boundary. Queue
   * writes execute ahead of any command buffer still being encoded, so an
   * upload must never take over a buffer those pending commands still read.
   */
  readonly requireSubmitted?: boolean;
}

/** Labels differing only by a block or recycle index share one group. */
const allocationGroup = (label: string): string => (label === "" ? "unlabelled" : label).replace(/-?\d+$/, "");

export class AllocatedGpuBuffer {
  readonly buffer: GPUBuffer;
  /** Logical range requested by the tensor using this buffer. */
  readonly byteLength: number;
  /** Usage the tensor asked for; the physical buffer may carry more. */
  readonly usage: GPUBufferUsageFlags;
  readonly #allocationByteLength: number;
  readonly #allocationUsage: GPUBufferUsageFlags;
  /**
   * The label this allocation was made under.
   *
   * A pooled buffer keeps the GPUBuffer label of whoever created it, so the
   * live-composition accounting has to remember who is holding it now.
   */
  readonly #label: string;
  #allocator: GpuBufferAllocator | undefined;

  constructor(allocator: GpuBufferAllocator, buffer: GPUBuffer, byteLength: number,
    allocationByteLength: number, usage: GPUBufferUsageFlags, allocationUsage: GPUBufferUsageFlags = usage,
    label = "") {
    this.#allocator = allocator;
    this.buffer = buffer;
    this.byteLength = byteLength;
    this.#allocationByteLength = allocationByteLength;
    this.usage = usage;
    this.#allocationUsage = allocationUsage;
    this.#label = label;
  }

  release(): void {
    const allocator = this.#allocator;
    if (allocator === undefined) return;
    this.#allocator = undefined;
    allocator.noteRelease(this.buffer, this.byteLength, this.#allocationByteLength, this.#allocationUsage,
      this.#label);
  }
}

export class GpuBufferAllocator {
  readonly device: GPUDevice;
  #currentBytes = 0;
  #peakBytes = 0;
  /** Live bytes by label, and the copy of it taken when the peak was last raised. */
  readonly #liveByLabel = new Map<string, { bytes: number; count: number }>();
  #peakComposition: readonly AllocationShare[] = [];
  #residentBytes = 0;
  #peakResidentBytes = 0;
  #allocationCount = 0;
  #bufferCount = 0;
  #pooledBytes = 0;
  readonly #pooling: boolean;
  readonly #maxPooledBytes: number;
  readonly #pool = new Map<string, PooledGpuBuffer[]>();
  /** Oldest-to-newest insertion order for bounded eviction. */
  readonly #pooledLru = new Set<PooledGpuBuffer>();
  #generation = 0;

  constructor(device: GPUDevice, pooling = false, maxPooledBytes = Number.POSITIVE_INFINITY) {
    if (maxPooledBytes !== Number.POSITIVE_INFINITY
      && (!Number.isSafeInteger(maxPooledBytes) || maxPooledBytes < 0)) {
      throw new RangeError("maxPooledBytes must be a non-negative safe integer or Infinity");
    }
    this.device = device;
    this.#pooling = pooling;
    this.#maxPooledBytes = maxPooledBytes;
  }

  allocate(
    label: string, requestedBytes: number, usage: GPUBufferUsageFlags, options: AllocateOptions = {},
  ): AllocatedGpuBuffer {
    if (!Number.isSafeInteger(requestedBytes) || requestedBytes <= 0) {
      throw new RangeError(`invalid allocation size ${requestedBytes} for ${label}`);
    }
    const byteLength = Math.ceil(requestedBytes / 4) * 4;
    // Readback buffers keep their exact size: callers map them whole.
    const physicalBytes = byteLength >= POOL_GRANULARITY_BYTES && (usage & GPUBufferUsage.MAP_READ) === 0
      ? Math.ceil(byteLength / POOL_GRANULARITY_BYTES) * POOL_GRANULARITY_BYTES : byteLength;
    // The smallest idle buffer that covers the request and carries every
    // usage it needs is reused rather than creating another buffer. AlphaFold
    // cycles through a handful of large but unequal shapes, and matching size
    // or usage exactly kept one retired buffer per distinct shape and one more
    // per usage combination: an uploaded template update could never serve as
    // triangle scratch of the same size. A buffer more than twice the request
    // is left alone, though: handing a pair-sized buffer to an attention window
    // only forces the next pair-sized request to create another one.
    let pooledEntry: PooledGpuBuffer | undefined;
    for (const candidate of this.#pooledLru) {
      if ((candidate.usage & usage) !== usage || candidate.byteLength < physicalBytes
        || candidate.byteLength > 2 * physicalBytes) continue;
      if (options.requireSubmitted === true && candidate.generation >= this.#generation) continue;
      if (pooledEntry === undefined || candidate.byteLength < pooledEntry.byteLength) pooledEntry = candidate;
    }
    let buffer = pooledEntry?.buffer;
    if (pooledEntry !== undefined) {
      const pooled = this.#pool.get(pooledEntry.key);
      if (pooled === undefined) throw new Error("GPU allocator pool key missing during reuse");
      const index = pooled.lastIndexOf(pooledEntry);
      if (index < 0) throw new Error("GPU allocator pool entry missing during reuse");
      pooled.splice(index, 1);
      if (pooled.length === 0) this.#pool.delete(pooledEntry.key);
      this.#pooledLru.delete(pooledEntry);
      this.#pooledBytes -= pooledEntry.byteLength;
    }
    const allocationByteLength = pooledEntry?.byteLength ?? physicalBytes;
    const allocationUsage = pooledEntry?.usage ?? usage;
    if (buffer === undefined) {
      const tally = deviceTally(this.device);
      if (tally.resident + physicalBytes > tally.budget) {
        throw new GpuMemoryBudgetError(label, physicalBytes, tally.resident, tally.budget);
      }
      buffer = this.device.createBuffer({ label, size: physicalBytes, usage });
      tally.resident += physicalBytes;
      this.#residentBytes += physicalBytes;
      this.#peakResidentBytes = Math.max(this.#peakResidentBytes, this.#residentBytes);
      this.#bufferCount += 1;
    }
    this.#currentBytes += byteLength;
    // Labels carry a block or recycle index; grouping without it keeps the
    // composition readable across the forty-eight blocks of a stack.
    const group = allocationGroup(label);
    const live = this.#liveByLabel.get(group) ?? { bytes: 0, count: 0 };
    live.bytes += byteLength; live.count += 1;
    this.#liveByLabel.set(group, live);
    if (this.#currentBytes > this.#peakBytes) {
      this.#peakBytes = this.#currentBytes;
      this.#peakComposition = [...this.#liveByLabel]
        .map(([name, entry]) => ({ label: name, bytes: entry.bytes, count: entry.count }))
        .sort((left, right) => right.bytes - left.bytes);
    }
    this.#allocationCount += 1;
    return new AllocatedGpuBuffer(this, buffer, byteLength, allocationByteLength, usage, allocationUsage, label);
  }

  upload(label: string, data: ArrayBufferView, usage: GPUBufferUsageFlags): AllocatedGpuBuffer {
    const allocation = this.allocate(label, data.byteLength, usage | GPUBufferUsage.COPY_DST,
      { requireSubmitted: true });
    if (data.byteLength % 4 === 0) {
      this.device.queue.writeBuffer(allocation.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    } else {
      // WebGPU queue writes must be four-byte aligned even when the logical
      // storage type is f16 and has an odd element count.
      const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      this.device.queue.writeBuffer(allocation.buffer, 0, padded);
    }
    return allocation;
  }

  noteRelease(buffer: GPUBuffer, byteLength: number, allocationByteLength: number,
    usage: GPUBufferUsageFlags, label = ""): void {
    this.#currentBytes -= byteLength;
    if (this.#currentBytes < 0) throw new Error("GPU allocator accounting underflow");
    const live = this.#liveByLabel.get(allocationGroup(label));
    const group = allocationGroup(label);
    if (live !== undefined) {
      live.bytes -= byteLength; live.count -= 1;
      if (live.count <= 0) this.#liveByLabel.delete(group);
    }
    if (this.#pooling) {
      // A tensor can become dead while its command encoder is still being
      // populated. Keep the physical buffer reusable until the caller marks a
      // post-submit boundary: destroying it here would invalidate every
      // already-encoded command that still references it. The pool may exceed
      // its idle cap transiently within one command buffer.
      const key = `${allocationByteLength}:${usage}`;
      const pooled = this.#pool.get(key) ?? [];
      const entry = { buffer, byteLength: allocationByteLength, usage, key, generation: this.#generation };
      pooled.push(entry);
      this.#pool.set(key, pooled);
      this.#pooledLru.add(entry);
      this.#pooledBytes += allocationByteLength;
    } else {
      buffer.destroy();
      deviceTally(this.device).resident -= allocationByteLength;
      this.#residentBytes -= allocationByteLength;
    }
  }

  /**
   * Enforce the idle-pool cap at a boundary where every command referencing a
   * retired buffer has already been submitted. GPU queue ordering permits the
   * remaining pooled buffers to be reused by a later submission without a
   * host-side wait.
   */
  trimPooled(): void {
    while (this.#pooledBytes > this.#maxPooledBytes) this.#evictOldestPooled();
  }

  /**
   * Mark a submitted boundary: every buffer retired so far may now be taken
   * over by queue writes as well as by later dispatches, and the idle cap is
   * enforced.
   */
  noteSubmitted(): void {
    this.#generation += 1;
    // Insertion order is retirement order, so the stale entries lead the LRU.
    for (const entry of this.#pooledLru) {
      if (entry.generation + POOL_IDLE_GENERATIONS > this.#generation) break;
      this.#evictPooled(entry);
    }
    this.trimPooled();
  }

  #evictOldestPooled(): void {
    const entry = this.#pooledLru.values().next().value as PooledGpuBuffer | undefined;
    if (entry === undefined) throw new Error("GPU allocator pool accounting mismatch");
    this.#evictPooled(entry);
  }

  #evictPooled(entry: PooledGpuBuffer): void {
    this.#pooledLru.delete(entry);
    const pooled = this.#pool.get(entry.key);
    if (pooled === undefined) throw new Error("GPU allocator pool key missing during eviction");
    const index = pooled.indexOf(entry);
    if (index < 0) throw new Error("GPU allocator pool entry missing during eviction");
    pooled.splice(index, 1);
    if (pooled.length === 0) this.#pool.delete(entry.key);
    entry.buffer.destroy();
    deviceTally(this.device).resident -= entry.byteLength;
    this.#residentBytes -= entry.byteLength;
    this.#pooledBytes -= entry.byteLength;
  }

  destroyPooled(): void {
    for (const entry of this.#pooledLru) {
      entry.buffer.destroy();
      deviceTally(this.device).resident -= entry.byteLength;
      this.#residentBytes -= entry.byteLength;
      this.#pooledBytes -= entry.byteLength;
    }
    this.#pool.clear();
    this.#pooledLru.clear();
    if (this.#residentBytes < 0 || this.#pooledBytes !== 0) {
      throw new Error("GPU allocator resident accounting underflow");
    }
  }

  snapshot(): AllocationSnapshot {
    return {
      currentBytes: this.#currentBytes,
      peakBytes: this.#peakBytes,
      peakComposition: this.#peakComposition,
      residentBytes: this.#residentBytes,
      peakResidentBytes: this.#peakResidentBytes,
      pooledBytes: this.#pooledBytes,
      allocationCount: this.#allocationCount,
      bufferCount: this.#bufferCount,
    };
  }
}
