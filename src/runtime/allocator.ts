export interface AllocationSnapshot {
  readonly currentBytes: number;
  readonly peakBytes: number;
  /** Bytes held by live and pooled GPUBuffer objects. */
  readonly residentBytes: number;
  /** Maximum physical GPUBuffer storage retained by this allocator. */
  readonly peakResidentBytes: number;
  readonly pooledBytes: number;
  readonly allocationCount: number;
  readonly bufferCount: number;
}

// Large enough to retain the repeated AF2 block working set for short inputs,
// while bounding idle scratch on unified-memory devices for long inputs.
export const COMPACT_GPU_POOL_BYTES = 576 * 1024 ** 2;

interface PooledGpuBuffer {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly key: string;
}

export class AllocatedGpuBuffer {
  readonly buffer: GPUBuffer;
  readonly byteLength: number;
  readonly usage: GPUBufferUsageFlags;
  #allocator: GpuBufferAllocator | undefined;

  constructor(allocator: GpuBufferAllocator, buffer: GPUBuffer, byteLength: number, usage: GPUBufferUsageFlags) {
    this.#allocator = allocator;
    this.buffer = buffer;
    this.byteLength = byteLength;
    this.usage = usage;
  }

  release(): void {
    const allocator = this.#allocator;
    if (allocator === undefined) return;
    this.#allocator = undefined;
    allocator.noteRelease(this.buffer, this.byteLength, this.usage);
  }
}

export class GpuBufferAllocator {
  readonly device: GPUDevice;
  #currentBytes = 0;
  #peakBytes = 0;
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

  constructor(device: GPUDevice, pooling = false, maxPooledBytes = Number.POSITIVE_INFINITY) {
    if (maxPooledBytes !== Number.POSITIVE_INFINITY
      && (!Number.isSafeInteger(maxPooledBytes) || maxPooledBytes < 0)) {
      throw new RangeError("maxPooledBytes must be a non-negative safe integer or Infinity");
    }
    this.device = device;
    this.#pooling = pooling;
    this.#maxPooledBytes = maxPooledBytes;
  }

  allocate(label: string, requestedBytes: number, usage: GPUBufferUsageFlags): AllocatedGpuBuffer {
    if (!Number.isSafeInteger(requestedBytes) || requestedBytes <= 0) {
      throw new RangeError(`invalid allocation size ${requestedBytes} for ${label}`);
    }
    const byteLength = Math.ceil(requestedBytes / 4) * 4;
    const key = `${byteLength}:${usage}`;
    const pooled = this.#pool.get(key);
    const pooledEntry = pooled?.pop();
    let buffer = pooledEntry?.buffer;
    if (pooledEntry !== undefined) {
      this.#pooledLru.delete(pooledEntry);
      this.#pooledBytes -= byteLength;
    }
    if (buffer === undefined) {
      buffer = this.device.createBuffer({ label, size: byteLength, usage });
      this.#residentBytes += byteLength;
      this.#peakResidentBytes = Math.max(this.#peakResidentBytes, this.#residentBytes);
      this.#bufferCount += 1;
    }
    if (pooled?.length === 0) this.#pool.delete(key);
    this.#currentBytes += byteLength;
    this.#peakBytes = Math.max(this.#peakBytes, this.#currentBytes);
    this.#allocationCount += 1;
    return new AllocatedGpuBuffer(this, buffer, byteLength, usage);
  }

  upload(label: string, data: ArrayBufferView, usage: GPUBufferUsageFlags): AllocatedGpuBuffer {
    const allocation = this.allocate(label, data.byteLength, usage | GPUBufferUsage.COPY_DST);
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

  noteRelease(buffer: GPUBuffer, byteLength: number, usage: GPUBufferUsageFlags): void {
    this.#currentBytes -= byteLength;
    if (this.#currentBytes < 0) throw new Error("GPU allocator accounting underflow");
    if (this.#pooling && byteLength <= this.#maxPooledBytes) {
      while (this.#pooledBytes + byteLength > this.#maxPooledBytes) this.#evictOldestPooled();
      const key = `${byteLength}:${usage}`;
      const pooled = this.#pool.get(key) ?? [];
      const entry = { buffer, byteLength, key };
      pooled.push(entry);
      this.#pool.set(key, pooled);
      this.#pooledLru.add(entry);
      this.#pooledBytes += byteLength;
    } else {
      buffer.destroy();
      this.#residentBytes -= byteLength;
    }
  }

  #evictOldestPooled(): void {
    const entry = this.#pooledLru.values().next().value as PooledGpuBuffer | undefined;
    if (entry === undefined) throw new Error("GPU allocator pool accounting mismatch");
    this.#pooledLru.delete(entry);
    const pooled = this.#pool.get(entry.key);
    if (pooled === undefined) throw new Error("GPU allocator pool key missing during eviction");
    const index = pooled.indexOf(entry);
    if (index < 0) throw new Error("GPU allocator pool entry missing during eviction");
    pooled.splice(index, 1);
    if (pooled.length === 0) this.#pool.delete(entry.key);
    entry.buffer.destroy();
    this.#residentBytes -= entry.byteLength;
    this.#pooledBytes -= entry.byteLength;
  }

  destroyPooled(): void {
    for (const entry of this.#pooledLru) {
      entry.buffer.destroy();
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
      residentBytes: this.#residentBytes,
      peakResidentBytes: this.#peakResidentBytes,
      pooledBytes: this.#pooledBytes,
      allocationCount: this.#allocationCount,
      bufferCount: this.#bufferCount,
    };
  }
}
