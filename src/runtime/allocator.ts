export interface AllocationSnapshot {
  readonly currentBytes: number;
  readonly peakBytes: number;
  /** Bytes held by live and pooled GPUBuffer objects. */
  readonly residentBytes: number;
  /** Maximum physical GPUBuffer storage retained by this allocator. */
  readonly peakResidentBytes: number;
  readonly allocationCount: number;
  readonly bufferCount: number;
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
  readonly #pooling: boolean;
  readonly #pool = new Map<string, GPUBuffer[]>();

  constructor(device: GPUDevice, pooling = false) {
    this.device = device;
    this.#pooling = pooling;
  }

  allocate(label: string, requestedBytes: number, usage: GPUBufferUsageFlags): AllocatedGpuBuffer {
    if (!Number.isSafeInteger(requestedBytes) || requestedBytes <= 0) {
      throw new RangeError(`invalid allocation size ${requestedBytes} for ${label}`);
    }
    const byteLength = Math.ceil(requestedBytes / 4) * 4;
    const key = `${byteLength}:${usage}`;
    const pooled = this.#pool.get(key);
    let buffer = pooled?.pop();
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
    if (this.#pooling) {
      const key = `${byteLength}:${usage}`;
      const pooled = this.#pool.get(key) ?? [];
      pooled.push(buffer);
      this.#pool.set(key, pooled);
    } else {
      buffer.destroy();
      this.#residentBytes -= byteLength;
    }
  }

  destroyPooled(): void {
    for (const [key, buffers] of this.#pool) {
      const byteLength = Number(key.slice(0, key.indexOf(":")));
      for (const buffer of buffers) {
        buffer.destroy();
        this.#residentBytes -= byteLength;
      }
    }
    this.#pool.clear();
    if (this.#residentBytes < 0) throw new Error("GPU allocator resident accounting underflow");
  }

  snapshot(): AllocationSnapshot {
    return {
      currentBytes: this.#currentBytes,
      peakBytes: this.#peakBytes,
      residentBytes: this.#residentBytes,
      peakResidentBytes: this.#peakResidentBytes,
      allocationCount: this.#allocationCount,
      bufferCount: this.#bufferCount,
    };
  }
}
