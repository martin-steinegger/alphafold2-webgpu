import { GpuBufferAllocator, type AllocatedGpuBuffer, type AllocationSnapshot } from "./allocator.js";
import { pipelineCacheForDevice, type ComputePipelineCache } from "./pipeline-cache.js";

const GRID_WIDTH = 32_768;
const ADD_IN_PLACE_SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read_write> base: array<f32>;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * 64u;
  base[index] += update[index];
}`;

export interface GpuTensor {
  readonly allocation: AllocatedGpuBuffer;
  readonly elements: number;
  readonly offsetElements?: number;
}

export interface GpuTimestampEntry {
  readonly label: string;
  readonly nanoseconds: number;
}

interface TimestampCapture {
  readonly querySet: GPUQuerySet;
  readonly labels: string[];
  nextQuery: number;
}

interface PendingTimestampReadback {
  readonly querySet: GPUQuerySet;
  readonly labels: readonly string[];
  readonly readback: GpuTensor;
}

export interface WebGpuExecutionOptions {
  readonly transitionBufferLimit?: number;
  readonly maxPooledBytes?: number;
}

export class WebGpuExecution {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  readonly transitionBufferLimit: number;
  readonly #allocations: AllocatedGpuBuffer[] = [];
  #timestamps: TimestampCapture | undefined;
  #activeEncoder: GPUCommandEncoder | undefined;
  #activePass: GPUComputePassEncoder | undefined;

  constructor(device: GPUDevice, options: WebGpuExecutionOptions = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device, true, options.maxPooledBytes);
    this.pipelines = pipelineCacheForDevice(device);
    this.transitionBufferLimit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      options.transitionBufferLimit ?? device.limits.maxStorageBufferBindingSize,
    );
    if (!Number.isSafeInteger(this.transitionBufferLimit) || this.transitionBufferLimit <= 0) {
      throw new RangeError("transitionBufferLimit must be a positive safe integer");
    }
  }

  upload(label: string, data: ArrayBufferView, usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE): GpuTensor {
    const allocation = this.allocator.upload(label, data, usage);
    this.#allocations.push(allocation);
    return { allocation, elements: data.byteLength / 4 };
  }

  allocate(label: string, elements: number, usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE): GpuTensor {
    const allocation = this.allocator.allocate(label, elements * 4, usage);
    this.#allocations.push(allocation);
    return { allocation, elements };
  }

  view(tensor: GpuTensor, offsetElements: number, elements: number): GpuTensor {
    if (!Number.isSafeInteger(offsetElements) || !Number.isSafeInteger(elements)
      || offsetElements < 0 || elements <= 0 || offsetElements + elements > tensor.elements) {
      throw new RangeError(`invalid GPU tensor view ${offsetElements}:${elements} of ${tensor.elements}`);
    }
    return {
      allocation: tensor.allocation,
      elements,
      offsetElements: (tensor.offsetElements ?? 0) + offsetElements,
    };
  }

  linearGrid(elements: number, workgroupSize = 64): readonly [number, number] {
    const groups = Math.ceil(elements / workgroupSize);
    return [Math.min(groups, GRID_WIDTH), Math.ceil(groups / GRID_WIDTH)];
  }

  dispatch(
    encoder: GPUCommandEncoder,
    pipeline: GPUComputePipeline,
    tensors: readonly GpuTensor[],
    x: number,
    y = 1,
    z = 1,
    label?: string,
  ): void {
    const timestamp = this.#timestamps;
    let timestampWrites: GPUComputePassTimestampWrites | undefined;
    if (timestamp !== undefined) {
      if (timestamp.nextQuery + 2 > timestamp.querySet.count) {
        throw new RangeError("GPU timestamp query capacity exceeded");
      }
      timestamp.labels.push(label ?? `dispatch-${timestamp.labels.length}`);
      timestampWrites = {
        querySet: timestamp.querySet,
        beginningOfPassWriteIndex: timestamp.nextQuery,
        endOfPassWriteIndex: timestamp.nextQuery + 1,
      };
      timestamp.nextQuery += 2;
    }
    let pass: GPUComputePassEncoder;
    const reusable = timestampWrites === undefined;
    if (reusable) {
      if (this.#activeEncoder !== encoder || this.#activePass === undefined) {
        this.endComputePass();
        this.#activeEncoder = encoder;
        this.#activePass = encoder.beginComputePass({ label: "afwebgpu.compute" });
      }
      pass = this.#activePass;
      if (label !== undefined) pass.pushDebugGroup(label);
    } else {
      pass = encoder.beginComputePass({
        ...(label === undefined ? {} : { label }),
        timestampWrites: timestampWrites!,
      });
    }
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: tensors.map((tensor, binding) => ({ binding, resource: tensor.offsetElements === undefined
        ? { buffer: tensor.allocation.buffer }
        : {
          buffer: tensor.allocation.buffer,
          offset: tensor.offsetElements * 4,
          size: tensor.elements * 4,
        } })),
    }));
    pass.dispatchWorkgroups(x, y, z);
    if (reusable) {
      if (label !== undefined) pass.popDebugGroup();
    } else {
      pass.end();
    }
  }

  endComputePass(encoder?: GPUCommandEncoder): void {
    if (encoder !== undefined && this.#activeEncoder !== undefined && this.#activeEncoder !== encoder) {
      throw new Error("attempted to end a compute pass with a different command encoder");
    }
    this.#activePass?.end();
    this.#activePass = undefined;
    this.#activeEncoder = undefined;
  }

  async addInPlace(encoder: GPUCommandEncoder, base: GpuTensor, update: GpuTensor, label: string): Promise<void> {
    if (base.elements !== update.elements) throw new RangeError("residual tensors must have equal sizes");
    const pipeline = await this.pipelines.get("runtime:add-in-place", ADD_IN_PLACE_SHADER);
    const grid = this.linearGrid(base.elements);
    this.dispatch(encoder, pipeline, [base, update], grid[0], grid[1], 1, label);
  }

  createReadback(label: string, tensor: GpuTensor, encoder: GPUCommandEncoder): GpuTensor {
    this.endComputePass(encoder);
    const readback = this.allocate(label, tensor.elements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    encoder.copyBufferToBuffer(
      tensor.allocation.buffer, (tensor.offsetElements ?? 0) * 4,
      readback.allocation.buffer, 0, tensor.elements * 4,
    );
    return readback;
  }

  async mapFloat32(readback: GpuTensor): Promise<Float32Array> {
    await readback.allocation.buffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.allocation.buffer.getMappedRange().slice(0));
    readback.allocation.buffer.unmap();
    return result;
  }

  snapshot(): AllocationSnapshot { return this.allocator.snapshot(); }

  beginTimestampProfile(maxDispatches = 256): void {
    if (!this.device.features.has("timestamp-query")) {
      throw new Error("timestamp-query was not requested on this WebGPU device");
    }
    if (this.#timestamps !== undefined) throw new Error("a GPU timestamp profile is already active");
    this.#timestamps = {
      querySet: this.device.createQuerySet({ type: "timestamp", count: maxDispatches * 2 }),
      labels: [],
      nextQuery: 0,
    };
  }

  finishTimestampProfile(encoder: GPUCommandEncoder): PendingTimestampReadback {
    this.endComputePass(encoder);
    const capture = this.#timestamps;
    if (capture === undefined) throw new Error("no GPU timestamp profile is active");
    this.#timestamps = undefined;
    const queryCount = capture.nextQuery;
    const elements = queryCount * 2;
    const resolve = this.allocate(
      "profile.timestamp-resolve", elements, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    );
    const readback = this.allocate(
      "profile.timestamp-readback", elements, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    );
    encoder.resolveQuerySet(capture.querySet, 0, queryCount, resolve.allocation.buffer, 0);
    encoder.copyBufferToBuffer(resolve.allocation.buffer, 0, readback.allocation.buffer, 0, queryCount * 8);
    return { querySet: capture.querySet, labels: capture.labels, readback };
  }

  async readTimestampProfile(pending: PendingTimestampReadback): Promise<readonly GpuTimestampEntry[]> {
    try {
      await pending.readback.allocation.buffer.mapAsync(GPUMapMode.READ);
      const values = new BigUint64Array(pending.readback.allocation.buffer.getMappedRange().slice(0));
      pending.readback.allocation.buffer.unmap();
      return pending.labels.map((label, index) => ({
        label,
        nanoseconds: Number(values[index * 2 + 1]! - values[index * 2]!),
      }));
    } finally {
      pending.querySet.destroy();
    }
  }

  checkpoint(): number { return this.#allocations.length; }

  releaseSince(checkpoint: number): void {
    if (!Number.isSafeInteger(checkpoint) || checkpoint < 0 || checkpoint > this.#allocations.length) {
      throw new RangeError(`invalid GPU allocation checkpoint ${checkpoint}`);
    }
    for (let index = this.#allocations.length - 1; index >= checkpoint; index -= 1) {
      this.#allocations[index]!.release();
    }
    this.#allocations.length = checkpoint;
  }

  release(): void {
    this.releaseSince(0);
    this.allocator.destroyPooled();
  }
}
