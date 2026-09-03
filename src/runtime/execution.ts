import { type ActivationStorage, storageWords } from "./storage.js";
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

const ADD_IN_PLACE_PACKED_SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read_write> base: array<u32>;
@group(0) @binding(1) var<storage, read> update: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let word = id.x + id.y * GRID_WIDTH * 64u;
  if (word >= arrayLength(&base)) { return; }
  let element = word * 2u;
  base[word] = pack2x16float(unpack2x16float(base[word])
    + vec2<f32>(update[element], update[element + 1u]));
}`;

const PACK_HALVES_SHADER = `
const GRID_WIDTH: u32 = 32768u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> packed: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let word = id.x + id.y * GRID_WIDTH * 64u;
  if (word >= arrayLength(&packed)) { return; }
  let element = word * 2u;
  packed[word] = pack2x16float(vec2<f32>(source[element], source[element + 1u]));
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
  /** Set once the set is full; later dispatches run untimed. */
  truncated: boolean;
}

interface PendingTimestampReadback {
  readonly querySet: GPUQuerySet;
  readonly labels: readonly string[];
  readonly readback: GpuTensor;
}

/**
 * How long one command buffer should take on the device.
 *
 * A driver that waits seconds for a command buffer to finish may decide the
 * GPU has hung; on Apple that shows as a stalled or crashed browser rather
 * than an error. Dispatches at 1500 residues are two orders of magnitude
 * longer than at 59, so a fixed dispatch count per buffer cannot bound its
 * duration: the count is measured and adjusted instead.
 */
const SUBMISSION_TARGET_MILLISECONDS = 250;

/** Dispatches the first buffers hold, before any of them has been timed. */
const SUBMISSION_START_DISPATCHES = 48;

/** The range the measured limit stays inside. */
const SUBMISSION_DISPATCH_RANGE = [8, 384] as const;

/**
 * The dispatch count for the next command buffers, given how the last one
 * went. See `noteSubmissionDuration` for why it falls faster than it climbs.
 */
export function nextSubmissionDispatchLimit(
  limit: number, milliseconds: number, dispatches: number,
): number {
  if (!(milliseconds > 0) || dispatches <= 0) return limit;
  const [low, high] = SUBMISSION_DISPATCH_RANGE;
  const adjusted = milliseconds > SUBMISSION_TARGET_MILLISECONDS
    ? Math.min(limit, dispatches * SUBMISSION_TARGET_MILLISECONDS / milliseconds)
    : limit + Math.max(1, limit / 10);
  return Math.min(high, Math.max(low, Math.round(adjusted)));
}

export interface WebGpuExecutionOptions {
  readonly transitionBufferLimit?: number;
  readonly maxPooledBytes?: number;
  /**
   * Bytes one binding may cover, at most the device's own limit.
   *
   * Every kernel that walks a tensor larger than this splits it into windows
   * or shards. Lowering it makes a short prediction take the path a long one
   * takes on a device with a small limit, and reports what still exceeds it.
   */
  readonly bindingBudgetBytes?: number;
}

export class WebGpuExecution {
  readonly device: GPUDevice;
  readonly allocator: GpuBufferAllocator;
  readonly pipelines: ComputePipelineCache;
  readonly transitionBufferLimit: number;
  /** Bytes one binding may cover: the device's limit unless a budget lowers it. */
  readonly bindingLimitBytes: number;
  #submissionDispatchLimit = SUBMISSION_START_DISPATCHES;
  readonly #allocations: AllocatedGpuBuffer[] = [];
  #timestamps: TimestampCapture | undefined;
  #dispatchCount = 0;
  #encoderHolder: { encoder: GPUCommandEncoder } | undefined;
  #activeEncoder: GPUCommandEncoder | undefined;
  #activePass: GPUComputePassEncoder | undefined;

  constructor(device: GPUDevice, options: WebGpuExecutionOptions = {}) {
    this.device = device;
    this.allocator = new GpuBufferAllocator(device, true, options.maxPooledBytes);
    this.pipelines = pipelineCacheForDevice(device);
    this.transitionBufferLimit = Math.min(
      device.limits.maxStorageBufferBindingSize,
      options.bindingBudgetBytes ?? device.limits.maxStorageBufferBindingSize,
      options.transitionBufferLimit ?? device.limits.maxStorageBufferBindingSize,
    );
    if (!Number.isSafeInteger(this.transitionBufferLimit) || this.transitionBufferLimit <= 0) {
      throw new RangeError("transitionBufferLimit must be a positive safe integer");
    }
    this.#bindingBudgetBytes = options.bindingBudgetBytes;
    this.bindingLimitBytes = Math.min(
      device.limits.maxStorageBufferBindingSize, options.bindingBudgetBytes ?? Number.MAX_SAFE_INTEGER);
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

  /**
   * Dispatches encoded so far.
   *
   * The loops that grow with the sequence length watch this to decide when a
   * command buffer has taken on enough work, since a submission a driver waits
   * seconds for is a submission it may decide has hung.
   */
  get dispatchCount(): number { return this.#dispatchCount; }

  /**
   * Directs every dispatch at the holder's current command buffer.
   *
   * A long block is split across several buffers, and the functions encoding
   * it hold whichever encoder they were called with. Rather than thread a
   * replacement back through each of them, dispatches follow the holder, which
   * the split updates. Cleared with `undefined` once the block is submitted.
   */
  setEncoderHolder(holder: { encoder: GPUCommandEncoder } | undefined): void {
    this.#encoderHolder = holder;
  }

  /**
   * Reports bindings larger than a budget, without a device that enforces one.
   *
   * A binding may cover only `maxStorageBufferBindingSize` bytes of a buffer,
   * and the adapters here allow gigabytes, so a kernel that binds a whole
   * pair passes at every length that fits in memory and then fails on a
   * device with the 128 MiB default. Setting a budget well under any real
   * tensor lists the kernels that would fail there, by label.
   */
  setBindingBudget(bytes: number | undefined): void {
    this.#bindingBudgetBytes = bytes;
    this.#oversizedBindings.clear();
  }

  /** Labels of the dispatches that exceeded the budget, and their largest binding. */
  get oversizedBindings(): ReadonlyMap<string, number> {
    return this.#oversizedBindings;
  }

  #recordOversizedBindings(tensors: readonly GpuTensor[], label: string | undefined): void {
    const budget = this.#bindingBudgetBytes;
    if (budget === undefined) return;
    for (const tensor of tensors) {
      const bytes = tensor.elements * 4;
      // Uploaded buffers hold weights and feature tables, which a kernel reads
      // whole; only activations can be walked in windows.
      if (bytes <= budget || tensor.allocation.uploaded) continue;
      const key = label ?? "unlabelled";
      this.#oversizedBindings.set(key, Math.max(this.#oversizedBindings.get(key) ?? 0, bytes));
    }
  }

  #bindingBudgetBytes: number | undefined;
  readonly #oversizedBindings = new Map<string, number>();

  /** Dispatches a growing loop puts in one command buffer before splitting it. */
  get submissionDispatchLimit(): number {
    return this.#submissionDispatchLimit;
  }

  /**
   * Records how long a command buffer took, and adjusts the limit.
   *
   * The caller measures the interval between two completions with the queue
   * kept full, which is the device time of one buffer. A buffer over the
   * target cuts the limit in proportion at once, and buffers under it raise it
   * by a tenth: the cost of one buffer that runs long is a driver deciding the
   * GPU has hung, so the control has to fall faster than it climbs. Averaging
   * instead would let the many cheap dispatches of the MSA stack hide the few
   * expensive ones of the pair stack, which is what sets the duration.
   */
  noteSubmissionDuration(milliseconds: number, dispatches: number): void {
    this.#submissionDispatchLimit = nextSubmissionDispatchLimit(
      this.#submissionDispatchLimit, milliseconds, dispatches);
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
    this.#dispatchCount += 1;
    const target = this.#encoderHolder?.encoder ?? encoder;
    const timestamp = this.#timestamps;
    let timestampWrites: GPUComputePassTimestampWrites | undefined;
    // A query set holds at most 2048 dispatches, and a block at a long chain
    // length has more. The profile then covers the first of them rather than
    // failing the prediction it was measuring.
    if (timestamp !== undefined && timestamp.nextQuery + 2 > timestamp.querySet.count) {
      timestamp.truncated = true;
    }
    if (timestamp !== undefined && !timestamp.truncated) {
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
      if (this.#activeEncoder !== target || this.#activePass === undefined) {
        this.endComputePass();
        this.#activeEncoder = target;
        this.#activePass = target.beginComputePass({ label: "afwebgpu.compute" });
      }
      pass = this.#activePass;
      if (label !== undefined) pass.pushDebugGroup(label);
    } else {
      pass = target.beginComputePass({
        ...(label === undefined ? {} : { label }),
        timestampWrites: timestampWrites!,
      });
    }
    if (this.#bindingBudgetBytes !== undefined) this.#recordOversizedBindings(tensors, label);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: tensors.map((tensor, binding) => ({ binding, resource: {
        buffer: tensor.allocation.buffer,
        offset: (tensor.offsetElements ?? 0) * 4,
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

  /**
   * Adds an f32 update into a base tensor, which may be packed.
   *
   * A packed base is written a word at a time, so one invocation owns the two
   * elements sharing a word and no two invocations touch the same word.
   */
  async addInPlace(encoder: GPUCommandEncoder, base: GpuTensor, update: GpuTensor, label: string,
    storage: ActivationStorage = "f32"): Promise<void> {
    if (base.elements !== storageWords(update.elements, storage)) {
      throw new RangeError("residual tensors must have equal sizes");
    }
    if (storage === "f16" && update.elements % 2 !== 0) {
      throw new RangeError("a packed residual needs an even element count");
    }
    const pipeline = storage === "f32"
      ? await this.pipelines.get("runtime:add-in-place", ADD_IN_PLACE_SHADER)
      : await this.pipelines.get("runtime:add-in-place-packed", ADD_IN_PLACE_PACKED_SHADER);
    const grid = this.linearGrid(base.elements);
    this.dispatch(encoder, pipeline, [base, update], grid[0], grid[1], 1, label);
  }

  /** Packs f32 elements into half-precision words. */
  async packHalves(encoder: GPUCommandEncoder, source: GpuTensor, packed: GpuTensor, label: string): Promise<void> {
    if (packed.elements !== storageWords(source.elements, "f16")) {
      throw new RangeError("packing needs a target of the source's word count");
    }
    const pipeline = await this.pipelines.get("runtime:pack-halves", PACK_HALVES_SHADER);
    const grid = this.linearGrid(packed.elements);
    this.dispatch(encoder, pipeline, [source, packed], grid[0], grid[1], 1, label);
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
    const result = new Float32Array(
      readback.allocation.buffer.getMappedRange(0, readback.elements * Float32Array.BYTES_PER_ELEMENT).slice(0),
    );
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
      truncated: false,
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
      const values = new BigUint64Array(
        pending.readback.allocation.buffer
          .getMappedRange(0, pending.readback.elements * Float32Array.BYTES_PER_ELEMENT).slice(0),
      );
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

  /** Mark a queue-submit boundary at which retired buffers may be destroyed. */
  noteSubmitted(): void { this.allocator.noteSubmitted(); }

  releaseSince(checkpoint: number): void {
    if (!Number.isSafeInteger(checkpoint) || checkpoint < 0 || checkpoint > this.#allocations.length) {
      throw new RangeError(`invalid GPU allocation checkpoint ${checkpoint}`);
    }
    for (let index = this.#allocations.length - 1; index >= checkpoint; index -= 1) {
      this.#allocations[index]!.release();
    }
    this.#allocations.length = checkpoint;
    // releaseSince is used after a submitted execution window. Enforce the
    // bound for allocations retired after that submit as well as scratch that
    // was retired while encoding it.
    this.allocator.trimPooled();
  }

  release(): void {
    this.releaseSince(0);
    this.allocator.destroyPooled();
  }
}
