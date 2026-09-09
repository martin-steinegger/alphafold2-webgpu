/**
 * A WebGPU surface over the wgpu addon.
 *
 * The model is written against WebGPU and stays that way; this presents the
 * part of it the model uses, so requestAlphaFoldDevice takes this adapter
 * exactly as it takes Dawn's. Nothing here is a general implementation: a
 * method the model never calls is not here, and one it calls in one shape
 * only accepts that shape.
 */
import {
  wgpuNative, type WgpuMatrixConfig, type WgpuNative, type WgpuReport,
} from "./backend.js";

/** WebGPU's own bit values, which the addon translates by name. */
export const WGPU_BUFFER_USAGE = {
  MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8,
  INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128,
  INDIRECT: 256, QUERY_RESOLVE: 512,
} as const;

const MAP_MODE = { READ: 1, WRITE: 2 } as const;

/**
 * Frees a handle when the object holding it is collected.
 *
 * WebGPU has no destroy for a bind group, a pipeline or a shader module: they
 * are released when the last reference goes, and the model relies on that,
 * creating one bind group a dispatch. Without this the registry keeps every
 * one of them, which is 8,027 after a 59-residue fold and far more after a
 * long one, each holding a descriptor set.
 */
const collected = new FinalizationRegistry<{ free: (handle: number) => void; handle: number }>(
  ({ free, handle }) => { free(handle); });

/**
 * Installs what a WebGPU caller reads off the global: the constant namespaces,
 * and GPUBuffer, which the model tests a binding against with instanceof.
 *
 * Assigned over anything already there rather than beside it. A process that
 * loaded Dawn's bindings has Dawn's GPUBuffer installed, and a buffer from
 * this backend is not one of those, so an instanceof against it would be false
 * and the caller would read the binding as the wrong shape.
 */
export function installWgpuGlobals(): void {
  const scope = globalThis as unknown as Record<string, unknown>;
  scope.GPUBufferUsage = WGPU_BUFFER_USAGE;
  scope.GPUMapMode = MAP_MODE;
  scope.GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
  scope.GPUBuffer = WgpuBuffer;
}

class WgpuBuffer {
  readonly handle: number;
  readonly size: number;
  readonly usage: number;
  #mapped: { offset: number; view: Uint8Array }[] = [];
  #writable = false;
  #destroyed = false;

  constructor(private readonly native: WgpuNative,
    size: number, usage: number, mappedAtCreation: boolean) {
    this.size = size;
    this.usage = usage;
    this.handle = native.createBuffer(size, usage, mappedAtCreation);
    this.#writable = mappedAtCreation;
    collected.register(this, { free: native.destroyBuffer, handle: this.handle }, this);
  }

  async mapAsync(mode: number, offset = 0, size?: number): Promise<void> {
    void offset;
    void size;
    this.#writable = mode === MAP_MODE.WRITE;
    await this.native.mapBuffer(this.handle, this.#writable);
  }

  /**
   * A copy of the range, not a view into the mapped buffer.
   *
   * A write-mapped range is remembered and copied back at unmap, so a caller
   * that fills the array it was handed sees the same result as it would from a
   * real mapping.
   */
  getMappedRange(offset = 0, size?: number): ArrayBuffer {
    const bytes = size ?? this.size - offset;
    const held = this.native.mappedRange(this.handle, offset, bytes);
    // The addon hands back a buffer of its own, so its memory can be given out
    // as it stands. Copying it into a fresh array first costs a second pass
    // over the whole readback, which at 348 MB is most of what a caller waits
    // for. A Buffer that shares a pool is copied, because handing out its
    // pool would hand out its neighbours as well.
    const whole = held.byteOffset === 0 && held.byteLength === held.buffer.byteLength;
    const view = whole ? held : new Uint8Array(held);
    if (this.#writable) this.#mapped.push({ offset, view });
    return view.buffer as ArrayBuffer;
  }

  unmap(): void {
    if (this.#writable) {
      for (const range of this.#mapped) {
        this.native.writeMappedRange(this.handle, range.offset, Buffer.from(range.view));
      }
    }
    this.#mapped = [];
    this.native.unmap(this.handle);
  }

  destroy(): void {
    if (this.#destroyed) return;
    this.#destroyed = true;
    collected.unregister(this);
    this.native.destroyBuffer(this.handle);
  }
}

class WgpuShaderModule {
  readonly handle: number;

  constructor(native: WgpuNative, code: string) {
    this.handle = native.createShaderModule(code);
    collected.register(this, { free: native.destroyShaderModule, handle: this.handle });
  }

  /** WebGPU always resolves this; the addon throws on a rejected module instead. */
  async getCompilationInfo(): Promise<{ messages: never[] }> {
    return { messages: [] };
  }
}

/** What getBindGroupLayout returns: which pipeline and which group. */
interface WgpuBindGroupLayout {
  readonly pipeline: number;
  readonly index: number;
}

class WgpuComputePipeline {
  constructor(native: WgpuNative, readonly handle: number) {
    collected.register(this, { free: native.destroyPipeline, handle });
  }

  getBindGroupLayout(index: number): WgpuBindGroupLayout {
    return { pipeline: this.handle, index };
  }
}

class WgpuQuerySet {
  readonly handle: number;

  constructor(private readonly native: WgpuNative, readonly count: number) {
    this.handle = native.createQuerySet(count);
    collected.register(this, { free: native.destroyQuerySet, handle: this.handle }, this);
  }

  destroy(): void {
    collected.unregister(this);
    this.native.destroyQuerySet(this.handle);
  }
}

interface BindGroupEntry {
  binding: number;
  resource: { buffer: WgpuBuffer; offset?: number; size?: number };
}

class WgpuBindGroup {
  constructor(native: WgpuNative, readonly handle: number) {
    collected.register(this, { free: native.destroyBindGroup, handle });
  }
}

class WgpuComputePass {
  constructor(private readonly native: WgpuNative, private readonly owner: WgpuCommandEncoder) {}

  private get encoder(): number { return this.owner.handle; }

  setPipeline(pipeline: WgpuComputePipeline): void {
    this.owner.retain(pipeline);
    this.native.passSetPipeline(this.encoder, pipeline.handle);
  }

  setBindGroup(index: number, group: WgpuBindGroup): void {
    this.owner.retain(group);
    this.native.passSetBindGroup(this.encoder, index, group.handle);
  }

  dispatchWorkgroups(x: number, y = 1, z = 1): void {
    this.native.passDispatch(this.encoder, x, y, z);
  }

  pushDebugGroup(label: string): void {
    this.native.passPushDebugGroup(this.encoder, label);
  }

  popDebugGroup(): void {
    this.native.passPopDebugGroup(this.encoder);
  }

  end(): void {
    this.native.passEnd(this.encoder);
  }
}

interface TimestampWrites {
  querySet: WgpuQuerySet;
  beginningOfPassWriteIndex?: number;
  endOfPassWriteIndex?: number;
}

class WgpuCommandBuffer {
  constructor(readonly handle: number) {}
}

class WgpuCommandEncoder {
  readonly handle: number;
  /**
   * Everything a recorded command names, until the commands are replayed.
   *
   * The model writes setBindGroup(0, device.createBindGroup(...)), so the
   * group is garbage the moment the call returns. Recorded commands hold only
   * handles, and the replay at finish looks them up, so a group collected
   * before then would take its native object with it.
   */
  #retained: object[] = [];

  constructor(private readonly native: WgpuNative) {
    this.handle = native.createCommandEncoder();
  }

  retain(held: object): void {
    this.#retained.push(held);
  }

  beginComputePass(descriptor?: { timestampWrites?: TimestampWrites }): WgpuComputePass {
    const writes = descriptor?.timestampWrites;
    this.native.beginComputePass(this.handle,
      writes === undefined ? -1 : writes.querySet.handle,
      writes?.beginningOfPassWriteIndex ?? -1,
      writes?.endOfPassWriteIndex ?? -1);
    return new WgpuComputePass(this.native, this);
  }

  clearBuffer(buffer: WgpuBuffer, offset = 0, bytes = 0): void {
    this.retain(buffer);
    this.native.clearBuffer(this.handle, buffer.handle, offset, bytes);
  }

  copyBufferToBuffer(source: WgpuBuffer, sourceOffset: number,
    destination: WgpuBuffer, destinationOffset: number, bytes: number): void {
    this.retain(source);
    this.retain(destination);
    this.native.copyBufferToBuffer(this.handle, source.handle, sourceOffset,
      destination.handle, destinationOffset, bytes);
  }

  resolveQuerySet(set: WgpuQuerySet, first: number, count: number,
    destination: WgpuBuffer, offset: number): void {
    this.retain(set);
    this.retain(destination);
    this.native.resolveQuerySet(this.handle, set.handle, first, count,
      destination.handle, offset);
  }

  finish(): WgpuCommandBuffer {
    const finished = new WgpuCommandBuffer(this.native.finish(this.handle));
    // The command buffer holds wgpu's own references from here on.
    this.#retained = [];
    return finished;
  }
}

class WgpuQueue {
  constructor(private readonly native: WgpuNative) {}

  writeBuffer(buffer: WgpuBuffer, offset: number, data: ArrayBufferView | ArrayBuffer,
    dataOffset = 0, size?: number): void {
    const view = ArrayBuffer.isView(data)
      ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
      : new Uint8Array(data);
    const start = dataOffset;
    const bytes = size ?? view.byteLength - start;
    this.native.writeBuffer(buffer.handle, offset,
      Buffer.from(view.buffer, view.byteOffset + start, bytes));
  }

  /**
   * Anything wgpu raised outside an error scope since the last submission is
   * thrown here. WebGPU sends an uncaptured error to the console and carries
   * on; wgpu's own default handler panics, and a panic crossing N-API aborts
   * the process, so the addon collects them and this is where they surface.
   */
  submit(buffers: readonly WgpuCommandBuffer[]): void {
    this.native.submit(buffers.map((buffer) => buffer.handle));
    const errors = this.native.takeUncapturedErrors();
    if (errors.length > 0) throw new Error(errors.join("\n"));
  }

  async onSubmittedWorkDone(): Promise<void> {
    await this.native.onSubmittedWorkDone();
  }
}

class WgpuDevice {
  readonly queue: WgpuQueue;
  readonly features: Set<string>;
  readonly limits: Readonly<Record<string, number>>;

  constructor(private readonly native: WgpuNative, report: WgpuReport) {
    this.queue = new WgpuQueue(native);
    this.features = new Set(report.features);
    this.limits = report.limits;
  }

  createBuffer(descriptor: { size: number; usage: number; mappedAtCreation?: boolean }): WgpuBuffer {
    return new WgpuBuffer(this.native, descriptor.size, descriptor.usage,
      descriptor.mappedAtCreation === true);
  }

  createShaderModule(descriptor: { code: string }): WgpuShaderModule {
    return new WgpuShaderModule(this.native, descriptor.code);
  }

  createComputePipeline(descriptor: {
    compute: { module: WgpuShaderModule; entryPoint?: string;
      constants?: Record<string, number> };
  }): WgpuComputePipeline {
    const constants = descriptor.compute.constants ?? {};
    const names = Object.keys(constants);
    return new WgpuComputePipeline(this.native, this.native.createComputePipeline(
      descriptor.compute.module.handle, descriptor.compute.entryPoint ?? "main",
      names, names.map((name) => constants[name]!)));
  }

  /**
   * wgpu compiles a pipeline on the calling thread, so this is the same work
   * as the synchronous form and only the shape differs.
   */
  async createComputePipelineAsync(descriptor: Parameters<
    WgpuDevice["createComputePipeline"]>[0]): Promise<WgpuComputePipeline> {
    return this.createComputePipeline(descriptor);
  }

  createBindGroup(descriptor: {
    layout: WgpuBindGroupLayout; entries: readonly BindGroupEntry[];
  }): WgpuBindGroup {
    const ordered = [...descriptor.entries].sort((left, right) => left.binding - right.binding);
    if (ordered.some((entry, index) => entry.binding !== index)) {
      throw new RangeError("the wgpu backend binds slots 0 upwards without gaps");
    }
    return new WgpuBindGroup(this.native, this.native.createBindGroup(
      descriptor.layout.pipeline, descriptor.layout.index,
      ordered.map((entry) => entry.resource.buffer.handle),
      ordered.map((entry) => entry.resource.offset ?? 0),
      // Zero stands for WebGPU's omitted size, which is the rest of the buffer.
      ordered.map((entry) => entry.resource.size ?? 0)));
  }

  createQuerySet(descriptor: { type: string; count: number }): WgpuQuerySet {
    if (descriptor.type !== "timestamp") {
      throw new RangeError(`the wgpu backend has no ${descriptor.type} query set`);
    }
    return new WgpuQuerySet(this.native, descriptor.count);
  }

  createCommandEncoder(): WgpuCommandEncoder {
    return new WgpuCommandEncoder(this.native);
  }

  pushErrorScope(filter: string): void {
    if (filter !== "validation") {
      throw new RangeError(`the wgpu backend scopes validation errors only, not ${filter}`);
    }
    this.native.pushErrorScope();
  }

  async popErrorScope(): Promise<{ message: string } | null> {
    const message = this.native.popErrorScope();
    return message === "" ? null : { message };
  }

  /**
   * The device outlives the process here: the addon holds one, opened once.
   * Nothing is freed, so this only stops a caller using it further.
   */
  destroy(): void {}
}

interface WgpuAdapterInfo {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly subgroupMinSize: number;
  readonly subgroupMaxSize: number;
  readonly subgroupMatrixConfigs: readonly WgpuMatrixConfig[];
}

class WgpuAdapter {
  readonly features: Set<string>;
  readonly limits: Readonly<Record<string, number>>;
  readonly info: WgpuAdapterInfo;

  constructor(private readonly native: WgpuNative, readonly report: WgpuReport,
    private readonly unchecked: boolean) {
    this.features = new Set(report.features);
    this.limits = report.limits;
    this.info = {
      vendor: report.backend,
      architecture: "",
      device: report.name,
      description: report.driver,
      subgroupMinSize: report.subgroupMinSize,
      subgroupMaxSize: report.subgroupMaxSize,
      subgroupMatrixConfigs: report.matrixConfigs,
    };
  }

  /**
   * Features the adapter does not have are dropped rather than refused, which
   * is what a WebGPU caller listing optional features expects; the model asks
   * for Dawn's names, and only those with a wgpu counterpart reach the addon.
   */
  async requestDevice(descriptor?: {
    requiredFeatures?: readonly string[];
    requiredLimits?: Readonly<Record<string, number>>;
  }): Promise<WgpuDevice> {
    const features = (descriptor?.requiredFeatures ?? []).filter(
      (feature) => this.features.has(feature));
    const limits = Object.entries(descriptor?.requiredLimits ?? {});
    const granted = JSON.parse(this.native.requestDevice(this.unchecked, features,
      limits.map(([name]) => name), limits.map(([, value]) => value))) as WgpuReport;
    return new WgpuDevice(this.native, granted);
  }
}

/**
 * Opens the addon's device and presents it as a WebGPU adapter.
 *
 * unchecked drops wgpu's injected bounds and loop checks, which is what a
 * Dawn instance built with dawnInstanceFlags({ unclamped: true }) does, and is
 * what a comparison against Dawn needs. AFWEBGPU_WGPU_CHECKED=1 puts them back,
 * for a measurement of what they cost. The addon holds one device a process, so
 * the first call settles this.
 */
export function requestWgpuAdapter(
  unchecked = process.env.AFWEBGPU_WGPU_CHECKED !== "1",
): GPUAdapter {
  installWgpuGlobals();
  const native = wgpuNative();
  const report = JSON.parse(native.openAdapter()) as WgpuReport;
  return new WgpuAdapter(native, report, unchecked) as unknown as GPUAdapter;
}
