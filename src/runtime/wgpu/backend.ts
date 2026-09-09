/**
 * The native addon, and what it reports about the adapter it opened.
 *
 * Loaded through createRequire rather than imported, because the addon is
 * built out of tree and a machine without it must still typecheck and run
 * everything that does not touch wgpu.
 */
import { createRequire } from "node:module";

/** The native surface. Every value crossing it is a handle, a number or a string. */
export interface WgpuNative {
  openAdapter(): string;
  requestDevice(unchecked: boolean, features: string[],
    limitNames: string[], limitValues: number[]): string;
  createBuffer(bytes: number, usage: number, mapped: boolean): number;
  writeBuffer(handle: number, offset: number, data: Buffer): void;
  createShaderModule(code: string): number;
  createComputePipeline(module: number, entry: string,
    constantNames: string[], constantValues: number[]): number;
  createBindGroup(pipeline: number, groupIndex: number,
    buffers: number[], offsets: number[], sizes: number[]): number;
  createQuerySet(count: number): number;
  createCommandEncoder(): number;
  beginComputePass(encoder: number, querySet: number, beginning: number, end: number): void;
  passSetPipeline(encoder: number, pipeline: number): void;
  passSetBindGroup(encoder: number, index: number, group: number): void;
  passDispatch(encoder: number, x: number, y: number, z: number): void;
  passPushDebugGroup(encoder: number, label: string): void;
  passPopDebugGroup(encoder: number): void;
  passEnd(encoder: number): void;
  clearBuffer(encoder: number, buffer: number, offset: number, bytes: number): void;
  copyBufferToBuffer(encoder: number, source: number, sourceOffset: number,
    destination: number, destinationOffset: number, bytes: number): void;
  resolveQuerySet(encoder: number, set: number, first: number, count: number,
    destination: number, offset: number): void;
  finish(encoder: number): number;
  submit(buffers: number[]): void;
  timeDispatches(pipeline: number, group: number,
    x: number, y: number, z: number, repeats: number): number;
  mapBuffer(buffer: number, write: boolean): Promise<void>;
  mappedRange(buffer: number, offset: number, bytes: number): Buffer;
  writeMappedRange(buffer: number, offset: number, data: Buffer): void;
  unmap(buffer: number): void;
  onSubmittedWorkDone(): Promise<void>;
  pushErrorScope(): void;
  popErrorScope(): string;
  takeUncapturedErrors(): string[];
  handleCounts(): number[];
  destroyBuffer(handle: number): void;
  destroyBindGroup(handle: number): void;
  destroyPipeline(handle: number): void;
  destroyShaderModule(handle: number): void;
  destroyQuerySet(handle: number): void;
}

/** One entry of the adapter's cooperative-matrix configurations. */
export interface WgpuMatrixConfig {
  readonly M: number;
  readonly N: number;
  readonly K: number;
  readonly componentType: string;
  readonly resultComponentType: string;
}

export interface WgpuReport {
  readonly name: string;
  readonly driver: string;
  readonly backend: string;
  /** The wgpu tree the addon was built against: a path, or "crates.io". */
  readonly wgpuSource: string;
  /** Whether the injected bounds and loop checks are on; null before a device. */
  readonly checks: boolean | null;
  readonly features: readonly string[];
  readonly subgroupMinSize: number;
  readonly subgroupMaxSize: number;
  readonly matrixConfigs: readonly WgpuMatrixConfig[];
  readonly limits: Readonly<Record<string, number>>;
}

const ADDON = "../../../afwebgpu-wgpu.node";

let loaded: WgpuNative | undefined;

/**
 * The addon, or a message saying how to build it.
 *
 * Kept out of module scope so importing anything here on a machine without the
 * addon costs nothing and throws nothing.
 */
export function wgpuNative(): WgpuNative {
  if (loaded !== undefined) return loaded;
  try {
    loaded = createRequire(import.meta.url)(ADDON) as WgpuNative;
  } catch (error) {
    throw new Error("the wgpu addon is not built: run native/wgpu-backend/build.sh"
      + ` (${String(error)})`);
  }
  return loaded;
}

export function wgpuAddonExists(): boolean {
  try {
    wgpuNative();
    return true;
  } catch {
    return false;
  }
}
