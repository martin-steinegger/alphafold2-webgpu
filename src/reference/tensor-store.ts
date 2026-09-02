import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { readTensor, readTensorRange, tensorByteLength, tensorElements } from "./dtype.js";

export interface BinaryTensorRecord {
  readonly file: string;
  readonly shape: readonly number[];
  readonly dtype: "float32" | "float16" | "int8";
  readonly byteOffset?: number;
  readonly block?: number;
  readonly scaleOffset?: number;
  readonly [metadata: string]: unknown;
}

export interface BinaryTensorShard {
  readonly file: string;
  readonly bytes: number;
  readonly sha256?: string;
}

export interface BinaryTensorBundle {
  readonly purpose?: string;
  readonly model?: string;
  readonly encoding?: string;
  readonly version?: number;
  readonly id?: string;
  readonly tensors?: number;
  readonly bytes?: number;
  /** Legacy manifests stored the shard count as a number. */
  readonly shards?: number;
  readonly files?: readonly BinaryTensorShard[];
}

export interface BinaryTensorManifest {
  readonly tensors: Readonly<Record<string, BinaryTensorRecord>>;
  readonly bundle?: BinaryTensorBundle;
  readonly [metadata: string]: unknown;
}

/** Element range of one block of a tensor stacked along its first axis, or the whole tensor. */
export function blockRange(elements: number, block?: number, blocks?: number): readonly [number, number] {
  if (block === undefined) return [0, elements];
  if (blocks === undefined || !Number.isSafeInteger(blocks) || blocks <= 0 || elements % blocks !== 0
    || !Number.isSafeInteger(block) || block < 0 || block >= blocks) {
    throw new RangeError("stacked parameter block is out of range");
  }
  const size = elements / blocks;
  return [block * size, size];
}

export class FileTensorStore {
  readonly manifestPath: string;
  readonly manifest: BinaryTensorManifest;
  readonly #directory: string;
  readonly #cache = new Map<string, Promise<Float32Array>>();
  readonly #fileCache = new Map<string, Promise<Buffer>>();
  readonly #loaded = new Map<string, Buffer>();

  private constructor(manifestPath: string, manifest: BinaryTensorManifest) {
    this.manifestPath = manifestPath;
    this.manifest = manifest;
    this.#directory = dirname(manifestPath);
  }

  static async open(manifestPath: string): Promise<FileTensorStore> {
    const absolutePath = resolve(manifestPath);
    const manifest = JSON.parse(await readFile(absolutePath, "utf8")) as BinaryTensorManifest;
    if (manifest.tensors === undefined) throw new Error(`${absolutePath} has no tensor table`);
    return new FileTensorStore(absolutePath, manifest);
  }

  tensor(name: string): Promise<Float32Array> {
    let pending = this.#cache.get(name);
    if (pending === undefined) {
      pending = this.#load(name);
      this.#cache.set(name, pending);
    }
    return pending;
  }

  shape(name: string): readonly number[] {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`manifest contains no tensor named ${name}`);
    return record.shape;
  }

  /** Load and retain the shard holding `name`, so `read` can decode it synchronously. */
  async ensureLoaded(name: string): Promise<void> {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`manifest contains no tensor named ${name}`);
    if (this.#loaded.has(record.file)) return;
    this.#loaded.set(record.file, await this.#file(record.file));
  }

  /**
   * Decode `name`, or block `block` of its `blocks` stacked blocks, from the
   * retained shard. Float32 tensors are views; compressed ones decode on each
   * call, which keeps the model in its stored form between uses.
   */
  read(name: string, block?: number, blocks?: number): Float32Array {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`manifest contains no tensor named ${name}`);
    const bytes = this.#loaded.get(record.file);
    if (bytes === undefined) throw new Error(`${name} has not been loaded; call ensureLoaded first`);
    const elements = tensorElements(record);
    const [start, count] = blockRange(elements, block, blocks);
    return readTensorRange(record, bytes.buffer, bytes.byteOffset + (record.byteOffset ?? 0), start, count);
  }

  #file(file: string): Promise<Buffer> {
    let pendingFile = this.#fileCache.get(file);
    if (pendingFile === undefined) {
      pendingFile = readFile(resolve(this.#directory, file));
      this.#fileCache.set(file, pendingFile);
    }
    return pendingFile;
  }

  async #load(name: string): Promise<Float32Array> {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`manifest contains no tensor named ${name}`);
    const bytes = await this.#file(record.file);
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = tensorByteLength(record);
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > bytes.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    // Return read-only-by-convention f32 views into the cached shard, matching
    // the browser store and avoiding hundreds of duplicate full-model buffers.
    return readTensor(record, bytes.buffer, bytes.byteOffset + byteOffset);
  }
}
