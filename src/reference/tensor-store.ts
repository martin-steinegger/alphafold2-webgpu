import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { readTensor, tensorByteLength } from "./dtype.js";

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

export class FileTensorStore {
  readonly manifestPath: string;
  readonly manifest: BinaryTensorManifest;
  readonly #directory: string;
  readonly #cache = new Map<string, Promise<Float32Array>>();
  readonly #fileCache = new Map<string, Promise<Buffer>>();

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

  async #load(name: string): Promise<Float32Array> {
    const record = this.manifest.tensors[name];
    if (record === undefined) throw new Error(`manifest contains no tensor named ${name}`);
    let pendingFile = this.#fileCache.get(record.file);
    if (pendingFile === undefined) {
      pendingFile = readFile(resolve(this.#directory, record.file));
      this.#fileCache.set(record.file, pendingFile);
    }
    const bytes = await pendingFile;
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
