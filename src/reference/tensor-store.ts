import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

export interface BinaryTensorRecord {
  readonly file: string;
  readonly shape: readonly number[];
  readonly dtype: "float32";
  readonly byteOffset?: number;
  readonly [metadata: string]: unknown;
}

export interface BinaryTensorManifest {
  readonly tensors: Readonly<Record<string, BinaryTensorRecord>>;
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
    if (record === undefined || record.dtype !== "float32") {
      throw new Error(`manifest contains no float32 tensor named ${name}`);
    }
    let pendingFile = this.#fileCache.get(record.file);
    if (pendingFile === undefined) {
      pendingFile = readFile(resolve(this.#directory, record.file));
      this.#fileCache.set(record.file, pendingFile);
    }
    const bytes = await pendingFile;
    const elements = record.shape.reduce((product, value) => product * value, 1);
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = elements * 4;
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > bytes.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    return new Float32Array(bytes.buffer.slice(
      bytes.byteOffset + byteOffset, bytes.byteOffset + byteOffset + byteLength,
    ));
  }
}
