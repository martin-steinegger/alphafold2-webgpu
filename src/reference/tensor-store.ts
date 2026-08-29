import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

export interface BinaryTensorRecord {
  readonly file: string;
  readonly shape: readonly number[];
  readonly dtype: "float32";
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
    const bytes = await readFile(resolve(this.#directory, record.file));
    const elements = record.shape.reduce((product, value) => product * value, 1);
    if (bytes.byteLength !== elements * 4) {
      throw new Error(`${name} has ${bytes.byteLength} bytes; expected ${elements * 4}`);
    }
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  }
}

