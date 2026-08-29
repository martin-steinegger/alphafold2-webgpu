import type { BinaryTensorManifest } from "./tensor-store.js";

/** Lazy browser tensor store backed by a JSON manifest and fetchable float32 files. */
export class HttpTensorStore {
  readonly manifestUrl: URL;
  readonly manifest: BinaryTensorManifest;
  readonly #cache = new Map<string, Promise<Float32Array>>();
  private constructor(manifestUrl: URL, manifest: BinaryTensorManifest) {
    this.manifestUrl = manifestUrl; this.manifest = manifest;
  }
  static async open(manifestUrlValue: URL | string): Promise<HttpTensorStore> {
    const manifestUrl = typeof manifestUrlValue === "string" ? new URL(manifestUrlValue, location.href) : manifestUrlValue;
    const response = await fetch(manifestUrl);
    if (!response.ok) throw new Error(`failed to load model manifest: ${response.status}`);
    const manifest = await response.json() as BinaryTensorManifest;
    if (manifest.tensors === undefined) throw new Error("model manifest has no tensor table");
    return new HttpTensorStore(manifestUrl, manifest);
  }
  tensor(name: string): Promise<Float32Array> {
    let value = this.#cache.get(name);
    if (value === undefined) { value = this.#load(name); this.#cache.set(name, value); }
    return value;
  }
  shape(name: string): readonly number[] {
    const record = this.manifest.tensors[name]; if (record === undefined) throw new Error(`missing tensor ${name}`);
    return record.shape;
  }
  async #load(name: string): Promise<Float32Array> {
    const record = this.manifest.tensors[name];
    if (record === undefined || record.dtype !== "float32") throw new Error(`missing float32 tensor ${name}`);
    const response = await fetch(new URL(record.file, this.manifestUrl));
    if (!response.ok) throw new Error(`failed to load tensor ${name}: ${response.status}`);
    const buffer = await response.arrayBuffer();
    const elements = record.shape.reduce((product, value) => product * value, 1);
    if (buffer.byteLength !== elements * 4) throw new Error(`${name} has an invalid byte length`);
    return new Float32Array(buffer);
  }
}
