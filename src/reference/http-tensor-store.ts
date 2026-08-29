import type { BinaryTensorManifest } from "./tensor-store.js";

const MAX_CONCURRENT_DOWNLOADS = 8;
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function fetchWithRetry(url: URL, label: string): Promise<Response> {
  let lastStatus = 0;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    const response = await fetch(url);
    if (response.ok) return response;
    lastStatus = response.status;
    if (!RETRYABLE_STATUS.has(response.status) || attempt === 4) break;
    await response.body?.cancel();
    await delay(250 * 2 ** attempt);
  }
  throw new Error(`failed to load ${label}: ${lastStatus}`);
}

/** Lazy browser tensor store backed by a JSON manifest and fetchable float32 files. */
export class HttpTensorStore {
  readonly manifestUrl: URL;
  readonly manifest: BinaryTensorManifest;
  readonly #cache = new Map<string, Promise<Float32Array>>();
  readonly #pending: (() => void)[] = [];
  #activeDownloads = 0;
  private constructor(manifestUrl: URL, manifest: BinaryTensorManifest) {
    this.manifestUrl = manifestUrl; this.manifest = manifest;
  }
  static async open(manifestUrlValue: URL | string): Promise<HttpTensorStore> {
    const manifestUrl = typeof manifestUrlValue === "string" ? new URL(manifestUrlValue, location.href) : manifestUrlValue;
    const response = await fetchWithRetry(manifestUrl, "model manifest");
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
    return new Promise<Float32Array>((resolve, reject) => {
      const start = (): void => {
        this.#activeDownloads += 1;
        void this.#download(name).then(resolve, reject).finally(() => {
          this.#activeDownloads -= 1;
          this.#pending.shift()?.();
        });
      };
      if (this.#activeDownloads < MAX_CONCURRENT_DOWNLOADS) start();
      else this.#pending.push(start);
    });
  }
  async #download(name: string): Promise<Float32Array> {
    const record = this.manifest.tensors[name];
    if (record === undefined || record.dtype !== "float32") throw new Error(`missing float32 tensor ${name}`);
    const response = await fetchWithRetry(new URL(record.file, this.manifestUrl), `tensor ${name}`);
    const buffer = await response.arrayBuffer();
    const elements = record.shape.reduce((product, value) => product * value, 1);
    if (buffer.byteLength !== elements * 4) throw new Error(`${name} has an invalid byte length`);
    return new Float32Array(buffer);
  }
}
