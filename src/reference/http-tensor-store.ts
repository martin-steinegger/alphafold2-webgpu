import type { BinaryTensorManifest } from "./tensor-store.js";

const MAX_CONCURRENT_DOWNLOADS = 8;
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);

export interface TensorDownloadProgress {
  readonly loadedBytes: number;
  readonly totalBytes: number;
  readonly loadedTensors: number;
  readonly totalTensors: number;
  readonly tensorName?: string;
}

export type TensorDownloadProgressCallback = (progress: TensorDownloadProgress) => void;

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
  readonly #fileCache = new Map<string, Promise<ArrayBuffer>>();
  readonly #pending: (() => void)[] = [];
  readonly #onProgress: TensorDownloadProgressCallback | undefined;
  readonly #totalBytes: number;
  readonly #totalTensors: number;
  #activeDownloads = 0;
  #loadedBytes = 0;
  #loadedTensors = 0;
  private constructor(manifestUrl: URL, manifest: BinaryTensorManifest, onProgress?: TensorDownloadProgressCallback) {
    this.manifestUrl = manifestUrl; this.manifest = manifest; this.#onProgress = onProgress;
    const records = Object.values(manifest.tensors);
    this.#totalTensors = records.length;
    this.#totalBytes = records.reduce(
      (sum, record) => sum + record.shape.reduce((product, dimension) => product * dimension, 1) * 4, 0,
    );
  }
  static async open(manifestUrlValue: URL | string,
    onProgress?: TensorDownloadProgressCallback): Promise<HttpTensorStore> {
    const manifestUrl = typeof manifestUrlValue === "string" ? new URL(manifestUrlValue, location.href) : manifestUrlValue;
    const response = await fetchWithRetry(manifestUrl, "model manifest");
    const manifest = await response.json() as BinaryTensorManifest;
    if (manifest.tensors === undefined) throw new Error("model manifest has no tensor table");
    const store = new HttpTensorStore(manifestUrl, manifest, onProgress);
    store.#reportProgress();
    return store;
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
    let pendingFile = this.#fileCache.get(record.file);
    if (pendingFile === undefined) {
      pendingFile = this.#scheduleDownload(record.file, name);
      this.#fileCache.set(record.file, pendingFile);
    }
    const buffer = await pendingFile;
    const elements = record.shape.reduce((product, value) => product * value, 1);
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = elements * 4;
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    this.#loadedTensors += 1;
    this.#reportProgress(name);
    return new Float32Array(buffer, byteOffset, elements);
  }
  async #scheduleDownload(file: string, tensorName: string): Promise<ArrayBuffer> {
    return new Promise<ArrayBuffer>((resolve, reject) => {
      const start = (): void => {
        this.#activeDownloads += 1;
        void this.#downloadFile(file, tensorName).then(resolve, reject).finally(() => {
          this.#activeDownloads -= 1;
          this.#pending.shift()?.();
        });
      };
      if (this.#activeDownloads < MAX_CONCURRENT_DOWNLOADS) start();
      else this.#pending.push(start);
    });
  }
  async #downloadFile(file: string, tensorName: string): Promise<ArrayBuffer> {
    const response = await fetchWithRetry(new URL(file, this.manifestUrl), `tensor ${tensorName}`);
    const contentLength = response.headers.get("content-length");
    const expectedLength = contentLength === null ? Number.NaN : Number(contentLength);
    if (response.body === null || !Number.isSafeInteger(expectedLength) || expectedLength < 0) {
      const buffer = await response.arrayBuffer();
      this.#loadedBytes += buffer.byteLength;
      this.#reportProgress();
      return buffer;
    }
    const output = new Uint8Array(expectedLength);
    const reader = response.body.getReader();
    let offset = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (offset + value.byteLength > output.byteLength) throw new Error(`${file} exceeds its content length`);
      output.set(value, offset);
      offset += value.byteLength;
      this.#loadedBytes += value.byteLength;
      this.#reportProgress();
    }
    if (offset !== output.byteLength) throw new Error(`${file} has an invalid byte length`);
    return output.buffer;
  }
  #reportProgress(tensorName?: string): void {
    const progress = {
      loadedBytes: this.#loadedBytes, totalBytes: this.#totalBytes,
      loadedTensors: this.#loadedTensors, totalTensors: this.#totalTensors,
      ...(tensorName === undefined ? {} : { tensorName }),
    };
    this.#onProgress?.(progress);
  }
}
