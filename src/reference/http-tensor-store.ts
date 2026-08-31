import type { BinaryTensorManifest, BinaryTensorShard } from "./tensor-store.js";
import { readTensor, tensorByteLength, tensorElements } from "./dtype.js";

const MAX_CONCURRENT_DOWNLOADS = 8;
const MAX_MODEL_BYTES = 1024 ** 3;
const CACHE_NAME = "afwebgpu-model-shards";
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
  let lastError: unknown;
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      lastStatus = response.status;
      lastError = new Error(`failed to load ${label}: HTTP ${response.status}`);
      if (!RETRYABLE_STATUS.has(response.status) || attempt === 4) break;
      await response.body?.cancel();
    } catch (error) {
      lastError = error;
      if (attempt === 4) break;
    }
    await delay(250 * 2 ** attempt);
  }
  if (lastError instanceof Error) throw lastError;
  throw new Error(`failed to load ${label}: HTTP ${lastStatus}`);
}

function validateManifest(manifest: BinaryTensorManifest): void {
  if (manifest.tensors === undefined || manifest.tensors === null || typeof manifest.tensors !== "object") {
    throw new Error("model manifest has no tensor table");
  }
  const records = Object.entries(manifest.tensors);
  if (records.length === 0 || records.length > 10_000) throw new Error("model manifest has an invalid tensor count");
  let totalBytes = 0;
  for (const [name, record] of records) {
    if (record === null || !["float32", "float16", "int8"].includes(record.dtype)
      || typeof record.file !== "string" || record.file === "") {
      throw new Error(`model tensor ${name} has an invalid record`);
    }
    if (!Array.isArray(record.shape) || !record.shape.every(
      (value) => Number.isSafeInteger(value) && value > 0,
    )) throw new Error(`model tensor ${name} has an invalid shape`);
    const elements = tensorElements(record);
    const bytes = tensorByteLength(record);
    const offset = record.byteOffset ?? 0;
    const alignment = record.dtype === "float32" ? 4 : record.dtype === "float16" ? 2 : 1;
    if (!Number.isSafeInteger(bytes) || !Number.isSafeInteger(offset) || offset < 0 || offset % alignment !== 0) {
      throw new Error(`model tensor ${name} has an invalid byte range`);
    }
    totalBytes += elements * 4;
    if (!Number.isSafeInteger(totalBytes) || totalBytes > MAX_MODEL_BYTES) {
      throw new Error("model manifest exceeds the 1 GiB browser model limit");
    }
  }
  const files = manifest.bundle?.files;
  if (files !== undefined && (manifest.bundle?.version !== 1 || manifest.bundle.id === undefined)) {
    throw new Error("model manifest uses an unsupported versioned bundle format");
  }
  if (files !== undefined) for (const shard of files) {
    if (typeof shard.file !== "string" || shard.file === ""
      || !Number.isSafeInteger(shard.bytes) || shard.bytes <= 0 || shard.bytes > MAX_MODEL_BYTES
      || (shard.sha256 !== undefined && !/^[0-9a-f]{64}$/.test(shard.sha256))) {
      throw new Error("model manifest contains invalid shard metadata");
    }
  }
}

async function persistentCache(): Promise<Cache | undefined> {
  if (globalThis.caches === undefined) return undefined;
  try { return await globalThis.caches.open(CACHE_NAME); } catch { return undefined; }
}

/** Lazy browser tensor store backed by a JSON manifest and fetchable float32 files. */
export class HttpTensorStore {
  readonly manifestUrl: URL;
  readonly manifest: BinaryTensorManifest;
  readonly #cache = new Map<string, Promise<Float32Array>>();
  readonly #fileCache = new Map<string, Promise<ArrayBuffer>>();
  readonly #fileByteLengths = new Map<string, number>();
  readonly #shards = new Map<string, BinaryTensorShard>();
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
    const tensorBytes = records.reduce(
      (sum, record) => sum + tensorByteLength(record), 0,
    );
    this.#totalBytes = manifest.bundle?.files?.reduce((sum, file) => sum + file.bytes, 0) ?? tensorBytes;
    for (const record of records) {
      const end = (record.byteOffset ?? 0) + tensorByteLength(record);
      this.#fileByteLengths.set(record.file, Math.max(this.#fileByteLengths.get(record.file) ?? 0, end));
    }
    for (const shard of manifest.bundle?.files ?? []) this.#shards.set(shard.file, shard);
  }
  static async open(manifestUrlValue: URL | string,
    onProgress?: TensorDownloadProgressCallback): Promise<HttpTensorStore> {
    const manifestUrl = typeof manifestUrlValue === "string" ? new URL(manifestUrlValue, location.href) : manifestUrlValue;
    const response = await fetchWithRetry(manifestUrl, "model manifest");
    const manifest = await response.json() as BinaryTensorManifest;
    validateManifest(manifest);
    const store = new HttpTensorStore(manifestUrl, manifest, onProgress);
    store.#reportProgress();
    return store;
  }
  static async clearPersistentCache(): Promise<boolean> {
    if (globalThis.caches === undefined) return false;
    try { return await globalThis.caches.delete(CACHE_NAME); } catch { return false; }
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
    if (record === undefined) throw new Error(`missing tensor ${name}`);
    let pendingFile = this.#fileCache.get(record.file);
    if (pendingFile === undefined) {
      pendingFile = this.#scheduleDownload(record.file, name);
      this.#fileCache.set(record.file, pendingFile);
    }
    const buffer = await pendingFile;
    const byteOffset = record.byteOffset ?? 0;
    const byteLength = tensorByteLength(record);
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
      throw new Error(`${name} points outside ${record.file}`);
    }
    this.#loadedTensors += 1;
    this.#reportProgress(name);
    return readTensor(record, buffer, byteOffset);
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
    const url = new URL(file, this.manifestUrl);
    const expectedLength = this.#fileByteLengths.get(file);
    if (expectedLength === undefined) throw new Error(`${file} is absent from the manifest`);
    const shard = this.#shards.get(file);
    if (shard !== undefined && shard.bytes !== expectedLength) {
      throw new Error(`${file} metadata has an invalid byte length`);
    }
    // Legacy manifests used mutable shard names. Only persist explicitly
    // versioned/declared files so an old URL can never pin updated weights.
    const cache = shard === undefined ? undefined : await persistentCache();
    const cached = await cache?.match(url.href);
    if (cached !== undefined) {
      const buffer = await cached.arrayBuffer();
      try {
        await this.#verifyFile(file, buffer, expectedLength, shard);
        this.#loadedBytes += buffer.byteLength;
        this.#reportProgress();
        return buffer;
      } catch {
        await cache?.delete(url.href);
      }
    }
    const response = await fetchWithRetry(url, `tensor ${tensorName}`);
    let buffer: ArrayBuffer;
    if (response.body === null) {
      buffer = await response.arrayBuffer();
      this.#loadedBytes += buffer.byteLength;
      this.#reportProgress();
    } else {
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
      buffer = output.buffer;
    }
    await this.#verifyFile(file, buffer, expectedLength, shard);
    if (cache !== undefined) {
      try { await cache.put(url.href, new Response(buffer.slice(0))); } catch { /* quota or private mode */ }
    }
    return buffer;
  }
  async #verifyFile(
    file: string, buffer: ArrayBuffer, expectedLength: number, shard?: BinaryTensorShard,
  ): Promise<void> {
    if (buffer.byteLength !== expectedLength) throw new Error(`${file} has an invalid byte length`);
    if (shard?.sha256 !== undefined) {
      if (globalThis.crypto?.subtle === undefined) throw new Error("SHA-256 verification is unavailable");
      const digest = new Uint8Array(await globalThis.crypto.subtle.digest("SHA-256", buffer));
      const actual = [...digest].map((value) => value.toString(16).padStart(2, "0")).join("");
      if (actual !== shard.sha256) throw new Error(`${file} has an invalid SHA-256 digest`);
    }
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
