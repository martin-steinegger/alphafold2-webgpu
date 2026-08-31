import { afterEach, describe, expect, it, vi } from "vitest";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";

afterEach(() => vi.unstubAllGlobals());

describe("HttpTensorStore", () => {
  it("bounds concurrent tensor downloads", async () => {
    const tensors = Object.fromEntries(Array.from({ length: 20 }, (_, index) => [`tensor${index}`, {
      file: `tensor${index}.f32.bin`, dtype: "float32", shape: [1],
    }]));
    let active = 0; let maximum = 0;
    vi.stubGlobal("fetch", vi.fn(async (input: URL | RequestInfo) => {
      const url = String(input);
      if (url.endsWith("manifest.json")) return new Response(JSON.stringify({ tensors }));
      active += 1; maximum = Math.max(maximum, active);
      await new Promise((resolve) => setTimeout(resolve, 2));
      active -= 1;
      return new Response(Float32Array.of(1));
    }));
    const progress: { loadedBytes: number; loadedTensors: number }[] = [];
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"), (value) => {
      progress.push({ loadedBytes: value.loadedBytes, loadedTensors: value.loadedTensors });
    });
    await Promise.all(Object.keys(tensors).map((name) => store.tensor(name)));
    expect(maximum).toBe(8);
    expect(progress.at(-1)).toEqual({ loadedBytes: 80, loadedTensors: 20 });
  });

  it("retries transient tensor responses", async () => {
    let attempts = 0;
    vi.stubGlobal("fetch", vi.fn(async (input: URL | RequestInfo) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify({
        tensors: { value: { file: "value.f32.bin", dtype: "float32", shape: [1] } },
      }));
      attempts += 1;
      return attempts === 1 ? new Response(null, { status: 503 }) : new Response(Float32Array.of(7));
    }));
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"));
    expect(Array.from(await store.tensor("value"))).toEqual([7]);
    expect(attempts).toBe(2);
  });

  it("downloads a shared shard once and returns tensor views at byte offsets", async () => {
    let downloads = 0;
    const shard = new Float32Array([1, 2, 3, 4]);
    vi.stubGlobal("fetch", vi.fn(async (input: URL | RequestInfo) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify({ tensors: {
        first: { file: "weights.bin", dtype: "float32", shape: [2], byteOffset: 0 },
        second: { file: "weights.bin", dtype: "float32", shape: [2], byteOffset: 8 },
      } }));
      downloads += 1;
      // CDNs may report the compressed transfer length while fetch exposes decoded bytes.
      return new Response(shard, { headers: { "content-encoding": "gzip", "content-length": "2" } });
    }));
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"));
    const [first, second] = await Promise.all([store.tensor("first"), store.tensor("second")]);
    expect(Array.from(first)).toEqual([1, 2]);
    expect(Array.from(second)).toEqual([3, 4]);
    expect(downloads).toBe(1);
  });

  it("decodes block-int8 tensors from a mixed model shard", async () => {
    const shard = new Uint8Array(10);
    new Int8Array(shard.buffer, 0, 5).set([1, -2, 3, -4, 5]);
    new Uint16Array(shard.buffer, 6, 2).set([0x3800, 0x4000]);
    vi.stubGlobal("fetch", vi.fn(async (input: URL | RequestInfo) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify({
        bundle: { version: 1, id: "q8-test", files: [{ file: "weights.v1.q8.bin", bytes: 10 }] },
        tensors: { value: {
          file: "weights.v1.q8.bin", dtype: "int8", shape: [5], byteOffset: 0,
          block: 4, scaleOffset: 6,
        } },
      }));
      return new Response(shard);
    }));
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"));
    expect([...await store.tensor("value")]).toEqual([0.5, -1, 1.5, -2, 10]);
  });

  it("validates versioned shards and reuses the persistent cache", async () => {
    const shard = Float32Array.of(3, 5);
    const manifest = { bundle: { version: 1, id: "test", files: [
      { file: "weights.v1.bin", bytes: shard.byteLength },
    ] }, tensors: { value: {
      file: "weights.v1.bin", dtype: "float32", shape: [2], byteOffset: 0,
    } } };
    const stored = new Map<string, Response>(); let shardDownloads = 0;
    vi.stubGlobal("caches", {
      open: vi.fn(async () => ({
        match: async (key: string) => stored.get(key)?.clone(),
        put: async (key: string, response: Response) => { stored.set(key, response.clone()); },
        delete: async (key: string) => stored.delete(key),
      })),
      delete: vi.fn(async () => { stored.clear(); return true; }),
    });
    vi.stubGlobal("fetch", vi.fn(async (input: URL | RequestInfo) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify(manifest));
      shardDownloads += 1; return new Response(shard);
    }));
    const url = new URL("https://example.test/model/manifest.json");
    expect(Array.from(await (await HttpTensorStore.open(url)).tensor("value"))).toEqual([3, 5]);
    expect(Array.from(await (await HttpTensorStore.open(url)).tensor("value"))).toEqual([3, 5]);
    expect(shardDownloads).toBe(1);
    expect(await HttpTensorStore.clearPersistentCache()).toBe(true);
  });

  it("rejects a shard that fails its declared byte length", async () => {
    const shard = Float32Array.of(7, 8);
    vi.stubGlobal("fetch", vi.fn(async (input: URL | RequestInfo) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify({
        bundle: { version: 1, id: "bad-length", files: [{ file: "weights.bin", bytes: 4 }] },
        tensors: { value: { file: "weights.bin", dtype: "float32", shape: [1] } },
      }));
      return new Response(shard);
    }));
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"));
    await expect(store.tensor("value")).rejects.toThrow(/exceeds its content length|invalid byte length/);
  });

  it("rejects same-length shard corruption using its declared SHA-256 digest", async () => {
    const expected = Float32Array.of(7, 8);
    const digestBytes = new Uint8Array(await crypto.subtle.digest("SHA-256", expected));
    const sha256 = [...digestBytes].map((value) => value.toString(16).padStart(2, "0")).join("");
    vi.stubGlobal("fetch", vi.fn(async (input: URL | RequestInfo) => {
      if (String(input).endsWith("manifest.json")) return new Response(JSON.stringify({
        bundle: { version: 1, id: "bad-digest", files: [
          { file: "weights.v1.bin", bytes: expected.byteLength, sha256 },
        ] },
        tensors: { value: { file: "weights.v1.bin", dtype: "float32", shape: [2] } },
      }));
      return new Response(Float32Array.of(7, 9));
    }));
    const store = await HttpTensorStore.open(new URL("https://example.test/model/manifest.json"));
    await expect(store.tensor("value")).rejects.toThrow(/invalid SHA-256 digest/);
  });
});
