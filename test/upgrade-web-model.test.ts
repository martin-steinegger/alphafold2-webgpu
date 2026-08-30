import { mkdtemp, readFile, rm, stat, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { upgradeWebModel } from "../tools/upgrade-web-model.js";

const temporaryDirectories: string[] = [];
afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe("legacy browser model upgrade", () => {
  it("versions shard names and adds persistent-cache metadata without changing bytes", async () => {
    const directory = await mkdtemp(join(tmpdir(), "afwebgpu-model-upgrade-"));
    temporaryDirectories.push(directory);
    const shard = new Uint8Array(Float32Array.of(1, 2).buffer);
    await writeFile(join(directory, "weights-00.f32.bin"), shard);
    const manifestPath = join(directory, "manifest.json");
    await writeFile(manifestPath, JSON.stringify({
      formatVersion: 1,
      bundle: { purpose: "browser-inference", model: "model_1_ptm", encoding: "float32-le" },
      tensors: {
        first: { file: "weights-00.f32.bin", dtype: "float32", shape: [1], byteOffset: 0 },
        second: { file: "weights-00.f32.bin", dtype: "float32", shape: [1], byteOffset: 4 },
      },
    }));

    expect(await upgradeWebModel(manifestPath)).toBe(true);
    const upgraded = JSON.parse(await readFile(manifestPath, "utf8")) as {
      bundle: { version: number; id: string; files: { file: string; bytes: number }[] };
      tensors: Record<string, { file: string }>;
    };
    expect(upgraded.bundle).toMatchObject({
      version: 1,
      id: "model_1_ptm-f32-v1",
      files: [{ file: "weights-00.v1.f32.bin", bytes: shard.byteLength }],
    });
    expect(upgraded.tensors.first?.file).toBe("weights-00.v1.f32.bin");
    expect(upgraded.tensors.second?.file).toBe("weights-00.v1.f32.bin");
    expect(new Uint8Array(await readFile(join(directory, "weights-00.v1.f32.bin")))).toEqual(shard);
    await expect(stat(join(directory, "weights-00.f32.bin"))).rejects.toThrow();
    expect(await upgradeWebModel(manifestPath)).toBe(false);
  });
});
