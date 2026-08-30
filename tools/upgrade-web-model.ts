import { appendFile, readFile, rename, stat, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import type { BinaryTensorManifest, BinaryTensorRecord } from "../src/reference/tensor-store.js";

export async function upgradeWebModel(manifestPathValue: string): Promise<boolean> {
  const manifestPath = resolve(manifestPathValue);
  const directory = dirname(manifestPath);
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as BinaryTensorManifest;
  if (manifest.bundle?.version === 1 && manifest.bundle.id !== undefined && manifest.bundle.files !== undefined) {
    return false;
  }
  if (manifest.tensors === undefined) throw new Error("legacy model manifest has no tensor table");
  const renamed = new Map<string, string>();
  for (const file of new Set(Object.values(manifest.tensors).map((tensor) => tensor.file))) {
    const versioned = file.endsWith(".f32.bin")
      ? `${file.slice(0, -".f32.bin".length)}.v1.f32.bin`
      : `${file}.v1`;
    const source = resolve(directory, file); const destination = resolve(directory, versioned);
    if (relative(directory, source).startsWith("..") || relative(directory, destination).startsWith("..")) {
      throw new Error(`${file} escapes the model directory`);
    }
    await rename(source, destination);
    renamed.set(file, versioned);
  }
  const tensors: Record<string, BinaryTensorRecord> = {};
  for (const [name, tensor] of Object.entries(manifest.tensors)) {
    const file = renamed.get(tensor.file);
    if (file === undefined) throw new Error(`no versioned file for tensor ${name}`);
    tensors[name] = { ...tensor, file };
  }
  const files = await Promise.all([...renamed.values()].sort().map(async (file) => ({
    file, bytes: (await stat(resolve(directory, file))).size,
  })));
  const bundle = {
    ...manifest.bundle,
    purpose: manifest.bundle?.purpose ?? "browser-inference",
    model: manifest.bundle?.model ?? "model_1_ptm",
    encoding: manifest.bundle?.encoding ?? "float32-le",
    version: 1,
    id: "model_1_ptm-f32-v1",
    tensors: Object.keys(tensors).length,
    bytes: files.reduce((sum, file) => sum + file.bytes, 0),
    shards: files.length,
    files,
  };
  await writeFile(manifestPath, `${JSON.stringify({ ...manifest, bundle, tensors }, null, 2)}\n`);
  return true;
}

if (process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const upgraded = await upgradeWebModel(process.argv[2] ?? "dist/web/model/manifest.json");
  console.log(upgraded ? "Upgraded legacy model bundle to model_1_ptm-f32-v1" : "Model bundle is already versioned");
  if (process.env.GITHUB_OUTPUT !== undefined) {
    await appendFile(process.env.GITHUB_OUTPUT, `upgraded=${String(upgraded)}\n`);
  }
}
