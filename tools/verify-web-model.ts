import { readFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import type { BinaryTensorManifest } from "../src/reference/tensor-store.js";

const manifestPath = resolve(process.argv[2] ?? "dist/web/model/manifest.json");
const directory = dirname(manifestPath);
const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as BinaryTensorManifest;
if (manifest.bundle?.version !== 1 || manifest.bundle.id === undefined || manifest.bundle.files === undefined) {
  throw new Error("model bundle lacks versioned shard metadata");
}
const files = new Map(manifest.bundle.files.map((file) => [file.file, file]));
for (const shard of files.values()) {
  const path = resolve(directory, shard.file);
  if (relative(directory, path).startsWith("..")) throw new Error(`${shard.file} escapes the model directory`);
  const bytes = await readFile(path);
  if (bytes.byteLength !== shard.bytes) throw new Error(`${shard.file} has an invalid byte length`);
}
for (const [name, tensor] of Object.entries(manifest.tensors)) {
  const shard = files.get(tensor.file);
  if (shard === undefined) throw new Error(`${name} points to undeclared shard ${tensor.file}`);
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  const end = (tensor.byteOffset ?? 0) + elements * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(end) || end > shard.bytes) throw new Error(`${name} points outside ${tensor.file}`);
}
console.log(`Validated ${manifest.bundle.id}: ${files.size} shards, ${Object.keys(manifest.tensors).length} tensors`);
