import { readFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";
import type { BinaryTensorManifest } from "../src/reference/tensor-store.js";

const manifestPath = resolve(process.argv[2] ?? "dist/web/model/manifest.json");
const directory = dirname(manifestPath);
const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as BinaryTensorManifest;
if (manifest.tensors === undefined) throw new Error("model bundle has no tensor table");
const requiredBytes = new Map<string, number>();
for (const [name, tensor] of Object.entries(manifest.tensors)) {
  const elements = tensor.shape.reduce((product, dimension) => product * dimension, 1);
  const end = (tensor.byteOffset ?? 0) + elements * Float32Array.BYTES_PER_ELEMENT;
  if (!Number.isSafeInteger(end)) throw new Error(`${name} has an invalid byte range`);
  requiredBytes.set(tensor.file, Math.max(requiredBytes.get(tensor.file) ?? 0, end));
}
const declaredFiles = manifest.bundle?.files;
if (declaredFiles !== undefined && (manifest.bundle?.version !== 1 || manifest.bundle.id === undefined)) {
  throw new Error("model bundle has unsupported versioned shard metadata");
}
const files = new Map((declaredFiles ?? [...requiredBytes].map(([file, bytes]) => ({ file, bytes })))
  .map((file) => [file.file, file]));
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
const id = manifest.bundle?.id ?? "legacy model bundle";
console.log(`Validated ${id}: ${files.size} shards, ${Object.keys(manifest.tensors).length} tensors`);
