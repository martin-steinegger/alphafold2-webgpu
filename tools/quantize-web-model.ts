import { copyFile, mkdir, open, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, relative, resolve } from "node:path";
import { float16ToNumber } from "../src/reference/dtype.js";
import { numberToFloat16 } from "../src/runtime/float16.js";
import type {
  BinaryTensorManifest, BinaryTensorRecord, BinaryTensorShard,
} from "../src/reference/tensor-store.js";

const BLOCK = 64;
const format = process.argv.find((value) => value.startsWith("--format="))?.slice(9) ?? "int8";
if (format !== "int8" && format !== "float16") throw new Error("--format must be int8 or float16");
const positional = process.argv.slice(2).filter((value) => !value.startsWith("--"));
const sourceDirectory = resolve(positional[0] ?? "dist/web/model-f32");
const outputDirectory = resolve(positional[1] ?? `dist/web/model-${format === "int8" ? "q8" : "f16"}`);
if (sourceDirectory === outputDirectory || outputDirectory === resolve("/")) throw new Error("unsafe output directory");

const manifest = JSON.parse(await readFile(resolve(sourceDirectory, "manifest.json"), "utf8")) as BinaryTensorManifest;
if (Object.values(manifest.tensors).some((record) => record.dtype !== "float32")) {
  throw new Error("quantization source must contain only float32 tensors");
}

function collectTensorNames(value: unknown, known: Readonly<Record<string, BinaryTensorRecord>>, into: Set<string>): void {
  if (typeof value === "string" && known[value] !== undefined) into.add(value);
  else if (Array.isArray(value)) value.forEach((item) => collectTensorNames(item, known, into));
  else if (value !== null && typeof value === "object") {
    Object.values(value).forEach((item) => collectTensorNames(item, known, into));
  }
}

// Rigid-frame composition and literal residue geometry are deliberately kept
// exact. PAE bin boundaries are constants rather than learned parameters.
const keepFloat32 = new Set<string>();
collectTensorNames(manifest.structureModule, manifest.tensors, keepFloat32);
collectTensorNames(manifest.residueGeometry, manifest.tensors, keepFloat32);
for (const name of Object.keys(manifest.tensors)) {
  if (name.startsWith("geometry") || name === "confidencePaeBreaks") keepFloat32.add(name);
}

const byShard = new Map<string, { name: string; record: BinaryTensorRecord }[]>();
for (const [name, record] of Object.entries(manifest.tensors)) {
  const entries = byShard.get(record.file) ?? [];
  entries.push({ name, record });
  byShard.set(record.file, entries);
}
for (const entries of byShard.values()) entries.sort(
  (left, right) => (left.record.byteOffset ?? 0) - (right.record.byteOffset ?? 0),
);

await mkdir(outputDirectory, { recursive: true });
const tensors: Record<string, BinaryTensorRecord> = {};
const files: BinaryTensorShard[] = [];
const counts = { float32: 0, float16: 0, int8: 0 };

for (const [sourceFile, entries] of [...byShard].sort(([left], [right]) => left.localeCompare(right))) {
  const sourcePath = resolve(sourceDirectory, sourceFile);
  if (relative(sourceDirectory, sourcePath).startsWith("..")) throw new Error(`${sourceFile} escapes the source directory`);
  const source = await readFile(sourcePath);
  const suffix = format === "int8" ? "q8" : "f16";
  const targetFile = sourceFile.replace(/(?:\.v\d+)?\.f32\.bin$/, `.v1.${suffix}.bin`);
  if (targetFile === sourceFile) throw new Error(`cannot derive a versioned target name from ${sourceFile}`);
  const targetPath = resolve(outputDirectory, targetFile);
  if (relative(outputDirectory, targetPath).startsWith("..")) throw new Error(`${targetFile} escapes the output directory`);
  await mkdir(dirname(targetPath), { recursive: true });
  const handle = await open(targetPath, "w");
  let position = 0;
  const write = async (value: ArrayBufferView): Promise<void> => {
    const bytes = new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
    await handle.write(bytes, 0, bytes.byteLength, position);
    position += bytes.byteLength;
  };
  const align = async (width: number): Promise<void> => {
    const padding = (width - position % width) % width;
    if (padding !== 0) await write(new Uint8Array(padding));
  };
  try {
    for (const { name, record } of entries) {
      await align(4);
      const elements = record.shape.reduce((product, dimension) => product * dimension, 1);
      const sourceOffset = source.byteOffset + (record.byteOffset ?? 0);
      if (sourceOffset % 4 !== 0 || sourceOffset + elements * 4 > source.byteOffset + source.byteLength) {
        throw new Error(`${name} points outside ${sourceFile}`);
      }
      const values = new Float32Array(source.buffer, sourceOffset, elements);
      if (values.some((value) => !Number.isFinite(value))) throw new Error(`${name} contains a non-finite value`);
      const { block: _block, scaleOffset: _scaleOffset, ...sourceRecord } = record;
      const base = { ...sourceRecord, file: targetFile, byteOffset: position };
      if (keepFloat32.has(name)) {
        await write(values);
        tensors[name] = { ...base, dtype: "float32" };
        counts.float32 += 1;
      } else if (format === "float16") {
        const half = new Uint16Array(elements);
        for (let index = 0; index < elements; index += 1) half[index] = numberToFloat16(values[index]!);
        await write(half);
        tensors[name] = { ...base, dtype: "float16" };
        counts.float16 += 1;
      } else {
        const codes = new Int8Array(elements);
        const blockCount = Math.ceil(elements / BLOCK);
        const scales = new Uint16Array(blockCount);
        for (let blockIndex = 0; blockIndex < blockCount; blockIndex += 1) {
          const start = blockIndex * BLOCK;
          const end = Math.min(elements, start + BLOCK);
          let maximum = 0;
          for (let index = start; index < end; index += 1) maximum = Math.max(maximum, Math.abs(values[index]!));
          const scaleBits = numberToFloat16(maximum / 127);
          const scale = float16ToNumber(scaleBits) || 1;
          scales[blockIndex] = scaleBits;
          for (let index = start; index < end; index += 1) {
            codes[index] = Math.max(-127, Math.min(127, Math.round(values[index]! / scale)));
          }
        }
        await write(codes);
        await align(2);
        const scaleOffset = position;
        await write(scales);
        tensors[name] = { ...base, dtype: "int8", block: BLOCK, scaleOffset };
        counts.int8 += 1;
      }
    }
  } finally {
    await handle.close();
  }
  const digest = createHash("sha256").update(await readFile(targetPath)).digest("hex");
  files.push({ file: targetFile, bytes: position, sha256: digest });
}

const bytes = files.reduce((sum, file) => sum + file.bytes, 0);
const encoding = format === "int8" ? "mixed-q8-f16-scale-le" : "mixed-f16-le";
const modelId = typeof manifest.bundle?.model === "string" && manifest.bundle.model !== ""
  ? manifest.bundle.model : "alphafold-model";
const id = `${modelId}-${format === "int8" ? "q8" : "f16"}-v1`;
const output: BinaryTensorManifest = {
  ...manifest,
  ...((manifest as { readonly weightsLicense?: unknown }).weightsLicense === undefined ? {} : {
    weightsLicense: {
      ...(manifest as unknown as { readonly weightsLicense: Record<string, unknown> }).weightsLicense,
      modified: true,
      modifications: ["repacked into versioned browser shards", `converted learned tensors to ${encoding}`],
    },
  }),
  bundle: {
    ...manifest.bundle, version: 1, id, encoding,
    tensors: Object.keys(tensors).length, bytes, shards: files.length, files,
    ...(format === "int8" ? { quantization: { scheme: "symmetric-per-block", bits: 8, block: BLOCK } } : {}),
  },
  tensors,
};
const license = (manifest as { readonly weightsLicense?: { readonly file?: unknown } }).weightsLicense;
if (typeof license?.file === "string" && license.file !== "") {
  await copyFile(resolve(sourceDirectory, license.file), resolve(outputDirectory, license.file));
}
await writeFile(resolve(outputDirectory, "manifest.json"), `${JSON.stringify(output, null, 2)}\n`);
console.log(`Wrote ${id}: ${counts.int8} int8, ${counts.float16} float16, ${counts.float32} float32 tensors; ${(bytes / 2 ** 20).toFixed(1)} MiB`);
