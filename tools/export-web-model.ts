import { mkdir, open, readFile, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";

interface TensorRecord { readonly file: string; readonly dtype: string; readonly shape: readonly number[];
  readonly [metadata: string]: unknown; }
interface SourceManifest { readonly tensors: Readonly<Record<string, TensorRecord>>; readonly [key: string]: unknown; }

const [, , sourceValue = "test/fixtures/evoformer/model1-query-59-stack/manifest.json", outputValue = "dist/web/model"] = process.argv;
const sourceManifestPath = resolve(sourceValue);
const sourceDirectory = dirname(sourceManifestPath);
const outputDirectory = resolve(outputValue);
if (outputDirectory === sourceDirectory || outputDirectory === resolve("/")) throw new Error("unsafe output directory");

const manifest = JSON.parse(await readFile(sourceManifestPath, "utf8")) as SourceManifest;
const section = <T>(name: string): T => {
  const value = manifest[name]; if (value === undefined) throw new Error(`source manifest has no ${name}`); return value as T;
};
type Stack = { readonly blocks: number; readonly parameterFormat: unknown; readonly parameters: unknown };
type Structure = { readonly implementation: unknown; readonly dtype: unknown; readonly iterations: number; readonly parameters: unknown };
type Confidence = { readonly parameters: unknown };
const evoformer = section<Stack>("evoformerStack");
const extraMsa = section<Stack>("extraMsaStack");
const structure = section<Structure>("structureModule");
const confidence = section<Confidence>("confidenceHeads");
const reduced: Record<string, unknown> = {
  formatVersion: manifest.formatVersion,
  source: manifest.source,
  model: manifest.model,
  bundle: { purpose: "browser-inference", model: "model_1_ptm", encoding: "float32-le" },
  evoformerStack: { blocks: evoformer.blocks, parameterFormat: evoformer.parameterFormat, parameters: evoformer.parameters },
  extraMsaStack: { blocks: extraMsa.blocks, parameterFormat: extraMsa.parameterFormat, parameters: extraMsa.parameters },
  embedding: section("embedding"),
  templateEmbedding: section("templateEmbedding"),
  structureModule: { implementation: structure.implementation, dtype: structure.dtype, iterations: structure.iterations, parameters: structure.parameters },
  residueGeometry: section("residueGeometry"),
  confidenceHeads: { parameters: confidence.parameters },
};

const names = new Set([
  "geometryDefaultFrames", "geometryAtom14ToGroup", "geometryAtom14Positions", "geometryAtom14Mask",
  "geometryAtom37ToAtom14", "geometryAtom37Mask", "confidencePaeBreaks",
]);
function collect(value: unknown): void {
  if (typeof value === "string" && manifest.tensors[value] !== undefined) names.add(value);
  else if (Array.isArray(value)) value.forEach(collect);
  else if (value !== null && typeof value === "object") Object.values(value).forEach(collect);
}
collect(reduced);

const SHARDS = 8;
interface TensorEntry { readonly name: string; readonly record: TensorRecord; readonly source: string; readonly bytes: number; }
const entries: TensorEntry[] = [];
await mkdir(outputDirectory, { recursive: true });
for (const name of [...names].sort()) {
  const record = manifest.tensors[name];
  if (record === undefined) throw new Error(`required tensor ${name} is missing`);
  const source = resolve(sourceDirectory, record.file);
  if (relative(sourceDirectory, source).startsWith("..")) throw new Error(`tensor ${name} escapes the source directory`);
  entries.push({ name, record, source,
    bytes: record.shape.reduce((product, dimension) => product * dimension, 1) * 4 });
}
const shards = Array.from({ length: SHARDS }, (_, index) => ({ index, bytes: 0, entries: [] as TensorEntry[] }));
for (const entry of entries.sort((left, right) => right.bytes - left.bytes || left.name.localeCompare(right.name))) {
  const shard = shards.reduce((smallest, candidate) => candidate.bytes < smallest.bytes ? candidate : smallest);
  shard.entries.push(entry); shard.bytes += entry.bytes;
}
const tensors: Record<string, TensorRecord> = {};
const files: { file: string; bytes: number }[] = [];
for (const shard of shards) {
  const file = `weights-${String(shard.index).padStart(2, "0")}.v1.f32.bin`;
  const handle = await open(resolve(outputDirectory, file), "w");
  let byteOffset = 0;
  try {
    for (const entry of shard.entries) {
      const data = await readFile(entry.source);
      if (data.byteLength !== entry.bytes) throw new Error(`${entry.name} has an invalid byte length`);
      await handle.write(data, 0, data.byteLength, byteOffset);
      tensors[entry.name] = { ...entry.record, file, byteOffset };
      byteOffset += data.byteLength;
    }
  } finally { await handle.close(); }
  files.push({ file, bytes: byteOffset });
}
const bytes = entries.reduce((sum, entry) => sum + entry.bytes, 0);
reduced.tensors = tensors;
const bundle = reduced.bundle as Record<string, unknown>;
bundle.version = 1;
bundle.id = "model_1_ptm-f32-v1";
bundle.tensors = names.size;
bundle.bytes = bytes;
bundle.shards = SHARDS;
bundle.files = files;
await writeFile(resolve(outputDirectory, "manifest.json"), `${JSON.stringify(reduced, null, 2)}\n`);
console.log(`Exported model_1_ptm: ${names.size} tensors in ${SHARDS} shards, ${(bytes / 1024 / 1024).toFixed(1)} MiB`);
