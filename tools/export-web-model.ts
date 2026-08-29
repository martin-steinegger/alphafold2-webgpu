import { copyFile, mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, relative, resolve } from "node:path";

interface TensorRecord { readonly file: string; readonly dtype: string; readonly shape: readonly number[]; }
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

const tensors: Record<string, TensorRecord> = {};
let bytes = 0;
await mkdir(outputDirectory, { recursive: true });
for (const name of [...names].sort()) {
  const record = manifest.tensors[name];
  if (record === undefined) throw new Error(`required tensor ${name} is missing`);
  const source = resolve(sourceDirectory, record.file);
  if (relative(sourceDirectory, source).startsWith("..")) throw new Error(`tensor ${name} escapes the source directory`);
  const destination = resolve(outputDirectory, record.file);
  if (relative(outputDirectory, destination).startsWith("..")) throw new Error(`tensor ${name} escapes the output directory`);
  await mkdir(dirname(destination), { recursive: true });
  await copyFile(source, destination);
  tensors[name] = record;
  bytes += record.shape.reduce((product, dimension) => product * dimension, 1) * 4;
}
reduced.tensors = tensors;
(reduced.bundle as Record<string, unknown>).tensors = names.size;
(reduced.bundle as Record<string, unknown>).bytes = bytes;
await writeFile(resolve(outputDirectory, "manifest.json"), `${JSON.stringify(reduced, null, 2)}\n`);
console.log(`Exported model_1_ptm: ${names.size} tensors, ${(bytes / 1024 / 1024).toFixed(1)} MiB`);
