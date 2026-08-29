import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import type { TriangleMultiplicationWeights } from "../triangle/types.js";
import type { TriangleReferenceBundle, TriangleReferenceManifest } from "./bundle.js";

const WEIGHT_NAMES = [
  "layerNormInWeight", "layerNormInBias", "linearAPWeight", "linearAPBias",
  "linearAGWeight", "linearAGBias", "linearBPWeight", "linearBPBias",
  "linearBGWeight", "linearBGBias", "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
] as const satisfies readonly (keyof TriangleMultiplicationWeights)[];

export async function loadTriangleReferenceBundleFromFiles(
  manifestPath: string,
): Promise<TriangleReferenceBundle> {
  const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as TriangleReferenceManifest;
  if (manifest.formatVersion !== 1 || manifest.operator !== "TriangleMultiplicationOutgoing") {
    throw new Error("unsupported triangle reference manifest");
  }
  const directory = dirname(manifestPath);
  const loadTensor = async (name: string): Promise<Float32Array> => {
    const record = manifest.tensors[name];
    if (record === undefined || record.dtype !== "float32") throw new Error(`missing float32 tensor ${name}`);
    const bytes = await readFile(resolve(directory, record.file));
    const expectedElements = record.shape.reduce((product, value) => product * value, 1);
    if (bytes.byteLength !== expectedElements * 4) throw new Error(`invalid byte length for tensor ${name}`);
    return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
  };
  const weights = {} as Record<keyof TriangleMultiplicationWeights, Float32Array>;
  await Promise.all(WEIGHT_NAMES.map(async (name) => { weights[name] = await loadTensor(name); }));
  return {
    source: manifest.source,
    input: {
      shape: manifest.shape,
      epsilon: manifest.epsilon,
      z: await loadTensor("z"),
      mask: await loadTensor("mask"),
      weights,
    },
    expected: await loadTensor("expected"),
  };
}
