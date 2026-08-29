import { float32ToFloat16Array } from "../runtime/float16.js";
import type { Precision, TriangleMultiplicationWeights, TriangleShape } from "./types.js";

const ORDER = [
  "layerNormInWeight", "layerNormInBias",
  "linearAPWeight", "linearAPBias", "linearAGWeight", "linearAGBias",
  "linearBPWeight", "linearBPBias", "linearBGWeight", "linearBGBias",
  "layerNormOutWeight", "layerNormOutBias",
  "linearZWeight", "linearZBias", "linearGWeight", "linearGBias",
] as const satisfies readonly (keyof TriangleMultiplicationWeights)[];

export type WeightName = (typeof ORDER)[number];
export type WeightOffsets = Readonly<Record<WeightName, number>>;

export interface PackedWeights {
  readonly data: Float32Array | Uint16Array;
  readonly offsets: WeightOffsets;
}

export function packWeights(weights: TriangleMultiplicationWeights, precision: Precision): PackedWeights {
  const offsets = {} as Record<WeightName, number>;
  let elementCount = 0;
  for (const name of ORDER) {
    offsets[name] = elementCount;
    elementCount += weights[name].length;
  }

  const f32 = new Float32Array(elementCount);
  for (const name of ORDER) f32.set(weights[name], offsets[name]);
  return {
    data: precision === "f16" ? float32ToFloat16Array(f32) : f32,
    offsets,
  };
}

export function expectedWeightElementCount(shape: TriangleShape): number {
  const { cZ, cHidden } = shape;
  return 2 * cZ + 4 * (cHidden * cZ + cHidden) + 2 * cHidden
    + (cZ * cHidden + cZ) + (cZ * cZ + cZ);
}

