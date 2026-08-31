import { describe, expect, it } from "vitest";
import {
  MultimerMockTemplateGpu,
  type MultimerMockTemplateWeights,
} from "../src/evoformer/multimer-template.js";

const runner = new MultimerMockTemplateGpu({} as GPUDevice);
const weights = {} as MultimerMockTemplateWeights;

describe("Multimer mock-template public boundary", () => {
  it("rejects invalid lengths before allocating GPU buffers", async () => {
    await expect(runner.run(new Float32Array(), new Float32Array(), 0, weights))
      .rejects.toThrow(/length must be positive/);
  });

  it("validates pair shape and dtype", async () => {
    await expect(runner.run(new Float32Array(127), new Float32Array(1), 1, weights))
      .rejects.toThrow(/shape \[L, L, 128\]/);
    await expect(runner.run(
      new Float64Array(128) as unknown as Float32Array, new Float32Array(1), 1, weights,
    )).rejects.toThrow(/must be a Float32Array/);
  });

  it("validates the pair-mask shape", async () => {
    await expect(runner.run(new Float32Array(128), new Float32Array(), 1, weights))
      .rejects.toThrow(/mask must have shape \[L, L\]/);
  });
});
