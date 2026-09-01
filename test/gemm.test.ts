import { describe, expect, it } from "vitest";
import { createTiledGemmShader } from "../src/runtime/gemm.js";

describe("tiled GEMM shader", () => {
  it("does not race component writes to shared source vectors", () => {
    const shader = createTiledGemmShader({
      preamble: `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;`,
      rows: "64u",
      inner: "8u",
      columns: "128u",
      sourceElement: "source[row * 8u + k]",
      weightElement: "weight[k * 128u + column]",
      store: "output[row * 128u + column] = element;",
    });

    expect(shader).toContain("var<workgroup> gemm_source: array<f32, 512>;");
    expect(shader).toContain("gemm_source[slot] = element;");
    expect(shader).not.toMatch(/gemm_source\[[^\n]+\]\[[^\n]+\]\s*=/u);
  });
});
