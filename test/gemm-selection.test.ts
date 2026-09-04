import { describe, expect, it } from "vitest";
import {
  forceGemmVariant, gemmVariantCandidates, gemmVariantName, SHIPPABLE_GEMM_PRECISIONS,
} from "../src/runtime/gemm-selection.js";
import {
  createTiledGemmShader, GEMM_VARIANT_F32, gemmVariant, setGemmVariant, type GemmVariant,
} from "../src/runtime/gemm.js";

function fakeDevice(halfPrecision: boolean): GPUDevice {
  return {
    features: new Set<GPUFeatureName>(halfPrecision ? ["shader-f16" as GPUFeatureName] : []),
    createBuffer: () => { throw new Error("this test must not allocate"); },
  } as unknown as GPUDevice;
}

const spec = {
  preamble: `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;`,
  rows: "64u", inner: "256u", columns: "128u",
  sourceElement: "source[row * 256u + k]",
  weightElement: "weights[k * 128u + column]",
  store: "output[row * 128u + column] = element;",
} as const;

describe("projection variant selection", () => {
  it("offers half precision only to a device that reports it", () => {
    expect(gemmVariantCandidates(fakeDevice(false))
      .every((variant) => variant.precision === "f32")).toBe(true);
    const offered = new Set(gemmVariantCandidates(fakeDevice(true))
      .map((variant) => variant.precision));
    expect([...offered].sort()).toEqual(["f16-chunked", "f16-mixed", "f32"]);
  });

  it("never offers the arrangement that took a deep MSA to NaN", () => {
    // Pure f16 was the fastest candidate measured and it cannot be selected:
    // no probe cheap enough to run at device creation reaches the depth that
    // exposes it, so it is excluded here rather than by measurement.
    expect(SHIPPABLE_GEMM_PRECISIONS).not.toContain("f16");
    expect(gemmVariantCandidates(fakeDevice(true))
      .some((variant) => variant.precision === "f16")).toBe(false);
  });

  it("measures both k depths for every arrangement it offers", () => {
    const candidates = gemmVariantCandidates(fakeDevice(true));
    expect(candidates).toHaveLength(SHIPPABLE_GEMM_PRECISIONS.length * 2);
    expect(candidates.filter((variant) => variant.inner === 16)).toHaveLength(3);
    expect(gemmVariantName({ precision: "f16-chunked", inner: 16 }))
      .toBe("f16-chunked-64x128k16");
  });
});

describe("what a variant changes in the shader", () => {
  it("leaves the f32 kernel exact and reduces half precision in f32", () => {
    const of = (variant: GemmVariant): string => createTiledGemmShader(spec, variant);
    expect(of({ precision: "f32", inner: 8 })).not.toContain("f16");
    // The pure arrangement is the only one whose accumulator is half width.
    expect(of({ precision: "f16", inner: 8 })).toContain("var gemm_acc0 = vec4<f16>(0.0)");
    expect(of({ precision: "f16-mixed", inner: 8 })).toContain("var acc0 = vec4<f32>(0.0)");
    expect(of({ precision: "f16-chunked", inner: 8 })).toContain("var acc0 = vec4<f32>(0.0)");
    // Chunked keeps a per-tile half-precision sum and folds it once per tile.
    expect(of({ precision: "f16-chunked", inner: 8 })).toContain("var chunk0 = vec4<f16>(0.0)");
    expect(of({ precision: "f16-chunked", inner: 8 })).toContain("acc0 += vec4<f32>(chunk0)");
    expect(of({ precision: "f16-mixed", inner: 8 })).not.toContain("chunk0");
  });

  it("hands the epilogue a single-precision accumulator whatever the arithmetic", () => {
    // This is the contract that keeps the choice away from the call sites.
    const epilogue = { ...spec, epilogue: "  let first = acc0[0];", stageElements: 4 };
    for (const precision of ["f32", "f16", "f16-mixed", "f16-chunked"] as const) {
      const shader = createTiledGemmShader(epilogue, { precision, inner: 8 });
      expect(shader, precision).toContain("let first = acc0[0];");
      if (precision === "f16") {
        expect(shader, precision).toContain("let acc0 = vec4<f32>(gemm_acc0);");
      }
    }
  });

  it("steps a k tile back when it would not fit beside an epilogue's staging", () => {
    const depth = (code: string): number =>
      Number(/gemm_source: array<(?:f16|f32), (\d+)>/u.exec(code)![1]) / 64;
    // 2,048 staged f32 is 8 KiB, which leaves an f32 k16 tile no room in the
    // 16 KiB every implementation guarantees; half precision halves the
    // operands and still fits.
    const staged = { ...spec, epilogue: "  let first = acc0[0];", stageElements: 2048 };
    expect(depth(createTiledGemmShader(staged, { precision: "f32", inner: 16 }))).toBe(8);
    expect(depth(createTiledGemmShader(staged, { precision: "f16-chunked", inner: 16 }))).toBe(16);
    // Without an epilogue there is nothing to make room for.
    expect(depth(createTiledGemmShader(spec, { precision: "f32", inner: 16 }))).toBe(16);
  });
});

describe("pinning a variant", () => {
  it("holds the choice still and releases it again", () => {
    const original = gemmVariant();
    try {
      forceGemmVariant({ precision: "f16-chunked", inner: 16 });
      expect(gemmVariant()).toEqual({ precision: "f16-chunked", inner: 16 });
      forceGemmVariant(undefined);
      expect(gemmVariant()).toEqual(GEMM_VARIANT_F32);
    } finally {
      setGemmVariant(original);
    }
  });
});
