import { describe, expect, it } from "vitest";
import {
  forceGemmVariant, gemmVariantCandidates, gemmVariantName, SHIPPABLE_GEMM_PRECISIONS,
} from "../src/runtime/gemm-selection.js";
import {
  createTiledGemmShader, GEMM_VARIANT_F32, gemmGrid, gemmVariant, setGemmVariant,
  type GemmVariant,
} from "../src/runtime/gemm.js";

function fakeDevice(halfPrecision: boolean, matrixUnits = false): GPUDevice {
  const features = new Set<GPUFeatureName>();
  if (halfPrecision) features.add("shader-f16" as GPUFeatureName);
  if (matrixUnits) features.add("chromium-experimental-subgroup-matrix" as GPUFeatureName);
  return {
    features,
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
    expect(SHIPPABLE_GEMM_PRECISIONS).toContain("matrix");
  });

  it("never offers the arrangement that took a deep MSA to NaN", () => {
    // Pure f16 was the fastest candidate measured and it cannot be selected:
    // no probe cheap enough to run at device creation reaches the depth that
    // exposes it, so it is excluded here rather than by measurement.
    expect(SHIPPABLE_GEMM_PRECISIONS).not.toContain("f16");
    expect(gemmVariantCandidates(fakeDevice(true))
      .some((variant) => variant.precision === "f16")).toBe(false);
  });

  it("measures both k depths for every arrangement that has two", () => {
    const candidates = gemmVariantCandidates(fakeDevice(true));
    // f32 and the two half-precision arrangements, at k8 and k16.
    expect(candidates).toHaveLength(6);
    expect(candidates.filter((variant) => variant.inner === 16)).toHaveLength(3);
    expect(gemmVariantName({ precision: "f16-chunked", inner: 16 }))
      .toBe("f16-chunked-64x128k16");
  });

  it("offers the matrix units only where they exist, and only once", () => {
    expect(gemmVariantCandidates(fakeDevice(true))
      .some((variant) => variant.precision === "matrix")).toBe(false);
    const withUnits = gemmVariantCandidates(fakeDevice(true, true));
    // The units fix the contraction step at 8, so there is no k depth to try.
    expect(withUnits.filter((variant) => variant.precision === "matrix")).toHaveLength(1);
    expect(gemmVariantName({ precision: "matrix", inner: 8 })).toBe("matrix-64x128");
    // And they do not depend on half precision being available.
    expect(gemmVariantCandidates(fakeDevice(false, true))
      .filter((variant) => variant.precision === "matrix")).toHaveLength(1);
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

describe("the matrix units and what they can serve", () => {
  const arrays = {
    sourceArray: { array: "source", stride: "256u" },
    weightArray: { array: "weights", stride: "128u" },
  };
  const matrix = { precision: "matrix", inner: 8 } as const;

  it("takes a caller that declared its operands as arrays", () => {
    const shader = createTiledGemmShader({ ...spec, ...arrays }, matrix);
    expect(shader).toContain("enable chromium_experimental_subgroup_matrix;");
    expect(shader).toContain("subgroupMatrixMultiplyAccumulate");
    // The caller's own store still runs, per element, over the staged region.
    expect(shader).toContain("output[row * 128u + column] = element;");
  });

  it("keeps the hand-tiled kernel for a caller that did not", () => {
    // Operands given only as expressions could be unpacking a half word or
    // windowing a tensor, neither of which a matrix load can do.
    const shader = createTiledGemmShader(spec, matrix);
    expect(shader).not.toContain("subgroup_matrix");
    expect(shader).toContain("var<workgroup> gemm_source");
  });

  it("keeps the hand-tiled kernel for an epilogue or a vector store", () => {
    const withEpilogue = createTiledGemmShader(
      { ...spec, ...arrays, epilogue: "  let z = acc0[0];", stageElements: 4 }, matrix);
    expect(withEpilogue).not.toContain("subgroup_matrix");
    const withVectorStore = createTiledGemmShader(
      { ...spec, ...arrays, storeVector: "  let z = values;" }, matrix);
    expect(withVectorStore).not.toContain("subgroup_matrix");
  });

  it("covers the same output tile as the kernel it replaces", () => {
    // gemmGrid does not know which shader is asking, so a matrix kernel on a
    // different tile would hand the wrong grid to any caller that opted out.
    const shader = createTiledGemmShader({ ...spec, ...arrays }, matrix);
    expect(shader).toContain("group.y * 64u");
    expect(shader).toContain("group.x * 128u");
    expect(gemmGrid(64, 128)).toEqual([1, 1]);
    expect(gemmGrid(65, 129)).toEqual([2, 2]);
  });
});

describe("what a browser without the extensions gets", () => {
  // AGENTS.md asks for standards-compliant WGSL. Half precision is a core
  // WebGPU feature; the subgroup matrix units are an experimental Chromium
  // extension. Neither may appear anywhere a device did not ask for it, or
  // the shader will not compile and the page will not run at all.
  it("offers nothing but f32 to a device that reports no features", () => {
    const bare = gemmVariantCandidates(fakeDevice(false, false));
    expect(bare.every((variant) => variant.precision === "f32")).toBe(true);
  });

  it("never names an extension in a kernel that does not use one", () => {
    const arrays = {
      sourceArray: { array: "source", stride: "256u" },
      weightArray: { array: "weights", stride: "128u" },
    };
    for (const precision of ["f32", "f16-mixed", "f16-chunked"] as const) {
      for (const inner of [8, 16] as const) {
        const shader = createTiledGemmShader({ ...spec, ...arrays }, { precision, inner });
        expect(shader, `${precision} k${inner}`).not.toContain("chromium");
        expect(shader, `${precision} k${inner}`).not.toContain("subgroup_matrix");
        // f16 is core WebGPU, but still only where it is actually used.
        if (precision === "f32") expect(shader, precision).not.toContain("enable f16");
      }
    }
  });

  it("emits no extension for a caller that did not declare arrays, even asked to", () => {
    // Which is most of them: an operand that unpacks a half word or windows a
    // tensor cannot be addressed by a matrix load, so those keep the portable
    // kernel whatever the device turns out to support.
    const shader = createTiledGemmShader(spec, { precision: "matrix", inner: 8 });
    expect(shader).not.toContain("chromium");
    expect(shader).not.toContain("subgroup_matrix");
  });

  it("keeps the f32 kernel free of every extension it does not need", () => {
    const shader = createTiledGemmShader(spec, GEMM_VARIANT_F32);
    for (const forbidden of ["chromium", "subgroup_matrix", "enable f16", "f16"]) {
      expect(shader, forbidden).not.toContain(forbidden);
    }
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
