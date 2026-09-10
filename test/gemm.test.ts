import { afterEach, describe, expect, it } from "vitest";
import {
  createTiledGemmShader, gemmGrid, matrixGemmStorageBytes, setGemmVariant, tallGemmTileRows,
  GEMM_TILE_COLUMNS, GEMM_TILE_ROWS, TALL_GEMM_TILE_ROWS,
} from "../src/runtime/gemm.js";

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

describe("the taller tile a contraction may ask for", () => {
  const UNIT = { componentType: "f16", M: 16, N: 16, K: 16 } as const;
  const MATRIX = { precision: "matrix", inner: 8, matrix: UNIT, matrixDepth: 2 } as const;
  const limits = (storage: number, invocations: number): GPUSupportedLimits =>
    ({ maxComputeWorkgroupStorageSize: storage,
      maxComputeInvocationsPerWorkgroup: invocations } as GPUSupportedLimits);

  afterEach(() => setGemmVariant({ precision: "f32", inner: 8 }));

  it("takes it where the device grants the subgroups and the storage", () => {
    setGemmVariant(MATRIX);
    // What an RTX PRO 6000 Blackwell reports, over Dawn and over wgpu alike.
    expect(tallGemmTileRows(limits(49152, 1024))).toBe(TALL_GEMM_TILE_ROWS);
    // Sixteen subgroups of thirty-two, against eight for the default tile, and
    // 39,648 workgroup bytes against 24,288.
    expect(matrixGemmStorageBytes(UNIT, undefined, 32, TALL_GEMM_TILE_ROWS)).toBe(39648);
    expect(matrixGemmStorageBytes(UNIT, undefined, 32)).toBe(24288);
  });

  it("declines it on either ceiling, and on a device with no matrix units", () => {
    setGemmVariant(MATRIX);
    // The WebGPU baseline is 16,384 B and 256 invocations; either alone refuses.
    expect(tallGemmTileRows(limits(16384, 1024))).toBe(GEMM_TILE_ROWS);
    expect(tallGemmTileRows(limits(49152, 256))).toBe(GEMM_TILE_ROWS);
    setGemmVariant({ precision: "f32", inner: 8 });
    expect(tallGemmTileRows(limits(49152, 1024))).toBe(GEMM_TILE_ROWS);
  });

  it("tiles the kernel and the grid the same way, or rows go unwritten", () => {
    setGemmVariant(MATRIX);
    // 256 rows is four tiles of the default and two of the taller one.
    expect(gemmGrid(256, 128)).toEqual([1, 4]);
    expect(gemmGrid(256, 128, GEMM_TILE_COLUMNS, TALL_GEMM_TILE_ROWS)).toEqual([1, 2]);
  });

  it("is refused by the kernels that are not written for it", () => {
    // The hand-tiled kernel and the f32 matrix one address the tile with the
    // constant, so a caller must not be able to ask them for another.
    const shader = {
      preamble: "@group(0) @binding(0) var<storage, read_write> output: array<f32>;",
      rows: "64u", inner: "8u", columns: "128u",
      sourceElement: "0.0", weightElement: "0.0",
      store: "output[row * 128u + column] = element;",
      tileRows: TALL_GEMM_TILE_ROWS,
    };
    expect(() => createTiledGemmShader(shader, { precision: "f32", inner: 8 }))
      .toThrow(/only the f16 matrix kernel/);
  });
});
