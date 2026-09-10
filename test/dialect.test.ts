import { describe, expect, it } from "vitest";
import {
  DAWN_MATRIX, WGPU_MATRIX, calibrateDialect, dialect, matrixSpelling, presetDialect,
} from "../src/runtime/dialect.js";

/** A device that accepts the sources a named implementation would. */
function stubDevice(accepts: (code: string) => boolean, features: string[]): GPUDevice {
  let failed = false;
  return {
    features: new Set(features as GPUFeatureName[]),
    createShaderModule: ({ code }: { code: string }) => { failed = !accepts(code); return {}; },
    pushErrorScope: () => undefined,
    popErrorScope: async () => (failed ? { message: "rejected" } : null),
  } as unknown as GPUDevice;
}

const dawnLike = (code: string): boolean => !code.includes("wgpu_cooperative_matrix");
const wgpuLike = (code: string): boolean =>
  !code.includes("enable subgroups;") && !code.includes("chromium_experimental");
// naga with the subgroups directive patched in: it takes that one, and still
// has no counterpart to subgroup-size-control or to the width attribute.
const patchedNagaLike = (code: string): boolean =>
  !code.includes("subgroup_size_control") && !code.includes("@subgroup_size")
  && !code.includes("chromium_experimental");

describe("the shader dialect", () => {
  it("takes the subgroup directives Dawn wants and leaves them off for naga", async () => {
    const dawn = await calibrateDialect(stubDevice(dawnLike, ["subgroups"]));
    expect(dawn.subgroupEnable).toBe("enable subgroups;\nenable subgroup_size_control;\n");
    expect(dawn.subgroupSize(32)).toBe(" @subgroup_size(32)");
    const wgpu = await calibrateDialect(stubDevice(wgpuLike, ["subgroups"]));
    expect(wgpu.subgroupEnable).toBe("");
    expect(wgpu.subgroupSize(32)).toBe("");
  });

  it("takes the directive without the width attribute where only that is implemented",
    async () => {
      const patched = await calibrateDialect(stubDevice(patchedNagaLike, ["subgroups"]));
      expect(patched.subgroupEnable).toBe("enable subgroups;\n");
      expect(patched.subgroupSize(32)).toBe("");
    });

  it("asks for no directive on a device without subgroups", async () => {
    expect((await calibrateDialect(stubDevice(dawnLike, []))).subgroupEnable).toBe("");
  });

  it("settles the matrix spelling each implementation accepts", async () => {
    const dawn = stubDevice(dawnLike, ["subgroups", "chromium-experimental-subgroup-matrix"]);
    const wgpu = stubDevice(wgpuLike, ["subgroups"]);
    expect((await calibrateDialect(dawn)).matrix).toBe(DAWN_MATRIX);
    expect((await calibrateDialect(wgpu)).matrix).toBe(WGPU_MATRIX);
  });

  it("keeps two devices apart, which module state could not", async () => {
    const withMatrix = stubDevice(dawnLike, ["subgroups", "chromium-experimental-subgroup-matrix"]);
    const without = stubDevice(() => false, []);
    await calibrateDialect(withMatrix);
    await calibrateDialect(without);
    expect(dialect(withMatrix).matrix).toBe(DAWN_MATRIX);
    expect(dialect(withMatrix).subgroupEnable)
      .toBe("enable subgroups;\nenable subgroup_size_control;\n");
    expect(dialect(without).matrix).toBeUndefined();
    expect(dialect(without).subgroupEnable).toBe("");
  });

  it("gives the directive-free form for a device it never saw", () => {
    expect(dialect(undefined).subgroupEnable).toBe("");
    expect(dialect(undefined).subgroupSize(32)).toBe("");
    expect(dialect(undefined).matrix).toBeUndefined();
  });

  it("refuses the matrix half where there is none", async () => {
    const plain = stubDevice(() => false, []);
    await calibrateDialect(plain);
    expect(() => matrixSpelling(plain)).toThrow(/no subgroup matrix/);
  });

  it("takes a forced dialect, for a test that wants one deliberately", () => {
    const device = stubDevice(dawnLike, []);
    presetDialect(device, { subgroupEnable: "", subgroupSize: () => "",
    subgroupBarrier: "workgroupBarrier()", matrix: WGPU_MATRIX });
    expect(matrixSpelling(device)).toBe(WGPU_MATRIX);
  });

  it("spells the operands the way each implementation reads them", () => {
    expect(DAWN_MATRIX.left("f16", 16, 16)).toBe("subgroup_matrix_left<f16, 16, 16>");
    expect(WGPU_MATRIX.left("f16", 16, 16)).toBe("coop_mat16x16<f16, A>");
    expect(DAWN_MATRIX.subgroupSize(32)).toBe(" @subgroup_size(32)");
    expect(WGPU_MATRIX.subgroupSize(32)).toBe("");
  });

  it("transposes a store where the caller asks, in both spellings", () => {
    expect(DAWN_MATRIX.store("t", "0u", "acc", "20u", "col")).toContain("col_major");
    expect(DAWN_MATRIX.store("t", "0u", "acc", "20u")).toContain("row_major");
    // coopStoreT is the row-major store, as coopLoadT is the row-major load.
    expect(WGPU_MATRIX.store("t", "0u", "acc", "20u", "col")).toContain("coopStore(");
    expect(WGPU_MATRIX.store("t", "0u", "acc", "20u")).toContain("coopStoreT(");
  });

  it("keeps the majorness the right way round, which a reversal would hide", () => {
    // coopLoadT is the row-major load. Reversed, this compiles and computes a
    // transposed product.
    expect(WGPU_MATRIX.load("A", "tile", "0u", "16u", "row")).toContain("coopLoadT");
    expect(WGPU_MATRIX.load("A", "tile", "0u", "16u", "col")).toContain("coopLoad<");
    expect(WGPU_MATRIX.load("A", "tile", "0u", "16u", "col")).not.toContain("coopLoadT");
    expect(DAWN_MATRIX.load("A", "tile", "0u", "16u", "row")).toContain("row_major");
    expect(DAWN_MATRIX.load("A", "tile", "0u", "16u", "col")).toContain("col_major");
  });

  it("refuses a shape naga has no generator for", () => {
    expect(() => WGPU_MATRIX.left("f16", 16, 8)).toThrow(/square/);
  });
});
