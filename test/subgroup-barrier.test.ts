import { describe, expect, it } from "vitest";
import { createAttentionMatrixFlashShader } from "../src/evoformer/attention-matrix.js";
import { DAWN_MATRIX, WGPU_MATRIX } from "../src/runtime/dialect.js";

/**
 * A barrier between accesses only one subgroup makes need only order that
 * subgroup. WGSL has subgroupBarrier and naga lowers it; Dawn answers
 * "unresolved call target", so the kernel takes the spelling from the dialect
 * and falls back to the workgroup barrier, which is stronger.
 */
const unit = { componentType: "f16" as const, M: 16, N: 16, K: 16 };
const build = (barrier?: string) =>
  createAttentionMatrixFlashShader(32, unit, barrier === undefined ? DAWN_MATRIX : WGPU_MATRIX,
    false, ...(barrier === undefined ? [] : [barrier]) as [string?]);

describe("which barrier the flash kernel takes", () => {
  it("uses the workgroup barrier when the caller says nothing", () => {
    const source = build();
    expect(source).not.toContain("subgroupBarrier");
    expect(source.match(/workgroupBarrier\(\)/g)).toHaveLength(6);
  });

  it("takes a subgroup barrier for the two that only order one subgroup", () => {
    const source = build("subgroupBarrier()");
    expect(source.match(/subgroupBarrier\(\)/g)).toHaveLength(2);
    // The other four order the staged tiles, which every subgroup reads.
    expect(source.match(/workgroupBarrier\(\)/g)).toHaveLength(4);
  });

  it("keeps a workgroup barrier at the end of the key pass", () => {
    // It is what stops a subgroup that has finished reading values_tile from
    // staging the next key tile over one still reading this one. A subgroup
    // barrier there would be a race, not an optimisation.
    const source = build("subgroupBarrier()");
    const tail = source.slice(source.lastIndexOf("out_0 = out_0"));
    expect(tail).toContain("workgroupBarrier()");
    expect(tail).not.toContain("subgroupBarrier()");
  });

  it("owns the output rows inside the subgroup, which is what frees the second", () => {
    // Every row a lane touches has to be one its own subgroup wrote.
    const source = build("subgroupBarrier()");
    expect(source).toContain("let in_lane = lane % 32u;");
    expect(source).toContain("let own_row_0 = rows_at + (in_lane + 0u) / 8u;");
    expect(source).toContain("let own_row_3 = rows_at + (in_lane + 96u) / 8u;");
  });
});
