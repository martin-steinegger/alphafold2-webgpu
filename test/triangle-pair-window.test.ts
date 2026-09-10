import { describe, expect, it } from "vitest";
import { createTriangleShaders } from "../src/triangle/shaders.js";
import { planShards } from "../src/runtime/sharded.js";
import { residueChunks } from "../src/evoformer/block.js";

/**
 * A projection reading only its block's pair rows takes them as one binding.
 *
 * The shard chain costs 2.4x on these kernels, so which of them may drop it is
 * worth pinning: a window is only correct where the rows a dispatch reads are
 * block.x + row, and the incoming direction reads a column window of every
 * row instead.
 */
const ORDER = [
  "W_LAYERNORMINWEIGHT", "W_LAYERNORMINBIAS", "W_LINEARAPWEIGHT", "W_LINEARAPBIAS",
  "W_LINEARAGWEIGHT", "W_LINEARAGBIAS", "W_LINEARBPWEIGHT", "W_LINEARBPBIAS",
  "W_LINEARBGWEIGHT", "W_LINEARBGBIAS", "W_LINEARGWEIGHT", "W_LINEARGBIAS",
  "W_LAYERNORMOUTWEIGHT", "W_LAYERNORMOUTBIAS", "W_LINEARZWEIGHT", "W_LINEARZBIAS",
];
const offsets = Object.fromEntries(ORDER.map((name) => [name, 0])) as never;
const SHAPE = { length: 256, cZ: 128, cHidden: 128 };
// Small enough to split the pair several ways, so a chain is visible.
const shards = planShards(256 * 256 * 128, 128, 4 * 1024 * 1024, 2);

const build = (direction: "outgoing" | "incoming", pairWindow: boolean) =>
  createTriangleShaders(SHAPE, "f32", offsets, 1e-5, direction, 64, "f16", "f16", false,
    shards, undefined, pairWindow);

/** Bindings the pair occupies, which a window reduces to one. */
const pairBindings = (source: string): number =>
  [...source.matchAll(/@binding\(\d+\) var<storage, read> z_\d+:/g)].length;

describe("the pair as a window of the block's own rows", () => {
  it("splits the pair over several bindings without one", () => {
    expect(shards.count).toBeGreaterThan(2);
    const plain = build("outgoing", false);
    expect(pairBindings(plain.projectWholeOperand)).toBe(shards.count);
    expect(plain.projectWholeOperand).toContain("pair_row_of(row) * CZ");
  });

  it("gives the whole operand one binding in both directions", () => {
    // It is written at its pair row either way, so it reads block.x + row.
    for (const direction of ["outgoing", "incoming"] as const) {
      const source = build(direction, true).projectWholeOperand;
      expect(pairBindings(source)).toBe(1);
      expect(source).toContain("fn pair_base_of(row: u32) -> u32 { return row * CZ; }");
    }
  });

  it("gives the block operand and the gate one binding in both directions", () => {
    // Incoming reads a column window of every pair row, so it only reaches one
    // binding once the residue axis is chunked; the chunk's rows are then
    // contiguous. See residueChunks.
    for (const direction of ["outgoing", "incoming"] as const) {
      const built = build(direction, true);
      expect(pairBindings(built.projectBlockOperand)).toBe(1);
      expect(pairBindings(built.projectGate)).toBe(1);
    }
  });

  it("indexes the chunk locally and the statistics globally", () => {
    const inward = build("incoming", true);
    // The window starts at the chunk, so the pair index drops block.z.
    expect(inward.projectBlockOperand)
      .toContain("fn pair_base_of(row: u32) -> u32 { return ((row / block.w) * L + row % block.w) * CZ; }");
    // The statistics and the mask are bound whole and keep the global row.
    expect(inward.projectBlockOperand)
      .toContain("return (block.z + row / block.w) * L + block.x / L + row % block.w;");
    expect(inward.projectBlockOperand).toContain("statistics[2u * pair_row]");
  });

  it("stores the gate at the block's own row, not the chunk's", () => {
    // The output projection reads the gate across the whole block, so a
    // chunked write has to land at the absolute row.
    expect(build("incoming", true).projectGate).toContain("gate[(block.z * block.w + row) * CZ + column]");
    expect(build("outgoing", true).projectGate).toContain("gate[row * CZ + column]");
  });

  it("still reads the statistics and the mask at the global pair row", () => {
    // Those two are bound whole, so a window must not make them local.
    const source = build("outgoing", true).projectBlockOperand;
    expect(source).toContain("let pair_row = pair_row_of(row);");
    expect(source).toContain("statistics[2u * pair_row]");
    expect(source).toContain("mask[pair_row_of(row)]");
  });

  it("passes the block row, never the pair row, to the windowed load", () => {
    // The bug this cost: the GEMM kept calling normalized_input with the
    // global row while the binding started at the block, and the fold ran to
    // completion with a pLDDT of 23.8 instead of 86.4.
    for (const direction of ["outgoing", "incoming"] as const) {
      for (const source of Object.values(build(direction, true))) {
        expect(source).not.toContain("normalized_input(pair_row_of(row)");
      }
    }
  });
});

describe("how many residue chunks the incoming projections need", () => {
  const GiB = 2_147_483_644;
  it("takes one where the whole pair already fits a binding", () => {
    // Below 2,897 residues the pair is one binding and nothing is cut.
    expect(residueChunks(825, 256, 128, "f16", GiB)).toBe(1);
    expect(residueChunks(1650, 576, 128, "f16", GiB)).toBe(1);
  });

  it("takes two for a tetramer and six at the ceiling", () => {
    // A 2 GiB binding holds 8.39M pair rows packed; a 3,300-residue pair has
    // 10.89M, so half of them fit at once.
    expect(residueChunks(3300, 256, 128, "f16", GiB)).toBe(2);
    expect(residueChunks(6688, 256, 128, "f16", GiB)).toBe(6);
  });

  it("keeps every chunk's window inside one binding", () => {
    for (const [length, rows] of [[3300, 256], [6688, 256], [4096, 128]] as const) {
      const chunks = residueChunks(length, rows, 128, "f16", GiB);
      const window = (Math.ceil(length / chunks) - 1) * length + rows;
      expect(window * 128 * 2).toBeLessThanOrEqual(GiB);
    }
  });
});
