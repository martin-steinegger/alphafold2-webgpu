import { describe, expect, it } from "vitest";
import {
  OUTER_PRODUCT_BLOCK_LIMIT_BYTES, outerProductMeanRowBlock,
} from "../src/evoformer/outer-product-mean.js";
import { transitionChunkRows } from "../src/evoformer/transition.js";

describe("bounded model scratch tensors", () => {
  it("keeps transition views aligned and below the scratch budget", () => {
    const totalRows = 508 * 291;
    const chunkRows = transitionChunkRows(totalRows, 256, 1024, 256 * 1024 ** 2, 256);
    expect(chunkRows).toBe(24_576);
    expect(chunkRows * 256 * 4 % 256).toBe(0);
    expect(chunkRows * 1024 * 4).toBeLessThanOrEqual(96 * 1024 ** 2);
  });

  it("uses the full transition when the exact tensor fits", () => {
    const rows = 508 * 59;
    expect(transitionChunkRows(rows, 256, 1024, 128 * 1024 ** 2, 256)).toBe(rows);
  });

  it("contracts the whole outer product at once when it fits the budget", () => {
    // 59 residues needs 14 MiB, so the block covers every residue.
    expect(outerProductMeanRowBlock(59, 32)).toBe(59);
  });

  it("blocks the outer-product contraction to stay inside the budget", () => {
    for (const length of [256, 291, 384, 512]) {
      const block = outerProductMeanRowBlock(length, 32);
      expect(block).toBeGreaterThan(0);
      expect(block * length * 32 * 32 * 4).toBeLessThanOrEqual(OUTER_PRODUCT_BLOCK_LIMIT_BYTES);
      // One more residue would exceed it, so the block is as large as it can be.
      expect((block + 1) * length * 32 * 32 * 4).toBeGreaterThan(OUTER_PRODUCT_BLOCK_LIMIT_BYTES);
    }
  });

  it("never reports a zero-residue block, however long the chain", () => {
    expect(outerProductMeanRowBlock(65_536, 32)).toBe(1);
  });
});
