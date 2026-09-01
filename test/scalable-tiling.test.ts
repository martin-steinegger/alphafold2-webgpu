import { describe, expect, it } from "vitest";
import {
  ATTENTION_WINDOW_TARGET_BYTES, attentionBatchWindow,
} from "../src/evoformer/attention.js";
import {
  OUTER_PRODUCT_BLOCK_LIMIT_BYTES, outerProductMeanRowBlock,
} from "../src/evoformer/outer-product-mean.js";
import {
  TRANSITION_CHUNK_TARGET_BYTES, TRANSITION_TILE_ROWS, transitionChunkRows,
} from "../src/evoformer/transition.js";

describe("bounded model scratch tensors", () => {
  it("keeps transition views aligned and below the scratch budget", () => {
    const totalRows = 508 * 291;
    const chunkRows = transitionChunkRows(totalRows, 256, 1024, 256 * 1024 ** 2, 256);
    expect(chunkRows * 1024 * 4).toBeLessThanOrEqual(TRANSITION_CHUNK_TARGET_BYTES);
    // Chunk starts must land on a valid binding offset and a whole GEMM tile.
    expect(chunkRows * 256 * 4 % 256).toBe(0);
    expect(chunkRows % TRANSITION_TILE_ROWS).toBe(0);
    // Still dozens of GEMM row tiles, so chunking costs dispatches, not efficiency.
    expect(chunkRows / TRANSITION_TILE_ROWS).toBeGreaterThanOrEqual(64);
  });

  it("bounds the window by the budget even when the device would allow more", () => {
    const totalRows = 508 * 291;
    const generous = transitionChunkRows(totalRows, 256, 1024, 2048 * 1024 ** 2, 256);
    const modest = transitionChunkRows(totalRows, 256, 1024, 256 * 1024 ** 2, 256);
    expect(generous).toBe(modest);
  });

  it("uses the full transition when the exact tensor fits the budget", () => {
    const rows = 64 * 59;
    expect(rows * 1024 * 4).toBeLessThanOrEqual(TRANSITION_CHUNK_TARGET_BYTES);
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

  it("covers the attention batch in one window when its tensors already fit", () => {
    // 59 residues: row attention over 256 sequences, column attention over 59.
    expect(attentionBatchWindow(256, 59, 256)).toBe(256);
    expect(attentionBatchWindow(59, 256, 256)).toBe(59);
    // The full 508-row alignment no longer fits the budget and is split.
    const window = attentionBatchWindow(508, 59, 256);
    expect(window).toBeLessThan(508);
    expect(window * 59 * 256 * 4).toBeLessThanOrEqual(ATTENTION_WINDOW_TARGET_BYTES);
  });

  it("windows the attention batch to stay inside the budget", () => {
    // Row, column and triangle attention at 384 residues, 256 clustered rows.
    for (const [batch, queries, channels] of [[256, 384, 256], [384, 256, 256], [384, 384, 128]] as const) {
      const window = attentionBatchWindow(batch, queries, channels);
      expect(window).toBeLessThan(batch);
      expect(window * queries * channels * 4).toBeLessThanOrEqual(ATTENTION_WINDOW_TARGET_BYTES);
      expect((window + 1) * queries * channels * 4).toBeGreaterThan(ATTENTION_WINDOW_TARGET_BYTES);
      // Still thousands of rows, so the projection stays a large GEMM.
      expect(window * queries).toBeGreaterThan(4096);
    }
  });

  it("never reports a zero-entry window, however wide the rows", () => {
    expect(attentionBatchWindow(64, 65_536, 256)).toBe(1);
  });
});
