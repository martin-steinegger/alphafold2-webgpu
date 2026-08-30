import { describe, expect, it } from "vitest";
import { outerProductMeanTileCapacity } from "../src/evoformer/outer-product-mean.js";
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

  it("adapts the outer-product tile to the selected binding limit", () => {
    const input = { sequences: 1024, length: 291, cOuter: 32, cZ: 128 };
    expect(outerProductMeanTileCapacity(input, 256 * 1024 ** 2)).toBe(32);
    expect(outerProductMeanTileCapacity(input, 64 * 1024 ** 2)).toBe(14);
  });
});
