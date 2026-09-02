import { describe, expect, it } from "vitest";
import { analyzeMsa } from "../web/msa-plot.js";

describe("MSA coverage plot", () => {
  it("computes query identity, per-position coverage, and sorted row order", () => {
    const data = analyzeMsa(">query\nACDE\n>partial\nA--E\n>close\nAC-E\n");
    expect(data.depth).toBe(3); expect(data.length).toBe(4);
    expect(Array.from(data.coverage)).toEqual([3, 2, 1, 3]);
    expect(Array.from(data.identities)).toEqual([1, .5, .75]);
    expect(Array.from(data.order)).toEqual([1, 2, 0]);
    // Without chains there is one block, drawn most similar first.
    expect(Array.from(data.rows)).toEqual([0, 2, 1]);
    expect(data.blockStarts).toEqual([]);
  });

  it("groups a complex into paired rows first, then each chain's own hits", () => {
    // Two chains of two columns: one row covers both, one only chain A, one only chain B.
    const data = analyzeMsa(
      ">query\nACDE\n>paired\nACDE\n>chainA\nAC--\n>chainB\n--DE\n", [2, 2],
    );
    expect(Array.from(data.rows)).toEqual([0, 1, 2, 3]);
    expect(data.blockStarts).toEqual([2, 3]);
  });

  it("orders the unpaired blocks by chain and the rows inside them by identity", () => {
    const data = analyzeMsa(
      ">query\nACDE\n>weakB\n--D-\n>strongB\n--DE\n>chainA\nAC--\n", [2, 2],
    );
    // The query covers both chains, then chain A's block, then chain B's, whose
    // closer row is drawn first.
    expect(Array.from(data.rows)).toEqual([0, 3, 2, 1]);
    expect(data.blockStarts).toEqual([1, 2]);
  });
});
