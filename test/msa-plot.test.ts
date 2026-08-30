import { describe, expect, it } from "vitest";
import { analyzeMsa } from "../web/msa-plot.js";

describe("MSA coverage plot", () => {
  it("computes query identity, per-position coverage, and sorted row order", () => {
    const data = analyzeMsa(">query\nACDE\n>partial\nA--E\n>close\nAC-E\n");
    expect(data.depth).toBe(3); expect(data.length).toBe(4);
    expect(Array.from(data.coverage)).toEqual([3, 2, 1, 3]);
    expect(Array.from(data.identities)).toEqual([1, .5, .75]);
    expect(Array.from(data.order)).toEqual([1, 2, 0]);
  });
});
