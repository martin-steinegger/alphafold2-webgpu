import { describe, expect, it } from "vitest";
import { chainBoundaries, chainColor, chainLetter, chainMeanPlddt, chainPairError, chainSpans } from "../web/chains.js";

describe("chain identity", () => {
  it("labels and places every chain in the concatenated sequence", () => {
    expect(chainSpans([3, 2])).toEqual([
      { index: 0, letter: "A", color: chainColor(0), start: 0, end: 3, length: 3 },
      { index: 1, letter: "B", color: chainColor(1), start: 3, end: 5, length: 2 },
    ]);
    expect(chainBoundaries([3, 2, 4])).toEqual([3, 5]);
    expect(chainBoundaries([5])).toEqual([]);
  });

  it("keeps letters and colours stable and cycles the palette", () => {
    expect(chainLetter(0)).toBe("A");
    expect(chainLetter(26)).toBe("a");
    expect(chainColor(0)).not.toBe(chainColor(1));
    expect(chainColor(12)).toBe(chainColor(0));
  });

  it("averages confidence within each chain", () => {
    const plddt = Float32Array.from([90, 80, 20, 40]);
    expect(chainMeanPlddt(plddt, [2, 2])).toEqual([85, 30]);
    expect(chainMeanPlddt(plddt, [4])).toEqual([57.5]);
  });

  it("averages the alignment error over each ordered chain pair", () => {
    // Two chains of one residue each, so every block is a single cell.
    const pae = Float32Array.from([1, 8, 6, 2]);
    expect(chainPairError(pae, [1, 1])).toEqual([[1, 8], [6, 2]]);
    // A larger block averages: chain A against chain B is cells 2 and 3.
    const wider = Float32Array.from([
      0, 1, 4, 5,
      1, 0, 6, 7,
      4, 6, 0, 1,
      5, 7, 1, 0,
    ]);
    expect(chainPairError(wider, [2, 2])).toEqual([[0.5, 5.5], [5.5, 0.5]]);
  });
});
