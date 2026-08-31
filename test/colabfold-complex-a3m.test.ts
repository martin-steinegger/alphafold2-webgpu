import { describe, expect, it } from "vitest";
import { parseColabFoldComplexA3m } from "../src/input/colabfold-complex-a3m.js";
import { assembleComplexA3m } from "../src/input/mmseqs2-api.js";

describe("ColabFold serialized complex A3M", () => {
  it("recovers paired and unpaired chain alignments before Multimer cropping", () => {
    const serialized = [
      "#2,2\t1,1",
      ">101\t102", "ACGG",
      ">paired-a\tpaired-b", "A-G-",
      ">unpaired-a", "AC--",
      ">unpaired-b", "--GG",
      "",
    ].join("\n");
    const actual = parseColabFoldComplexA3m(serialized);
    expect(actual).toBeDefined();
    const expected = assembleComplexA3m(
      ["AC", "GG"], ["AC", "GG"],
      [">unpaired-a\nAC\n", ">unpaired-b\nGG\n"],
      [">101\nAC\n>paired-a\nA-\n", ">102\nGG\n>paired-b\nG-\n"],
    );
    expect(actual?.chains).toEqual(["AC", "GG"]);
    expect(actual?.cardinalities).toEqual([1, 1]);
    expect(actual?.a3m).toBe(expected.a3m);
    expect(actual?.mask).toEqual(expected.mask);
    expect(actual?.depth).toBe(expected.depth);
  });

  it("leaves ordinary monomer A3Ms to the normal parser", () => {
    expect(parseColabFoldComplexA3m(">query\nACDE\n")).toBeUndefined();
  });
});
