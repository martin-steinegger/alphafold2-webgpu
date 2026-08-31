import { describe, expect, it } from "vitest";
import { parseSequenceExpression } from "../src/input/sequence-expression.js";

describe("unified monomer and multimer sequence input", () => {
  it("keeps a plain sequence on the monomer path", () => {
    expect(parseSequenceExpression(" acd\nEF ")).toEqual({
      chains: ["ACDEF"], sequence: "ACDEF", multimer: false,
    });
  });

  it("detects and preserves ordered complex chains", () => {
    expect(parseSequenceExpression("ACD : efg : XX")).toEqual({
      chains: ["ACD", "EFG", "XX"], sequence: "ACDEFGXX", multimer: true,
    });
  });

  it("rejects empty chains and invalid public input", () => {
    expect(() => parseSequenceExpression("")).toThrow(/valid amino-acid chains/);
    expect(() => parseSequenceExpression("AC::GG")).toThrow(/valid amino-acid chains/);
    expect(() => parseSequenceExpression(":AC")).toThrow(/valid amino-acid chains/);
    expect(() => parseSequenceExpression("AC:B")).toThrow(/valid amino-acid chains/);
  });
});
