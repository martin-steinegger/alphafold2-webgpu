import { describe, expect, it } from "vitest";
import {
  multimerRankingConfidence, predictedInterfaceTmScore,
} from "../src/heads/confidence.js";

describe("AlphaFold-Multimer confidence", () => {
  it("computes ipTM from inter-chain pairs only", () => {
    const length = 4;
    const breaks = new Float32Array([0, 2]);
    const bins = breaks.length + 1;
    const asymId = new Float32Array([1, 1, 2, 2]);
    const logits = new Float32Array(length * length * bins);
    for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
      const base = (i * length + j) * bins;
      logits[base + (asymId[i] === asymId[j] ? 2 : 0)] = 40;
    }
    const d0 = 1.24 * Math.cbrt(19 - 15) - 1.8;
    const expected = 1 / (1 + 1 / (d0 * d0));
    expect(predictedInterfaceTmScore(logits, length, breaks, asymId)).toBeCloseTo(expected, 6);
  });

  it("uses the official 0.8 ipTM plus 0.2 pTM ranking", () => {
    expect(multimerRankingConfidence(0.5, 0.75)).toBeCloseTo(0.7, 12);
    expect(() => multimerRankingConfidence(0.5, Number.NaN)).toThrow(/finite scores/);
  });

  it("validates chain identifiers", () => {
    expect(() => predictedInterfaceTmScore(
      new Float32Array(12), 2, new Float32Array([0, 2]), new Float32Array([1]),
    )).toThrow(/asymId/);
  });
});
