import { describe, expect, it } from "vitest";
import {
  MULTIMER_RELATIVE_CHANNELS, makeMultimerQueryOnlyFeatures, makeMultimerSequenceFeatures,
  multimerChainIdentifiers, multimerRelativeFeatures,
} from "../src/input/multimer-features.js";

describe("AlphaFold-Multimer sequence features", () => {
  it("assigns asym, entity and symmetry identifiers to heteromers and homomers", () => {
    const features = multimerChainIdentifiers(["AC", "G", "AC"]);
    expect([...features.asymId]).toEqual([1, 1, 2, 3, 3]);
    expect([...features.entityId]).toEqual([1, 1, 2, 1, 1]);
    expect([...features.symId]).toEqual([1, 1, 1, 2, 2]);
    expect([...features.residueIndex]).toEqual([0, 1, 0, 0, 1]);
  });

  it("uses the official 66+1+6 relative encoding", () => {
    const tables = {
      atom37ToAtom14: Float32Array.from({ length: 21 * 37 }, (_, index) => index % 37),
      atom37Mask: new Float32Array(21 * 37).fill(1),
    };
    const features = makeMultimerSequenceFeatures("AC:AC:G", tables);
    const length = features.sequence.length;
    expect(features.targetFeatures.length).toBe(length * 21);
    const relativeFeatures = multimerRelativeFeatures(
      features.residueIndex, features.asymId, features.entityId, features.symId,
    );
    expect(relativeFeatures.length).toBe(length * length * MULTIMER_RELATIVE_CHANNELS);
    const active = (i: number, j: number): number[] => {
      const base = (i * length + j) * MULTIMER_RELATIVE_CHANNELS;
      const result: number[] = [];
      for (let channel = 0; channel < MULTIMER_RELATIVE_CHANNELS; channel += 1) {
        if (relativeFeatures[base + channel] === 1) result.push(channel);
      }
      return result;
    };
    expect(active(0, 1)).toEqual([31, 66, 69]); // same chain/entity/copy, offset -1
    expect(active(0, 2)).toEqual([65, 66, 68]); // another copy of the same entity
    expect(active(0, 4)).toEqual([65, 72]); // different chain and entity
  });

  it("rejects malformed public inputs", () => {
    expect(() => multimerChainIdentifiers("AC" )).toThrow(/at least two chains/);
    expect(() => multimerChainIdentifiers("AC::G")).toThrow(/each multimer chain/);
    expect(() => multimerChainIdentifiers("AC:B")).toThrow(/each multimer chain/);
  });

  it("builds deterministic 21/49-channel recycle inputs without joining chain identities", () => {
    const tables = {
      atom37ToAtom14: Float32Array.from({ length: 21 * 37 }, (_, index) => index % 14),
      atom37Mask: new Float32Array(21 * 37).fill(1),
    };
    const recycles = makeMultimerQueryOnlyFeatures("AC:GG", tables, { recycles: 1, randomSeed: 7 });
    expect(recycles).toHaveLength(2);
    expect(recycles[0]!.targetFeatures).toHaveLength(4 * 21);
    expect(recycles[0]!.msaFeatures).toHaveLength(4 * 49);
    expect([...recycles[0]!.chainRelative.asymId]).toEqual([1, 1, 2, 2]);
    expect([...recycles[0]!.chainRelative.entityId]).toEqual([1, 1, 2, 2]);
    expect(recycles[0]!.msaFeatures).not.toEqual(recycles[1]!.msaFeatures);
  });
});
