import { describe, expect, it } from "vitest";
import {
  MULTIMER_RELATIVE_CHANNELS, makeMultimerQueryOnlyFeatures, makeMultimerSequenceFeatures,
  multimerChainIdentifiers, multimerRelativeFeatures,
} from "../src/input/multimer-features.js";

const HOMOMER_CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

describe("AlphaFold-Multimer sequence features", () => {
  it("preserves both copies and chain boundaries for the 59-residue homodimer", () => {
    const tables = {
      atom37ToAtom14: Float32Array.from({ length: 21 * 37 }, (_, index) => index % 14),
      atom37Mask: new Float32Array(21 * 37).fill(1),
    };
    const features = makeMultimerSequenceFeatures(`${HOMOMER_CHAIN}:${HOMOMER_CHAIN}`, tables);

    expect(features.chains).toEqual([HOMOMER_CHAIN, HOMOMER_CHAIN]);
    expect(features.chainLengths).toEqual([59, 59]);
    expect(features.sequence).toBe(HOMOMER_CHAIN.repeat(2));
    expect(features.targetFeatures).toHaveLength(118 * 21);
    expect([...features.asymId.subarray(0, 59)]).toEqual(new Array(59).fill(1));
    expect([...features.asymId.subarray(59)]).toEqual(new Array(59).fill(2));
    expect([...features.entityId]).toEqual(new Array(118).fill(1));
    expect([...features.symId.subarray(0, 59)]).toEqual(new Array(59).fill(1));
    expect([...features.symId.subarray(59)]).toEqual(new Array(59).fill(2));
    expect([...features.residueIndex.subarray(0, 59)]).toEqual(
      Array.from({ length: 59 }, (_, index) => index),
    );
    expect([...features.residueIndex.subarray(59)]).toEqual(
      Array.from({ length: 59 }, (_, index) => index),
    );
  });

  it("assigns three asymmetric chains and symmetry copies to the 59-residue homotrimer", () => {
    const tables = {
      atom37ToAtom14: Float32Array.from({ length: 21 * 37 }, (_, index) => index % 14),
      atom37Mask: new Float32Array(21 * 37).fill(1),
    };
    const features = makeMultimerSequenceFeatures(new Array(3).fill(HOMOMER_CHAIN), tables);

    expect(features.chains).toEqual(new Array(3).fill(HOMOMER_CHAIN));
    expect(features.chainLengths).toEqual([59, 59, 59]);
    expect(features.sequence).toBe(HOMOMER_CHAIN.repeat(3));
    expect(features.targetFeatures).toHaveLength(177 * 21);
    expect([...features.entityId]).toEqual(new Array(177).fill(1));
    for (let copy = 0; copy < 3; copy += 1) {
      const start = copy * 59;
      const end = start + 59;
      expect([...features.asymId.subarray(start, end)]).toEqual(new Array(59).fill(copy + 1));
      expect([...features.symId.subarray(start, end)]).toEqual(new Array(59).fill(copy + 1));
      expect([...features.residueIndex.subarray(start, end)]).toEqual(
        Array.from({ length: 59 }, (_, index) => index),
      );
    }
  });

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
