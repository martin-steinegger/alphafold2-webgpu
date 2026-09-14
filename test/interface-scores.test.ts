import { describe, expect, it } from "vitest";
import { interfaceMaxima, interfaceScores } from "../src/heads/interface-scores.js";

/**
 * A complex with placed and unplaced residues, and chains that touch or do not.
 *
 * The numbers this is checked against come from ColabFold's own
 * colabfold/alphafold/ipsae.py, run on the tensors this builds.
 */
function complex(chainLengths: readonly number[], separation: number): {
  predictedAlignedError: Float32Array; plddt: Float32Array; asymId: Float32Array;
  atom37: Float32Array; atom37Mask: Float32Array;
} {
  const length = chainLengths.reduce((sum, count) => sum + count, 0);
  let state = 12345;
  // A plain linear congruential generator, so the case is the same everywhere.
  const random = (): number => {
    state = (Math.imul(state, 1103515245) + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
  const asymId = new Float32Array(length);
  const atom37 = new Float32Array(length * 37 * 3);
  const atom37Mask = new Float32Array(length * 37);
  let residue = 0;
  for (const [chain, count] of chainLengths.entries()) {
    for (let index = 0; index < count; index += 1, residue += 1) {
      asymId[residue] = chain + 1;
      const base = (residue * 37 + 1) * 3;
      for (let axis = 0; axis < 3; axis += 1) {
        const centre = axis === 0 ? chain * separation : 0;
        atom37[base + axis] = centre + (random() - 0.5) * 6;
        atom37[(residue * 37 + 3) * 3 + axis] = atom37[base + axis]! + (axis === 0 ? 1.5 : 0);
      }
      atom37Mask[residue * 37 + 1] = 1;
      atom37Mask[residue * 37 + 3] = 1;
    }
  }
  // A glycine, which has no CB and stands in with its CA, and a residue the
  // structure module never placed, which is in no interface at all.
  atom37Mask[2 * 37 + 3] = 0;
  atom37Mask[(length - 1) * 37 + 1] = 0;
  atom37Mask[(length - 1) * 37 + 3] = 0;
  const predictedAlignedError = new Float32Array(length * length);
  for (let anchor = 0; anchor < length; anchor += 1) {
    for (let partner = 0; partner < length; partner += 1) {
      predictedAlignedError[anchor * length + partner] = anchor === partner ? 0.4 : 0.2 + random() * 29.8;
    }
  }
  const plddt = Float32Array.from({ length }, () => 30 + random() * 68);
  return { predictedAlignedError, plddt, asymId, atom37, atom37Mask };
}

describe("interface confidence scores", () => {
  it("matches ColabFold on a dimer", () => {
    const scores = interfaceScores(complex([12, 10], 7))!;
    expect(scores).toBeDefined();
    expect(scores.ipsae["A-B"]).toBeCloseTo(0.28847915, 6);
    expect(scores.ipsae["B-A"]).toBeCloseTo(0.37090337, 6);
    expect(scores.pdockq2["A-B"]).toBeCloseTo(0.03322730, 6);
    expect(scores.pdockq2["B-A"]).toBeCloseTo(0.02983343, 6);
  });

  it("matches ColabFold on a trimer where one pair does not touch", () => {
    const scores = interfaceScores(complex([9, 7, 6], 9))!;
    expect(scores.ipsae["A-B"]).toBeCloseTo(0.31839003, 6);
    expect(scores.ipsae["B-A"]).toBeCloseTo(0.47274426, 6);
    expect(scores.ipsae["B-C"]).toBeCloseTo(0.22812681, 6);
    expect(scores.ipsae["C-B"]).toBeCloseTo(0.26826403, 6);
    expect(scores.pdockq2["A-B"]).toBeCloseTo(0.05912389, 6);
    expect(scores.pdockq2["C-B"]).toBeCloseTo(0.03911966, 6);
    // A and C are two separations apart, so they share no contact.
    expect(scores.pdockq2["A-C"]).toBe(0);
    expect(scores.pdockq2["C-A"]).toBe(0);
    // ipSAE reads the error, not the coordinates, so a distant pair still scores.
    expect(scores.ipsae["A-C"]).toBeCloseTo(0.76906867, 6);
  });

  it("gives one number an interface, the better of its two directions", () => {
    const scores = interfaceScores(complex([12, 10], 7))!;
    expect(interfaceMaxima(scores.ipsae)).toEqual({ "A-B": scores.ipsae["B-A"] });
    expect(interfaceMaxima({})).toEqual({});
  });

  it("has nothing to say about a single chain", () => {
    expect(interfaceScores(complex([12], 7))).toBeUndefined();
  });

  it("rejects inputs that disagree about the length, or interleave the chains", () => {
    const dimer = complex([12, 10], 7);
    expect(() => interfaceScores({ ...dimer, plddt: new Float32Array(3) }))
      .toThrow(/do not match the chain identifiers/);
    expect(() => interfaceScores({ ...dimer, atom37Mask: new Float32Array(3) }))
      .toThrow(/atom37 output does not match/);
    // Every loop walks a chain as a range, so a chain in two runs is refused.
    const interleaved = Float32Array.from(dimer.asymId, (_, index) => (index % 2) + 1);
    expect(() => interfaceScores({ ...dimer, asymId: interleaved }))
      .toThrow(/one run of residues/);
  });
});
