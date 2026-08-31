import { describe, expect, it } from "vitest";
import { multimerRecycleDistanceRms } from "../src/model/multimer-recycling.js";

function atoms(ca: readonly (readonly [number, number, number])[]): Float32Array {
  const output = new Float32Array(ca.length * 37 * 3);
  ca.forEach((position, residue) => output.set(position, (residue * 37 + 1) * 3));
  return output;
}

describe("AlphaFold-Multimer recycling convergence", () => {
  it("uses pairwise CA distances and is invariant to global translation", () => {
    const previous = atoms([[0, 0, 0], [2, 0, 0]]);
    const translated = atoms([[5, -1, 3], [7, -1, 3]]);
    expect(multimerRecycleDistanceRms(previous, translated, Float32Array.of(1, 1))).toBeCloseTo(1e-4, 8);
    const changed = atoms([[0, 0, 0], [4, 0, 0]]);
    // Ordered off-diagonal pairs differ by 2; diagonal pairs differ by zero.
    expect(multimerRecycleDistanceRms(previous, changed, Float32Array.of(1, 1)))
      .toBeCloseTo(Math.sqrt(2), 6);
  });

  it("excludes masked residues", () => {
    const previous = atoms([[0, 0, 0], [2, 0, 0], [0, 3, 0]]);
    const changedOnlyMasked = atoms([[0, 0, 0], [2, 0, 0], [100, 100, 100]]);
    expect(multimerRecycleDistanceRms(previous, changedOnlyMasked, Float32Array.of(1, 1, 0)))
      .toBeCloseTo(1e-4, 8);
  });
});
