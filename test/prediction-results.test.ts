import { describe, expect, it } from "vitest";
import { predictionToPdb, safeJobName } from "../web/prediction-results.js";

describe("browser prediction result formatting", () => {
  it("writes only present atom37 coordinates and pLDDT B-factors", () => {
    const atom37 = new Float32Array(37 * 3);
    atom37.set([1.25, -2.5, 3.75], 0);
    const atom37Mask = new Float32Array(37); atom37Mask[0] = 1;
    const pdb = predictionToPdb("A", {
      atom14: new Float32Array(), atom37, atom37Mask, finalRepresentation: new Float32Array(),
      affine: new Float32Array(), angles: new Float32Array(), unnormalizedAngles: new Float32Array(),
      elapsedMilliseconds: 0,
    }, Float32Array.of(97.25));
    expect(pdb).toContain("ATOM      1    N ALA A   1");
    expect(pdb).toContain("   1.250  -2.500   3.750  1.00 97.25");
    expect(pdb.endsWith("TER\nEND\n")).toBe(true);
  });

  it("makes download names safe", () => {
    expect(safeJobName(" ../../my fold ")).toBe("my_fold");
    expect(safeJobName("***")).toBe("prediction");
  });

  it("writes distinct PDB chains with residue numbering reset", () => {
    const atom37 = new Float32Array(2 * 37 * 3);
    const atom37Mask = new Float32Array(2 * 37); atom37Mask[0] = 1; atom37Mask[37] = 1;
    const pdb = predictionToPdb("AG", {
      atom14: new Float32Array(), atom37, atom37Mask, finalRepresentation: new Float32Array(),
      affine: new Float32Array(), angles: new Float32Array(), unnormalizedAngles: new Float32Array(),
      elapsedMilliseconds: 0,
    }, Float32Array.of(80, 70), [1, 1]);
    expect(pdb).toContain(" N ALA A   1");
    expect(pdb).toContain(" N GLY B   1");
    expect(pdb.match(/TER/g)).toHaveLength(2);
  });
});
