import { describe, expect, it } from "vitest";
import { summarizeMonomerRecycle, type MonomerRecycleDetails } from "../src/model/monomer.js";

describe("monomer recycle result retention", () => {
  it("retains metrics without pair, structure, or categorical logit tensors", () => {
    const details = {
      msaFirstRow: Float32Array.of(1),
      structure: { atom37: Float32Array.of(2) },
      confidence: {
        lddtLogits: Float32Array.of(3), paeLogits: Float32Array.of(4),
        plddt: Float32Array.of(90), meanPlddt: 90,
        predictedAlignedError: Float32Array.of(1), maxPredictedAlignedError: 31, ptm: 0.8,
      },
      elapsedMilliseconds: 12,
      trunkSubmissions: { embedding: 1, template: 0, extraMsa: 4, mainEvoformer: 48,
        readback: 1, total: 54 },
    } as unknown as MonomerRecycleDetails;
    const summary = summarizeMonomerRecycle(details);
    expect(summary).toEqual({
      confidence: { meanPlddt: 90, ptm: 0.8 }, elapsedMilliseconds: 12,
      trunkSubmissions: details.trunkSubmissions,
    });
    expect(summary).not.toHaveProperty("msaFirstRow");
    expect(summary).not.toHaveProperty("structure");
    expect(summary.confidence).not.toHaveProperty("paeLogits");
  });
});
