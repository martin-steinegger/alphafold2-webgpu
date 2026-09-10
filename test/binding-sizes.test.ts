import { afterEach, describe, expect, it } from "vitest";
import {
  maximumPredictionLength, oversizedBindings, predictionBindingSizes,
  STORAGE_BINDING_LIMIT_BYTES, triangleSlotsFit,
} from "../src/runtime/binding-sizes.js";
import { setScratchBudgetScale } from "../src/runtime/scratch-budget.js";

/** The scale a native host fits to itself; a browser leaves it at one. */
const NATIVE_SCALE = 16;
afterEach(() => setScratchBudgetScale(1));

const MONOMER = { msaSequences: 512, extraSequences: 5120 } as const;
const MULTIMER = { msaSequences: 252, extraSequences: 1152, multimer: true } as const;

const named = (shape: Parameters<typeof predictionBindingSizes>[0], label: string) =>
  predictionBindingSizes(shape).find((size) => size.label === label);

describe("what a prediction binds, without running one", () => {
  it("reproduces the binding that stopped a tetramer", () => {
    // A 3,300-residue tetramer failed with "Binding size (2787840000) ... is
    // larger than the maximum storage buffer binding size (2147483644)". The
    // tensor was the pair residual, and this is the same number.
    setScratchBudgetScale(NATIVE_SCALE);
    const residual = named({ ...MULTIMER, length: 3300 }, "pair.residual-add");
    expect(residual?.totalBytes).toBe(2_787_840_000);
    expect(residual?.totalBytes).toBeGreaterThan(STORAGE_BINDING_LIMIT_BYTES);
    // Windowed since 01c4f40, so the binding is the limit rather than the tensor.
    expect(residual?.kind).toBe("window");
    expect(residual?.bindingBytes).toBeLessThanOrEqual(STORAGE_BINDING_LIMIT_BYTES);
  });

  it("passes the three shapes that have actually been folded", () => {
    setScratchBudgetScale(NATIVE_SCALE);
    for (const shape of [
      { ...MONOMER, length: 825 }, { ...MULTIMER, length: 1650 }, { ...MULTIMER, length: 3300 },
    ]) {
      expect(oversizedBindings(shape)).toEqual([]);
    }
  });

  it("shards the pair rather than capping on it", () => {
    setScratchBudgetScale(NATIVE_SCALE);
    // One binding at 1,650 residues and two at 3,300, which is what the
    // planner reported when the shard theory was tested against a device.
    expect(named({ ...MULTIMER, length: 1650 }, "pair")?.kind).toBe("whole");
    expect(named({ ...MULTIMER, length: 3300 }, "pair")?.kind).toBe("shards");
  });

  it("names the next ceiling and where it falls", () => {
    setScratchBudgetScale(NATIVE_SCALE);
    // 12 heads times L squared times four bytes, bound whole: sqrt(2 GiB / 48).
    expect(maximumPredictionLength(MONOMER)).toBe(6688);
    expect(maximumPredictionLength(MULTIMER)).toBe(6688);
    // Two tensors sit at that ceiling, not one: invariant point attention
    // scores every head over the pairs it also biases by them, so lifting the
    // bias alone would move nothing.
    const over = oversizedBindings({ ...MULTIMER, length: 6689 });
    expect(over.map((size) => size.label).sort()).toEqual(["ipa.logits", "ipa.pair-bias"]);
  });

  it("covers every quadratic tensor a fold was seen to bind", () => {
    // probeBindings recorded all 229 binding labels of a fold at two lengths.
    // These are the ones whose size grew with the square of the length, read
    // back to their allocations because a pooled buffer keeps a stale label.
    setScratchBudgetScale(NATIVE_SCALE);
    const covered = new Set(predictionBindingSizes({ ...MULTIMER, length: 384 })
      .map((size) => size.label));
    for (const label of [
      "pair", "triangle.whole", "triangle.output", "triangle.statistics",
      "opm.pair-count", "monomer.pair-mask", "attention.pair-bias",
      "multimer-template.pair", "multimer-template.query-statistics",
      "confidence.pae-logits", "confidence.predicted-aligned-error", "confidence.tm-score-terms",
      "ipa.pair-bias", "ipa.logits", "ipa.pair-statistics",
      "multimer-template.pair-update", "multimer-template.attention.pair-bias",
    ]) expect(covered).toContain(label);
  });

  it("puts the tensors that only count pairs far above the ones that channel them", () => {
    setScratchBudgetScale(NATIVE_SCALE);
    // One value a pair reaches the limit at 23,170 residues, which no card
    // holds a pair for. They are listed to show they were checked, not because
    // they are near.
    expect(maximumPredictionLength({ ...MULTIMER,
      bindingLimitBytes: STORAGE_BINDING_LIMIT_BYTES }, 32_768)).toBe(6688);
    const single = named({ ...MULTIMER, length: 23_170 }, "monomer.pair-mask");
    expect(single?.bindingBytes).toBeLessThanOrEqual(STORAGE_BINDING_LIMIT_BYTES);
    expect(named({ ...MULTIMER, length: 23_171 }, "monomer.pair-mask")?.exceedsBy)
      .toBeGreaterThan(0);
  });

  it("costs the unreduced confidence head its own much lower ceiling", () => {
    setScratchBudgetScale(NATIVE_SCALE);
    // L squared times 64 bins times four bytes, bound whole: sqrt(2 GiB / 256).
    expect(maximumPredictionLength({ ...MULTIMER, reducedConfidence: false })).toBe(2896);
    expect(oversizedBindings({ ...MULTIMER, length: 2897, reducedConfidence: false })
      .map((size) => size.label)).toEqual(["confidence.pae-logits"]);
  });

  it("does not let the scratch budget change what fits a binding", () => {
    // The budget decides window sizes, the limit decides bindings, and a
    // bigger budget must never push a window past the limit.
    for (const scale of [1, 16, 64]) {
      setScratchBudgetScale(scale);
      for (const size of predictionBindingSizes({ ...MULTIMER, length: 3300 })) {
        expect(size.bindingBytes).toBeLessThanOrEqual(STORAGE_BINDING_LIMIT_BYTES);
      }
    }
  });
});

describe("the second ceiling: bindings a stage may make", () => {
  it("reports only the size ceiling when the count is not given", () => {
    // What this always did, and what a caller that has not asked still gets.
    setScratchBudgetScale(NATIVE_SCALE);
    expect(maximumPredictionLength(MONOMER)).toBe(6688);
  });

  it("caps a browser far lower than any binding size does", () => {
    // 128 MiB and eight bindings is the WebGPU default. Every binding a
    // 1,500-residue fold makes is comfortably inside the size limit; the
    // triangle still cannot bind them all in one stage.
    setScratchBudgetScale(1);
    const browser = { ...MONOMER, bindingLimitBytes: 128 * 1024 ** 2, storageBuffersPerStage: 8 };
    expect(oversizedBindings({ ...browser, length: 1500 })).toEqual([]);
    expect(triangleSlotsFit({ ...browser, length: 1500 })).toBe(false);
    expect(maximumPredictionLength(browser)).toBe(1448);
  });

  it("moves with the bindings the adapter reports, until something else bites", () => {
    setScratchBudgetScale(1);
    const at = (perStage: number) => maximumPredictionLength(
      { ...MONOMER, bindingLimitBytes: 128 * 1024 ** 2, storageBuffersPerStage: perStage });
    // Eight is the triangle's own ceiling. Sixteen is not: the triangle would
    // run to 2,508 there, and ipa.pair-bias stops the model at 1,672 first.
    expect(at(8)).toBe(1448);
    expect(at(16)).toBe(1672);
    const browser = { ...MONOMER, bindingLimitBytes: 128 * 1024 ** 2, storageBuffersPerStage: 16 };
    expect(triangleSlotsFit({ ...browser, length: 1672 })).toBe(true);
    expect(oversizedBindings({ ...browser, length: 1673 }).map((size) => size.label))
      .toEqual(["ipa.pair-bias", "ipa.logits"]);
  });

  it("stops being the constraint where a binding covers 2 GiB", () => {
    // The triangle would run to 10,033 there; ipa.pair-bias caps it first.
    setScratchBudgetScale(NATIVE_SCALE);
    expect(maximumPredictionLength({ ...MONOMER, storageBuffersPerStage: 16 })).toBe(6688);
  });
});
