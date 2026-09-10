import { describe, expect, it } from "vitest";
import {
  fitScratchBudgetScale, SCRATCH_BUDGET_SCALES, scratchBudgetScalesFor,
  WIDE_SCRATCH_BUDGET_SCALES, WIDE_SCRATCH_RESIDUES,
} from "../src/runtime/dawn.js";

describe("which scratch budget scales a length is offered", () => {
  it("keeps a short chain off the wide rungs", () => {
    // An 825-residue monomer reads 4213 ms a recycle at sixteen and 5461 at
    // sixty-four, for twice the resident memory. Offering it would lose 28%.
    expect(scratchBudgetScalesFor(825)).toEqual(SCRATCH_BUDGET_SCALES);
    expect(scratchBudgetScalesFor(825)).not.toContain(64);
    expect(scratchBudgetScalesFor(WIDE_SCRATCH_RESIDUES)).not.toContain(64);
  });

  it("offers them to a chain long enough to have won on them", () => {
    // 1,650 reads 11897 ms against 12500, and 3,300 reads 62474 against 74788.
    for (const residues of [WIDE_SCRATCH_RESIDUES + 1, 1650, 3300, 6688]) {
      expect(scratchBudgetScalesFor(residues)).toEqual(WIDE_SCRATCH_BUDGET_SCALES);
    }
  });

  it("offers every rung widest first, so the fit takes the best that fits", () => {
    for (const ladder of [SCRATCH_BUDGET_SCALES, WIDE_SCRATCH_BUDGET_SCALES]) {
      expect([...ladder].sort((a, b) => b - a)).toEqual([...ladder]);
      expect(ladder[ladder.length - 1]).toBe(1);
    }
  });

  it("still lets the memory estimate decide", () => {
    const ladder = scratchBudgetScalesFor(3300);
    // A host with room takes the widest rung on offer.
    expect(fitScratchBudgetScale(() => 1, Number.MAX_SAFE_INTEGER, ladder)).toBe(64);
    // One that can only afford the narrow ones gets the widest of those: a
    // long chain on a small card must still run, not fail to be offered.
    expect(fitScratchBudgetScale((scale) => scale * 100, 1600, ladder)).toBe(16);
    expect(fitScratchBudgetScale((scale) => scale * 100, 100, ladder)).toBe(1);
  });

  it("falls back to the narrowest rather than refusing", () => {
    // Nothing fits, so the caller still gets a scale it can run at.
    expect(fitScratchBudgetScale(() => Number.MAX_SAFE_INTEGER, 1, scratchBudgetScalesFor(3300))).toBe(1);
  });
});
