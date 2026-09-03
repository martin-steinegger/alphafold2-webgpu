import { describe, expect, it } from "vitest";
import { remainingPhrase, remainingTrunkSeconds } from "../web/progress.js";

const shape = { extraBlocks: 4, mainBlocks: 48, recycles: 3 } as const;
const rates = { extraSeconds: 10, mainSeconds: 20 } as const;

describe("time remaining in the trunk", () => {
  it("says nothing until a block of each stack has been timed", () => {
    const position = { phase: "evoformer", completed: 1, recycle: 0 } as const;
    expect(remainingTrunkSeconds(position, shape, { mainSeconds: 20 })).toBeUndefined();
    expect(remainingTrunkSeconds(position, shape, { extraSeconds: 10 })).toBeUndefined();
  });

  it("counts the rest of this stack, the stack after it and the recycles left", () => {
    // Two extra blocks done in the first of four recycles: two extra blocks
    // and all 48 main blocks left here, then three whole recycles.
    const seconds = remainingTrunkSeconds(
      { phase: "extra-msa", completed: 2, recycle: 0 }, shape, rates);
    expect(seconds).toBe(2 * 10 + 48 * 20 + 3 * (4 * 10 + 48 * 20));
  });

  it("counts only the main stack once the extra stack has finished", () => {
    const seconds = remainingTrunkSeconds(
      { phase: "evoformer", completed: 40, recycle: 3 }, shape, rates);
    expect(seconds).toBe(8 * 20);
  });

  it("counts only the recycles left once the structure module runs", () => {
    const seconds = remainingTrunkSeconds(
      { phase: "structure", completed: 1, recycle: 1 }, shape, rates);
    expect(seconds).toBe(2 * (4 * 10 + 48 * 20));
  });

  it("reaches zero on the last block of the last recycle", () => {
    const seconds = remainingTrunkSeconds(
      { phase: "evoformer", completed: 48, recycle: 3 }, shape, rates);
    expect(seconds).toBe(0);
  });

  it("phrases only what is worth saying", () => {
    expect(remainingPhrase(undefined)).toBe("");
    expect(remainingPhrase(12)).toBe("");
    expect(remainingPhrase(45)).toBe(" · about 1 min left");
    expect(remainingPhrase(20 * 60)).toBe(" · about 20 min left");
  });
});
