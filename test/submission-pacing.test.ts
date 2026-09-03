/**
 * How large a command buffer the trunk builds.
 *
 * A driver that waits seconds for one buffer may decide the GPU has hung, so
 * the dispatch count follows the measured duration rather than the shape.
 */
import { describe, expect, it } from "vitest";
import { nextSubmissionDispatchLimit } from "../src/runtime/execution.js";

describe("command buffer pacing", () => {
  it("cuts the count in proportion when a buffer runs past the target", () => {
    // 200 dispatches took a second, so a quarter of them is a quarter second.
    expect(nextSubmissionDispatchLimit(200, 1000, 200)).toBe(50);
  });

  it("cuts by what the buffer measured, not by what the limit was", () => {
    // A buffer smaller than the limit still says how long a dispatch takes.
    expect(nextSubmissionDispatchLimit(384, 500, 40)).toBe(20);
  });

  it("climbs a tenth at a time while buffers stay under the target", () => {
    expect(nextSubmissionDispatchLimit(100, 50, 100)).toBe(110);
    expect(nextSubmissionDispatchLimit(8, 1, 8)).toBe(9);
  });

  it("never leaves the range, however extreme the measurement", () => {
    expect(nextSubmissionDispatchLimit(8, 100_000, 8)).toBe(8);
    let limit = 8;
    for (let step = 0; step < 200; step += 1) limit = nextSubmissionDispatchLimit(limit, 1, limit);
    expect(limit).toBe(384);
  });

  it("ignores a measurement it cannot use", () => {
    expect(nextSubmissionDispatchLimit(64, 0, 10)).toBe(64);
    expect(nextSubmissionDispatchLimit(64, 10, 0)).toBe(64);
  });

  it("settles where buffers take about the target", () => {
    // A dispatch costs a millisecond: the count should sit near 250.
    let limit = 48;
    for (let step = 0; step < 100; step += 1) limit = nextSubmissionDispatchLimit(limit, limit, limit);
    expect(limit).toBeGreaterThan(200);
    expect(limit).toBeLessThan(300);
  });
});
