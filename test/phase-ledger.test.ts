import { afterEach, describe, expect, it } from "vitest";
import {
  PhaseLedger, formatPhaseReport, timed, timedSync, setPhaseLedger,
  type PhaseReport, type PhaseRow,
} from "../src/runtime/phase-ledger.js";

const spin = (ms: number): void => {
  const until = performance.now() + ms;
  while (performance.now() < until) { /* a busy wait measures the same clock */ }
};

const row = (report: PhaseReport, name: string): PhaseRow | undefined =>
  report.rows.find((candidate) => candidate.name === name);

afterEach(() => { setPhaseLedger(undefined); });

describe("the phase ledger", () => {
  it("accounts for every millisecond, so the rows sum to the total", () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    timedSync("a", () => spin(10));
    spin(10); // outside any phase
    timedSync("b", () => spin(10));
    const report = ledger.report();
    const summed = report.rows.reduce((sum, r) => sum + r.milliseconds, 0);
    expect(summed).toBeCloseTo(report.totalMilliseconds, 6);
  });

  it("names the time no phase claimed rather than losing it", () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    timedSync("named", () => spin(5));
    spin(20);
    const report = ledger.report();
    expect(row(report, "unaccounted")!.milliseconds).toBeGreaterThan(10);
  });

  it("charges a nested phase to itself and not also to the outer one", () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    timedSync("outer", () => { spin(5); timedSync("inner", () => spin(20)); spin(5); });
    const report = ledger.report();
    expect(row(report, "inner")!.milliseconds).toBeGreaterThan(15);
    // Exclusive: the outer keeps its own 10 ms and not the inner's 20.
    expect(row(report, "outer")!.milliseconds).toBeLessThan(18);
    const summed = report.rows.reduce((sum, r) => sum + r.milliseconds, 0);
    expect(summed).toBeCloseTo(report.totalMilliseconds, 6);
  });

  it("keeps the first entry apart, which is where a compile lands", () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    for (const cost of [40, 5, 5]) timedSync("recycle", () => spin(cost));
    const recycle = row(ledger.report(), "recycle")!;
    expect(recycle.count).toBe(3);
    expect(recycle.firstMilliseconds).toBeGreaterThan(30);
    // The rest average far below it, which is the cold start showing up.
    expect((recycle.milliseconds - recycle.firstMilliseconds) / 2).toBeLessThan(20);
  });

  it("never reports a first entry costing more than the phase's total", () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    timedSync("outer", () => timedSync("inner", () => spin(15)));
    for (const r of ledger.report().rows) {
      expect(r.firstMilliseconds).toBeLessThanOrEqual(r.milliseconds + 1e-6);
    }
  });

  it("awaits an asynchronous phase", async () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    await timed("slow", async () => { await new Promise((r) => setTimeout(r, 25)); });
    expect(row(ledger.report(), "slow")!.milliseconds).toBeGreaterThan(15);
  });

  it("charges a phase that throws and then lets the error out", () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    expect(() => timedSync("bad", () => { spin(10); throw new Error("boom"); })).toThrow("boom");
    expect(row(ledger.report(), "bad")!.milliseconds).toBeGreaterThan(5);
  });

  it("runs the body untouched when no ledger is installed", () => {
    expect(timedSync("off", () => 41 + 1)).toBe(42);
  });

  it("formats a report whose rows carry their share", () => {
    const ledger = new PhaseLedger();
    setPhaseLedger(ledger);
    timedSync("featurise", () => spin(10));
    const text = formatPhaseReport(ledger.report());
    expect(text).toContain("featurise");
    expect(text).toMatch(/total/);
    expect(text).toMatch(/%/);
  });
});
