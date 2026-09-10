import { afterEach, describe, expect, it, vi } from "vitest";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";

/** Restored between cases: these are process-wide. */
const original = { cuda: process.env.CUDA_VISIBLE_DEVICES, prime: process.env.DRI_PRIME };

afterEach(() => {
  vi.restoreAllMocks();
  for (const [name, value] of [["CUDA_VISIBLE_DEVICES", original.cuda], ["DRI_PRIME", original.prime]] as const) {
    if (value === undefined) delete process.env[name]; else process.env[name] = value;
  }
});

function warningsFrom(environment: Record<string, string | undefined>): readonly string[] {
  for (const [name, value] of Object.entries(environment)) {
    if (value === undefined) delete process.env[name]; else process.env[name] = value;
  }
  const warnings: string[] = [];
  vi.spyOn(console, "error").mockImplementation((message: unknown) => { warnings.push(String(message)); });
  dawnInstanceFlags({ unclamped: true });
  return warnings;
}

describe("asking for a card that nothing selected", () => {
  it("warns when CUDA_VISIBLE_DEVICES is set and DRI_PRIME is not", () => {
    // The failure this catches: a harness that never calls selectGpu ignores
    // the variable in silence and measures whichever card the loader picks.
    // One that did read a dimer at 37.2 s a recycle against a real 12.0.
    const warnings = warningsFrom({ CUDA_VISIBLE_DEVICES: "3", DRI_PRIME: undefined });
    expect(warnings).toHaveLength(1);
    expect(warnings[0]).toMatch(/DRI_PRIME is unset/);
    expect(warnings[0]).toMatch(/selectGpu/);
  });

  it("says nothing once the card has been selected", () => {
    expect(warningsFrom({ CUDA_VISIBLE_DEVICES: "3", DRI_PRIME: "pci-0000_3b_00_0" })).toEqual([]);
  });

  it("says nothing when no card was asked for", () => {
    expect(warningsFrom({ CUDA_VISIBLE_DEVICES: undefined, DRI_PRIME: undefined })).toEqual([]);
  });

  it("still returns the flags an instance needs", () => {
    expect(dawnInstanceFlags({ unclamped: true }).join(" ")).toMatch(/enable-dawn-features=/);
  });
});
