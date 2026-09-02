import { describe, expect, it } from "vitest";
import {
  CLUSTERED_MSA_CHANNELS, MSA_CODE_NONE, compactClusteredMsaFeatures, expandClusteredMsaFeatures,
} from "../src/input/msa-features.js";

function denseSlot(code: number | undefined, hasDeletion: number, deletionValue: number,
  profile: readonly number[], deletionMean: number): Float32Array {
  const dense = new Float32Array(49);
  if (code !== undefined) dense[code] = 1;
  dense[23] = hasDeletion; dense[24] = deletionValue;
  profile.forEach((value, index) => { dense[25 + index] = value; });
  dense[48] = deletionMean;
  return dense;
}

describe("compact clustered MSA features", () => {
  const profile = Array.from({ length: 23 }, (_, index) => index / 100);

  it("round-trips a row that has a one-hot", () => {
    const dense = denseSlot(7, 1, 0.25, profile, 0.5);
    const compact = compactClusteredMsaFeatures(dense, 1);
    expect(compact).toHaveLength(CLUSTERED_MSA_CHANNELS);
    expect(compact[0]).toBe(7);
    expect(compact[1]).toBe(1);
    expect(compact[2]).toBe(0.25);
    expect(compact[26]).toBe(0.5);
    expect(Array.from(expandClusteredMsaFeatures(compact, 1))).toEqual(Array.from(dense));
  });

  it("marks a masked row, whose one-hot is entirely zero", () => {
    const dense = denseSlot(undefined, 0, 0, new Array(23).fill(0), 0.75);
    const compact = compactClusteredMsaFeatures(dense, 1);
    expect(compact[0]).toBe(MSA_CODE_NONE);
    expect(Array.from(expandClusteredMsaFeatures(compact, 1))).toEqual(Array.from(dense));
  });

  it("round-trips several rows at once", () => {
    const rows = [denseSlot(0, 0, 0.1, profile, 0.2), denseSlot(22, 1, 0.3, profile, 0.4),
      denseSlot(undefined, 0, 0, new Array(23).fill(0), 0)];
    const dense = new Float32Array(rows.length * 49);
    rows.forEach((row, index) => dense.set(row, index * 49));
    const compact = compactClusteredMsaFeatures(dense, rows.length);
    expect(Array.from(expandClusteredMsaFeatures(compact, rows.length))).toEqual(Array.from(dense));
  });

  it("rejects a dense row that is not a one-hot", () => {
    const dense = denseSlot(3, 0, 0, new Array(23).fill(0), 0);
    dense[5] = 1;
    expect(() => compactClusteredMsaFeatures(dense, 1)).toThrow(/more than one entry/);
    const scaled = denseSlot(3, 0, 0, new Array(23).fill(0), 0);
    scaled[3] = 0.5;
    expect(() => compactClusteredMsaFeatures(scaled, 1)).toThrow(/not one/);
  });

  it("rejects arrays that do not match the row count", () => {
    expect(() => compactClusteredMsaFeatures(new Float32Array(50), 1)).toThrow(/49 channels/);
    expect(() => expandClusteredMsaFeatures(new Float32Array(28), 1)).toThrow(/27 channels/);
  });
});
