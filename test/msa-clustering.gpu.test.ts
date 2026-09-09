import { describe, expect, it, beforeAll } from "vitest";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import {
  assignNearestCentres, nearestCentreSets, tieSetWords,
} from "../src/input/msa-clustering-webgpu.js";
import { testGpu } from "./support/gpu-instance.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
let gpu: GPU;
let device: GPUDevice;

/** The host loop this kernel replaces, copied so the two cannot drift apart. */
function hostAssign(
  centres: Uint8Array, nc: number, extras: Uint8Array, ne: number, length: number,
): Uint32Array {
  const out = new Uint32Array(ne);
  for (let e = 0; e < ne; e += 1) {
    let best = 0; let bestScore = -1;
    for (let c = 0; c < nc; c += 1) {
      let score = 0;
      for (let r = 0; r < length; r += 1) {
        const code = centres[c * length + r]!;
        if (code <= 20 && code === extras[e * length + r]!) score += 1;
      }
      if (score > bestScore) { bestScore = score; best = c; }
    }
    out[e] = best;
  }
  return out;
}

/** The multimer host loop, which keeps every centre tied at the best agreement. */
function hostTies(
  centres: Uint8Array, nc: number, extras: Uint8Array, ne: number, length: number,
  valid: Uint8Array,
): number[][] {
  const out: number[][] = [];
  for (let e = 0; e < ne; e += 1) {
    let bestAgreement = -1; const nearest: number[] = [];
    for (let c = 0; c < nc; c += 1) {
      if (valid[c] === 0) continue;
      let agreement = 0;
      for (let r = 0; r < length; r += 1) {
        const code = centres[c * length + r]!;
        if (code <= 20 && code === extras[e * length + r]!) agreement += 1;
      }
      if (agreement > bestAgreement) { bestAgreement = agreement; nearest.length = 0; nearest.push(c); }
      else if (agreement === bestAgreement) nearest.push(c);
    }
    out.push(nearest);
  }
  return out;
}

function setMembers(sets: Uint32Array, extra: number, words: number, nc: number): number[] {
  const members: number[] = [];
  for (let c = 0; c < nc; c += 1) {
    if ((sets[extra * words + (c >>> 5)]! & (1 << (c & 31))) !== 0) members.push(c);
  }
  return members;
}

describe.skipIf(!enabled)("nearest cluster centre on the GPU", () => {
  beforeAll(async () => {
    gpu = testGpu({ unclamped: true });
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    device = await requestAlphaFoldDevice(adapter!);
  });

  let state = 7 >>> 0;
  const rand = (): number => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296;

  it.each([
    [17, 3, 5], [64, 16, 32], [200, 300, 64], [128, 1, 4],
  ])("matches the host loop at %i residues, %i centres, %i extras", async (L, nc, ne) => {
    // Codes run past 20 so the gap and mask guard is exercised, and a tie must
    // fall to the lowest centre, which duplicate centres make certain.
    const centres = Uint8Array.from({ length: nc * L }, () => Math.floor(rand() * 23));
    if (nc > 1) centres.copyWithin(L, 0, L);
    const extras = Uint8Array.from({ length: ne * L }, () => Math.floor(rand() * 23));
    const got = await assignNearestCentres(device, centres, nc, extras, ne, L);
    expect([...got]).toEqual([...hostAssign(centres, nc, extras, ne, L)]);
  });

  it.each([
    [17, 3, 5], [64, 16, 32], [200, 300, 64], [40, 70, 8],
  ])("returns the whole tie set at %i residues, %i centres, %i extras", async (L, nc, ne) => {
    // Duplicated centres force ties, and a masked centre must stay out of the
    // search without shifting the indices of the ones that remain.
    const centres = Uint8Array.from({ length: nc * L }, () => Math.floor(rand() * 23));
    for (let c = 1; c < nc; c += 3) centres.copyWithin(c * L, 0, L);
    const extras = Uint8Array.from({ length: ne * L }, () => Math.floor(rand() * 23));
    const valid = Uint8Array.from({ length: nc }, (_, c) => (c % 5 === 4 ? 0 : 1));
    const words = tieSetWords(nc);
    const sets = await nearestCentreSets(device, centres, nc, extras, ne, L, valid);
    const want = hostTies(centres, nc, extras, ne, L, valid);
    for (let e = 0; e < ne; e += 1) {
      expect(setMembers(sets, e, words, nc)).toEqual(want[e]);
    }
  });

  it("comes back empty when no centre is valid, which is a fully masked block", async () => {
    const sets = await nearestCentreSets(
      device, new Uint8Array(2 * 4), 2, new Uint8Array(3 * 4), 3, 4, new Uint8Array(2));
    expect([...sets]).toEqual([0, 0, 0]);
  });

  it("refuses codes that are not one a residue", async () => {
    await expect(assignNearestCentres(device, new Uint8Array(3), 2, new Uint8Array(4), 2, 2))
      .rejects.toThrow(/one code a residue/);
  });
});
