import { describe, expect, it, beforeAll } from "vitest";
import { create, globals } from "webgpu";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { assignNearestCentres } from "../src/input/msa-clustering-webgpu.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
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

describe.skipIf(!enabled)("nearest cluster centre on the GPU", () => {
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const gpu = create(dawnInstanceFlags({ unclamped: true }));
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

  it("refuses codes that are not one a residue", async () => {
    await expect(assignNearestCentres(device, new Uint8Array(3), 2, new Uint8Array(4), 2, 2))
      .rejects.toThrow(/one code a residue/);
  });
});
