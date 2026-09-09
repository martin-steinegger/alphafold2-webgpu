import { describe, expect, it, beforeAll, afterAll } from "vitest";
import { create, globals } from "webgpu";
import { CLUSTERED_MSA_CHANNELS } from "../src/input/msa-features.js";
import { clusterProfile } from "../src/input/msa-profile-webgpu.js";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
let device: GPUDevice;

const deletionValue = (value: number): number => Math.atan(value / 3) * 2 / Math.PI;

/** The host loops this kernel replaces, copied so the two cannot drift apart. */
function hostProfile(
  centreCodes: Uint8Array, centres: number, extraCodes: Uint8Array, extras: number,
  length: number, assignments: Uint32Array,
  centreDeletion: Float32Array, extraDeletion: Float32Array,
): Float32Array {
  const features = new Float32Array(centres * length * CLUSTERED_MSA_CHANNELS);
  const deletionSums = new Float32Array(centres * length);
  const counts = new Float32Array(centres * length).fill(1 + 1e-6);
  for (let centre = 0; centre < centres; centre += 1) for (let r = 0; r < length; r += 1) {
    const slot = centre * length + r;
    features[slot * CLUSTERED_MSA_CHANNELS + 3 + centreCodes[slot]!] = 1;
    deletionSums[slot] = centreDeletion[slot]!;
  }
  for (let extra = 0; extra < extras; extra += 1) {
    const centre = assignments[extra]!;
    if (centre >= centres) continue;
    for (let r = 0; r < length; r += 1) {
      const slot = centre * length + r;
      counts[slot] = counts[slot]! + 1;
      const channel = slot * CLUSTERED_MSA_CHANNELS + 3 + extraCodes[extra * length + r]!;
      features[channel] = features[channel]! + 1;
      deletionSums[slot] = deletionSums[slot]! + extraDeletion[extra * length + r]!;
    }
  }
  for (let centre = 0; centre < centres; centre += 1) for (let r = 0; r < length; r += 1) {
    const slot = centre * length + r; const out = slot * CLUSTERED_MSA_CHANNELS;
    features[out] = centreCodes[slot]!;
    const deletion = centreDeletion[slot]!;
    features[out + 1] = Math.min(deletion, 1);
    features[out + 2] = deletionValue(deletion);
    for (let code = 0; code < 23; code += 1) {
      features[out + 3 + code] = features[out + 3 + code]! / counts[slot]!;
    }
    features[out + 26] = deletionValue(deletionSums[slot]! / counts[slot]!);
  }
  return features;
}

describe.skipIf(!enabled)("the cluster profile on the GPU", () => {
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const gpu = create(dawnInstanceFlags({ unclamped: true }));
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    device = await requestAlphaFoldDevice(adapter!);
  });

  let state = 11 >>> 0;
  const rand = (): number => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 4294967296;

  it.each([
    [17, 3, 5], [64, 16, 40], [200, 48, 96], [128, 1, 7],
  ])("matches the host loops at %i residues, %i centres, %i extras", async (L, nc, ne) => {
    const centreCodes = Uint8Array.from({ length: nc * L }, () => Math.floor(rand() * 23));
    const extraCodes = Uint8Array.from({ length: ne * L }, () => Math.floor(rand() * 23));
    // Uneven buckets, including centres nothing lands on.
    const assignments = Uint32Array.from({ length: ne },
      () => Math.floor(rand() * rand() * nc) % nc);
    const centreDeletion = Float32Array.from({ length: nc * L }, () => Math.floor(rand() * 5));
    const extraDeletion = Float32Array.from({ length: ne * L }, () => Math.floor(rand() * 5));
    const got = await clusterProfile(device, {
      centreCodes, centres: nc, extraCodes, extras: ne, length: L,
      assignments, centreDeletion, extraDeletion,
    });
    const want = hostProfile(centreCodes, nc, extraCodes, ne, L, assignments,
      centreDeletion, extraDeletion);
    expect(got.length).toBe(want.length);
    // The host divides and takes atan in f64 before rounding to f32; the kernel
    // works in f32 throughout, so the two agree to f32 rounding, not bit for
    // bit. Channels 2 and 26 are the only ones through atan, for which WGSL
    // promises no particular accuracy: they agree to 2.1e-6 on llvmpipe and far
    // closer on this NVIDIA card, so they are held apart from the rest. The
    // profile itself divides small integers and has no such excuse.
    const worst = [0, 0];
    for (let index = 0; index < want.length; index += 1) {
      const channel = index % CLUSTERED_MSA_CHANNELS;
      const group = channel === 2 || channel === 26 ? 0 : 1;
      worst[group] = Math.max(worst[group]!, Math.abs(got[index]! - want[index]!));
    }
    expect(worst[0]).toBeLessThan(1e-5);
    expect(worst[1]).toBeLessThan(1e-6);
  });

  it("drops an extra row assigned to no centre, as the host loop did", async () => {
    const shared = {
      centreCodes: new Uint8Array([1, 2]), centres: 1,
      length: 2, centreDeletion: new Float32Array([0, 0]),
    };
    const dropped = await clusterProfile(device, {
      ...shared, extraCodes: new Uint8Array([1, 2]), extras: 1,
      assignments: Uint32Array.from([0xffffffff]), extraDeletion: new Float32Array(2),
    });
    const alone = await clusterProfile(device, {
      ...shared, extraCodes: new Uint8Array(0), extras: 0,
      assignments: new Uint32Array(0), extraDeletion: new Float32Array(0),
    });
    expect([...dropped]).toEqual([...alone]);
  });

  it("refuses codes that are not one a residue", async () => {
    await expect(clusterProfile(device, {
      centreCodes: new Uint8Array(3), centres: 2, extraCodes: new Uint8Array(4), extras: 2,
      length: 2, assignments: new Uint32Array(2),
      centreDeletion: new Float32Array(4), extraDeletion: new Float32Array(4),
    })).rejects.toThrow(/one code a residue/);
  });
});

// dawn.node pumps ProcessEvents from the event loop and deadlocks on the
// instance mutex if the device outlives the worker's threads, which shows up
// as a vitest worker exiting unexpectedly rather than as a failure.
afterAll(() => { device?.destroy(); });
