import { describe, expect, it } from "vitest";
import {
  attentionFlashCandidates, attentionFlashKernelForShape, calibrateAttentionFlashKernel,
  presetAttentionFlashKernel, REGISTER_QUERY_BLOCK_THRESHOLD,
} from "../src/evoformer/attention-calibration.js";
import { recordSubgroupRange } from "../src/runtime/subgroups.js";

function fakeDevice(subgroups: boolean): GPUDevice {
  const device = {
    features: new Set<GPUFeatureName>(subgroups
      ? ["subgroups" as GPUFeatureName, "subgroup-size-control" as GPUFeatureName] : []),
    limits: {
      minSubgroupSize: 32, maxSubgroupSize: 32,
      maxComputeInvocationsPerWorkgroup: 256, maxComputeWorkgroupStorageSize: 16_384,
    },
    createBuffer: () => { throw new Error("the probe must not allocate for a single candidate"); },
  } as unknown as GPUDevice;
  if (subgroups) {
    recordSubgroupRange(device, {
      info: { subgroupMinSize: 32, subgroupMaxSize: 32 },
    } as unknown as GPUAdapter);
  }
  return device;
}

describe("flash attention calibration", () => {
  it("measures the register kernel against the subgroup kernel when both are available", () => {
    expect(attentionFlashCandidates(fakeDevice(true), 32)).toEqual(["register", "subgroup-key32"]);
  });

  it("has nothing to measure when the device offers only one kernel", () => {
    expect(attentionFlashCandidates(fakeDevice(false), 32)).toEqual(["register"]);
    // Head dimensions the register kernel cannot serve leave no candidate at all.
    expect(attentionFlashCandidates(fakeDevice(false), 48)).toEqual([]);
  });

  it("skips the probe entirely when only one kernel is available", async () => {
    // fakeDevice throws from createBuffer, so this resolving at all proves no probe ran.
    const kernel = await calibrateAttentionFlashKernel(fakeDevice(false), 32);
    expect(kernel.variant).toBe("register");
  });

  it("caches one verdict per device and head dimension", async () => {
    const device = fakeDevice(false);
    const first = await calibrateAttentionFlashKernel(device, 32);
    expect(await calibrateAttentionFlashKernel(device, 32)).toBe(first);
  });

  it("honours an explicit override without measuring", async () => {
    const device = fakeDevice(true);
    presetAttentionFlashKernel(device, 32, "subgroup-8x64");
    expect((await calibrateAttentionFlashKernel(device, 32)).variant).toBe("subgroup-8x64");
  });

  it("falls back to the static heuristic when the probe cannot run", async () => {
    // A subgroup-capable device whose buffers fail: two candidates, so the probe
    // is attempted, and every allocation throws.
    const device = fakeDevice(true);
    expect((await calibrateAttentionFlashKernel(device, 32)).variant).toBe("subgroup-key32");
  });

  it("blocks two queries per invocation only when there are enough of them", async () => {
    const device = fakeDevice(false);
    const threshold = REGISTER_QUERY_BLOCK_THRESHOLD;
    // Column attention runs hundreds of sequences; row attention runs one chain length.
    expect((await attentionFlashKernelForShape(device, 32, 512)).variant).toBe("register-2q");
    expect((await attentionFlashKernelForShape(device, 32, threshold)).variant).toBe("register-2q");
    expect((await attentionFlashKernelForShape(device, 32, threshold - 1)).variant).toBe("register");
    expect((await attentionFlashKernelForShape(device, 32, 59)).variant).toBe("register");
  });

  it("covers every query with the tile it reports", async () => {
    for (const queries of [59, 128, 256, 508, 512]) {
      const kernel = await attentionFlashKernelForShape(fakeDevice(false), 32, queries);
      const covered = Math.ceil(queries / kernel.queryTile) * kernel.queryTile;
      expect(covered).toBeGreaterThanOrEqual(queries);
    }
  });

  it("leaves a subgroup verdict alone", async () => {
    const device = fakeDevice(true);
    presetAttentionFlashKernel(device, 32, "subgroup-key32");
    expect((await attentionFlashKernelForShape(device, 32, 512)).variant).toBe("subgroup-key32");
  });
});
