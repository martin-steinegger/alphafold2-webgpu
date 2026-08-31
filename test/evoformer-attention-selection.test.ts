import { describe, expect, it } from "vitest";
import {
  selectAttentionFlashKernel,
  supportsAttentionSubgroups,
} from "../src/evoformer/attention.js";
import { recordSubgroupRange } from "../src/runtime/subgroups.js";

function deviceWithSubgroupRange(minSubgroupSize: number, maxSubgroupSize: number): GPUDevice {
  const device = {
    features: new Set<GPUFeatureName>([
      "subgroups" as GPUFeatureName,
      "subgroup-size-control" as GPUFeatureName,
    ]),
    limits: {
      minSubgroupSize,
      maxSubgroupSize,
      maxComputeInvocationsPerWorkgroup: 256,
      maxComputeWorkgroupStorageSize: 16_384,
    },
  } as unknown as GPUDevice;
  recordSubgroupRange(device, {
    info: { subgroupMinSize: minSubgroupSize, subgroupMaxSize: maxSubgroupSize },
  } as unknown as GPUAdapter);
  return device;
}

describe("attention subgroup kernel selection", () => {
  it("uses the 32-lane fast path when the device supports that subgroup width", () => {
    const device = deviceWithSubgroupRange(4, 32);
    expect(supportsAttentionSubgroups(device)).toBe(true);
    expect(selectAttentionFlashKernel(device).variant).toBe("subgroup-key32");
  });

  it("falls back to portable register attention for fixed four-lane subgroups", () => {
    const device = deviceWithSubgroupRange(4, 4);
    expect(supportsAttentionSubgroups(device)).toBe(false);
    expect(selectAttentionFlashKernel(device).variant).toBe("register");
    expect(() => selectAttentionFlashKernel(device, 32, "subgroup-key32"))
      .toThrow(/unsupported by this device/);
  });

  it("does not assume subgroup width support when the experimental limits are absent", () => {
    const device = {
      features: new Set<GPUFeatureName>([
        "subgroups" as GPUFeatureName,
        "subgroup-size-control" as GPUFeatureName,
      ]),
      limits: {},
    } as unknown as GPUDevice;
    expect(supportsAttentionSubgroups(device)).toBe(false);
  });
});
