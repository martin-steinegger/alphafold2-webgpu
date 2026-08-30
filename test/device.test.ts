import { describe, expect, it, vi } from "vitest";
import {
  estimateMonomerMemory, monomerDeviceRequirements, planMonomerDevice, requestAlphaFoldDevice,
  suggestMonomerRows,
} from "../src/runtime/device.js";

function adapterWithLimits(maxStorageBufferBindingSize: number, maxBufferSize: number): {
  readonly adapter: GPUAdapter;
  readonly requestDevice: ReturnType<typeof vi.fn>;
} {
  const requestDevice = vi.fn(async () => ({}) as GPUDevice);
  const adapter = {
    features: new Set<GPUFeatureName>(["subgroups" as GPUFeatureName, "timestamp-query"]),
    limits: { maxStorageBufferBindingSize, maxBufferSize },
    requestDevice,
  } as unknown as GPUAdapter;
  return { adapter, requestDevice };
}

describe("requestAlphaFoldDevice", () => {
  it("requests only portable baseline limits when no prediction shape is supplied", async () => {
    const { adapter, requestDevice } = adapterWithLimits(2 * 1024 ** 3, 4 * 1024 ** 3);
    await requestAlphaFoldDevice(adapter);
    expect(requestDevice).toHaveBeenCalledWith({
      requiredFeatures: ["subgroups", "timestamp-query"],
      requiredLimits: {
        maxBufferSize: 256 * 1024 ** 2,
        maxStorageBufferBindingSize: 128 * 1024 ** 2,
      },
    });
  });

  it("sizes limits from persistent tensors rather than transition intermediates", () => {
    expect(monomerDeviceRequirements(291, 508, 1024)).toEqual({
      maxBufferSize: 256 * 1024 ** 2,
      maxStorageBufferBindingSize: 32 * 291 * 32 * 128 * 4,
    });
    expect(monomerDeviceRequirements(1_000, 508, 1024)).toEqual({
      maxBufferSize: 32 * 1_000 * 32 * 128 * 4,
      maxStorageBufferBindingSize: 32 * 1_000 * 32 * 128 * 4,
    });
  });

  it("uses full transitions only when their shape fits the adapter", () => {
    const capable = adapterWithLimits(2 * 1024 ** 3, 2 * 1024 ** 3).adapter;
    expect(planMonomerDevice(capable, 291, 508, 1024)).toMatchObject({
      transitionMode: "full",
      requirements: {
        maxBufferSize: 508 * 291 * 1024 * 4,
        maxStorageBufferBindingSize: 508 * 291 * 1024 * 4,
      },
    });
    const constrained = adapterWithLimits(256 * 1024 ** 2, 256 * 1024 ** 2).adapter;
    expect(planMonomerDevice(constrained, 291, 508, 1024)).toMatchObject({
      transitionMode: "chunked",
      requirements: monomerDeviceRequirements(291, 508, 1024),
    });
    expect(planMonomerDevice(capable, 291, 508, 1024, undefined, true)).toMatchObject({
      transitionMode: "chunked",
      requirements: monomerDeviceRequirements(291, 508, 1024),
    });
  });

  it("estimates aggregate memory and suggests explicit MSA row limits", () => {
    const full = estimateMonomerMemory(291, 508, 1024, "full");
    const compact = estimateMonomerMemory(291, 508, 1024, "chunked");
    expect(full.estimatedPeakBytes).toBeGreaterThan(full.persistentBytes);
    expect(compact.estimatedPeakBytes).toBeLessThanOrEqual(full.estimatedPeakBytes);
    const budget = estimateMonomerMemory(291, 128, 256, "chunked").estimatedPeakBytes;
    const suggestion = suggestMonomerRows(291, 508, 1024, "chunked", budget);
    expect(suggestion).toBeDefined();
    expect(suggestion!.estimatedPeakBytes).toBeLessThanOrEqual(budget);
    expect(suggestion!.msaSequences).toBeLessThanOrEqual(508);
    expect(suggestion!.extraSequences).toBeLessThanOrEqual(1024);
  });

  it("rounds an exact shape requirement to a stable capability tier", async () => {
    const { adapter, requestDevice } = adapterWithLimits(2 * 1024 ** 3, 2 * 1024 ** 3);
    await requestAlphaFoldDevice(adapter, monomerDeviceRequirements(291, 508, 1024));
    expect(requestDevice).toHaveBeenCalledWith(expect.objectContaining({
      requiredLimits: {
        maxBufferSize: 256 * 1024 ** 2,
        maxStorageBufferBindingSize: 256 * 1024 ** 2,
      },
    }));
  });

  it("rejects a shape that exceeds the adapter instead of silently lowering its requirements", async () => {
    const { adapter, requestDevice } = adapterWithLimits(128 * 1024 ** 2, 256 * 1024 ** 2);
    const requirements = monomerDeviceRequirements(291, 508, 1024);
    await expect(requestAlphaFoldDevice(adapter, requirements)).rejects.toThrow(/requires a 146 MiB storage binding/);
    expect(requestDevice).not.toHaveBeenCalled();
  });
});
