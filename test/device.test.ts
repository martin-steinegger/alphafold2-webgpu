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
    // 291 residues over 508 clustered rows: the clustered MSA is the largest
    // persistent tensor, and every scratch tensor is bounded below it.
    expect(monomerDeviceRequirements(291, 508, 1024)).toEqual({
      maxBufferSize: 256 * 1024 ** 2,
      maxStorageBufferBindingSize: 508 * 291 * 256 * 4,
    });
    // At 1000 residues the clustered MSA is still the larger of the two, and
    // the requirement follows it rather than any scratch tensor.
    expect(monomerDeviceRequirements(1_000, 508, 1024)).toEqual({
      maxBufferSize: 508 * 1_000 * 256 * 4,
      maxStorageBufferBindingSize: 508 * 1_000 * 256 * 4,
    });
    // With a shallow alignment the pair tensor is what the device must hold.
    expect(monomerDeviceRequirements(1_000, 32, 64).maxStorageBufferBindingSize)
      .toBe(1_000 * 1_000 * 128 * 4);
  });

  it("does not size limits for scratch that every operation now bounds", () => {
    // The outer-product contraction, the transition window and the attention
    // window all cap themselves, so none of them may drive the requirement.
    const requirement = monomerDeviceRequirements(512, 256, 512).maxStorageBufferBindingSize;
    expect(requirement).toBe(Math.max(256 * 512 * 256 * 4, 512 * 512 * 128 * 4));
    // Scratch that grew with the shape would have asked for 256 MiB here.
    expect(requirement).toBeLessThan(256 * 1024 ** 2);
  });

  it("accounts for packed MSA storage in monomer limits", () => {
    expect(monomerDeviceRequirements(291, 508, 1024, { msaStorage: "f16" })).toEqual({
      maxBufferSize: 256 * 1024 ** 2,
      maxStorageBufferBindingSize: 128 * 1024 ** 2,
    });
    const constrained = adapterWithLimits(128 * 1024 ** 2, 256 * 1024 ** 2).adapter;
    expect(planMonomerDevice(
      constrained, 291, 508, 1024, undefined, true, { msaStorage: "f16", triangleWholeStorage: "f16" },
    )).toMatchObject({
      transitionMode: "chunked",
      requirements: { maxBufferSize: 256 * 1024 ** 2, maxStorageBufferBindingSize: 128 * 1024 ** 2 },
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
    await expect(requestAlphaFoldDevice(adapter, requirements)).rejects.toThrow(/requires a 144 MiB storage binding/);
    expect(requestDevice).not.toHaveBeenCalled();
  });
});

describe("estimateMonomerMemory", () => {
  // Combined resident peaks measured on GB10. The estimate gates whether a
  // browser prediction is allowed to start, so it must cover physical pooled
  // GPUBuffer residency rather than only the allocator's logically live bytes.
  const measured: ReadonlyArray<readonly [number, number, number, number]> = [
    [59, 508, 1024, 117], [128, 256, 512, 117], [256, 256, 512, 206],
    [384, 256, 512, 351], [512, 256, 512, 504], [1_000, 256, 512, 1_447],
  ];

  it("stays an upper bound on measured combined residency", () => {
    for (const [length, msa, extra, residentMib] of measured) {
      const estimate = estimateMonomerMemory(length, msa, extra, "full").estimatedPeakBytes / 1024 ** 2;
      expect(estimate, `${length} residues`).toBeGreaterThan(residentMib);
    }
  });

  it("does not drift far above what actually runs", () => {
    for (const [length, msa, extra, residentMib] of measured) {
      const estimate = estimateMonomerMemory(length, msa, extra, "full").estimatedPeakBytes / 1024 ** 2;
      expect(estimate / residentMib, `${length} residues`).toBeLessThan(1.4);
    }
  });

  it("grows with both alignment depth and chain length", () => {
    const shallow = estimateMonomerMemory(256, 64, 128, "full").estimatedPeakBytes;
    const deep = estimateMonomerMemory(256, 512, 1024, "full").estimatedPeakBytes;
    const longer = estimateMonomerMemory(512, 64, 128, "full").estimatedPeakBytes;
    expect(deep).toBeGreaterThan(shallow);
    expect(longer).toBeGreaterThan(shallow);
  });

  it("reflects optional packed monomer storage", () => {
    const mib = 1024 ** 2;
    const exact = estimateMonomerMemory(384, 256, 512, "chunked").estimatedPeakBytes / mib;
    const triangle = estimateMonomerMemory(
      384, 256, 512, "chunked", { triangleWholeStorage: "f16" },
    ).estimatedPeakBytes / mib;
    const msa = estimateMonomerMemory(
      384, 256, 512, "chunked", { msaStorage: "f16" },
    ).estimatedPeakBytes / mib;
    const combined = estimateMonomerMemory(384, 256, 512, "chunked", {
      triangleWholeStorage: "f16", msaStorage: "f16",
    }).estimatedPeakBytes / mib;
    expect(triangle).toBeLessThan(exact);
    expect(msa).toBeLessThan(exact);
    expect(combined).toBeLessThan(triangle);
    expect(combined).toBeLessThan(msa);
    // L384 combined resident peaks measured for exact / triangle-f16 /
    // MSA-f16 / both. Every admission estimate must remain above its physical
    // GPUBuffer peak, including the pool floor shared by both packed options.
    for (const [label, estimate, resident] of [
      ["exact", exact, 351], ["triangle f16", triangle, 331],
      ["MSA f16", msa, 307], ["combined f16", combined, 307],
    ] as const) {
      expect(estimate, label).toBeGreaterThan(resident);
      expect(estimate / resident, label).toBeLessThan(1.35);
    }
  });
});
