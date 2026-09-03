import { EXACT_STORAGE } from "../src/model/monomer.js";
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
    limits: { maxStorageBufferBindingSize, maxBufferSize, maxStorageBuffersPerShaderStage: 16 },
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
        maxStorageBuffersPerShaderStage: 16,
      },
    });
  });

  it("sizes limits from persistent tensors rather than transition intermediates", () => {
    // Activations are packed, so every persistent tensor asks for two bytes an
    // element. 291 residues over 508 clustered rows: the clustered MSA is the
    // largest of them, and every scratch tensor is bounded below it.
    expect(monomerDeviceRequirements(291, 508, 1024)).toEqual({
      maxBufferSize: 256 * 1024 ** 2,
      maxStorageBufferBindingSize: 128 * 1024 ** 2,
    });
    // At 1000 residues the clustered MSA is still the larger of the two, and
    // the requirement follows it rather than any scratch tensor.
    expect(monomerDeviceRequirements(1_000, 508, 1024)).toEqual({
      maxBufferSize: 256 * 1024 ** 2,
      maxStorageBufferBindingSize: 508 * 1_000 * 256 * 2,
    });
    // With a shallow alignment the pair tensor is what the device must hold.
    expect(monomerDeviceRequirements(1_000, 32, 64).maxStorageBufferBindingSize)
      .toBe(1_000 * 1_000 * 128 * 2);
    // The exact storages ask for twice as much, which is why they are not the
    // shipped path.
    expect(monomerDeviceRequirements(1_000, 32, 64, EXACT_STORAGE).maxStorageBufferBindingSize)
      .toBe(1_000 * 1_000 * 128 * 4);
  });

  it("does not size limits for scratch that every operation now bounds", () => {
    // The outer-product contraction, the transition window and the attention
    // window all cap themselves, so none of them may drive the requirement.
    const requirement = monomerDeviceRequirements(512, 256, 512).maxStorageBufferBindingSize;
    // Both persistent tensors are 64 MiB packed, so the base tier covers them.
    expect(Math.max(256 * 512 * 256 * 2, 512 * 512 * 128 * 2)).toBe(64 * 1024 ** 2);
    expect(requirement).toBe(128 * 1024 ** 2);
    // Scratch that grew with the shape would have asked for more than the tier.
    expect(requirement).toBeLessThanOrEqual(128 * 1024 ** 2);
  });

  it("asks for twice the binding on the exact path the tests use", () => {
    expect(monomerDeviceRequirements(291, 508, 1024, EXACT_STORAGE)).toEqual({
      maxBufferSize: 256 * 1024 ** 2,
      maxStorageBufferBindingSize: 508 * 291 * 256 * 4,
    });
    const constrained = adapterWithLimits(128 * 1024 ** 2, 256 * 1024 ** 2).adapter;
    expect(planMonomerDevice(constrained, 291, 508, 1024, undefined, true)).toMatchObject({
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
        maxStorageBufferBindingSize: 128 * 1024 ** 2,
        maxStorageBuffersPerShaderStage: 16,
      },
    }));
  });

  it("takes the adapter's binding limit when the shape wants more, and windows the rest", async () => {
    // A tensor past the binding limit is bound as several windows of the same
    // buffer, so a small limit costs dispatch slots rather than the run.
    const { adapter, requestDevice } = adapterWithLimits(64 * 1024 ** 2, 256 * 1024 ** 2);
    await requestAlphaFoldDevice(adapter, monomerDeviceRequirements(291, 508, 1024));
    expect(requestDevice).toHaveBeenCalledWith(expect.objectContaining({
      requiredLimits: {
        maxBufferSize: 256 * 1024 ** 2,
        maxStorageBufferBindingSize: 64 * 1024 ** 2,
        maxStorageBuffersPerShaderStage: 16,
      },
    }));
  });

  it("rejects a shape whose largest tensor exceeds the adapter's buffer size", async () => {
    const { adapter, requestDevice } = adapterWithLimits(64 * 1024 ** 2, 64 * 1024 ** 2);
    await expect(requestAlphaFoldDevice(adapter, monomerDeviceRequirements(291, 508, 1024)))
      .rejects.toThrow(/requires a 256 MiB buffer/);
    expect(requestDevice).not.toHaveBeenCalled();
  });
});

describe("estimateMonomerMemory", () => {
  it("charges Multimer-v3 for its template rows and template module", () => {
    const monomer = estimateMonomerMemory(590, 508, 2048, "full");
    const multimer = estimateMonomerMemory(590, 508, 2048, "full", { multimer: true, templateRows: 4 });
    expect(multimer.estimatedPeakBytes).toBeGreaterThan(monomer.estimatedPeakBytes);
    // The template module holds several pair-sized tensors beside the pair and
    // the extra alignment, so at this size it sets the peak: about three pair
    // representations (170 MiB each at 590 residues) above the trunk terms.
    const pair = 590 * 590 * 128 * 2;
    expect(multimer.estimatedPeakBytes - monomer.estimatedPeakBytes).toBeGreaterThan(pair);
    // A ten-copy 59-mer at the page's multimer defaults is over a gigabyte live.
    expect(multimer.estimatedPeakBytes).toBeGreaterThan(1000 * 1024 ** 2);
  });

  // Combined resident peaks measured on GB10 with the model's packed storage.
  // The estimate gates whether a browser prediction is allowed to start, so it
  // must cover physical pooled GPUBuffer residency rather than only the
  // allocator's logically live bytes.
  const measured: ReadonlyArray<readonly [number, number, number, number]> = [
    [59, 508, 1024, 102], [128, 256, 512, 97], [256, 256, 512, 130],
    [384, 256, 512, 231], [512, 256, 512, 312],
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

  it("estimates the exact storages above the packed model", () => {
    const mib = 1024 ** 2;
    const packed = estimateMonomerMemory(384, 256, 512, "chunked").estimatedPeakBytes / mib;
    const exact = estimateMonomerMemory(384, 256, 512, "chunked", EXACT_STORAGE).estimatedPeakBytes / mib;
    expect(packed).toBeLessThan(exact);
    // Combined resident peaks measured at 384 residues for the model's packed
    // storage and for the exact path the differential tests use. Every
    // admission estimate must stay above its physical GPUBuffer peak.
    for (const [label, estimate, resident] of [
      ["packed", packed, 231], ["exact", exact, 351],
    ] as const) {
      expect(estimate, label).toBeGreaterThan(resident);
      expect(estimate / resident, label).toBeLessThan(1.4);
    }
  });
});
