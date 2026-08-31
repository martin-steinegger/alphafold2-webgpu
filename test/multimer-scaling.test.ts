import { describe, expect, it } from "vitest";
import {
  estimateMonomerMemory, monomerDeviceRequirements, planMonomerDevice, suggestMonomerRows,
} from "../src/runtime/device.js";

function constrainedAdapter(): GPUAdapter {
  return {
    limits: {
      maxBufferSize: 256 * 1024 ** 2,
      maxStorageBufferBindingSize: 256 * 1024 ** 2,
    },
  } as unknown as GPUAdapter;
}

describe("Multimer shared scaling policy", () => {
  it("plans the concatenated homotrimer and complex MSA through bounded transitions", () => {
    const totalLength = 3 * 59;
    const msaSequences = 508;
    const extraSequences = 1024;
    const plan = planMonomerDevice(
      constrainedAdapter(), totalLength, msaSequences, extraSequences, undefined, true,
    );
    expect(plan.transitionMode).toBe("chunked");
    expect(plan.requirements).toEqual(
      monomerDeviceRequirements(totalLength, msaSequences, extraSequences),
    );
    expect(plan.memory).toEqual(
      estimateMonomerMemory(totalLength, msaSequences, extraSequences, "chunked"),
    );
  });

  it("reduces paired/unpaired row counts against the same explicit memory budget", () => {
    const totalLength = 2 * 59;
    const budget = estimateMonomerMemory(totalLength, 96, 192, "chunked").estimatedPeakBytes;
    const suggestion = suggestMonomerRows(totalLength, 508, 1024, "chunked", budget);
    expect(suggestion).toBeDefined();
    expect(suggestion!.estimatedPeakBytes).toBeLessThanOrEqual(budget);
    expect(suggestion!.msaSequences).toBeLessThanOrEqual(508);
    expect(suggestion!.extraSequences).toBeLessThanOrEqual(1024);
  });
});
