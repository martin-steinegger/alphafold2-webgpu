import { describe, expect, it } from "vitest";
import { ComputePipelineCache } from "../src/runtime/pipeline-cache.js";

/** A device whose first pipeline creation fails the way WebKit reports it. */
function flakyDevice(failures: number) {
  let attempts = 0;
  const device = {
    createShaderModule: () => ({
      getCompilationInfo: async () => ({ messages: [] }),
    }),
    createComputePipelineAsync: async () => {
      attempts += 1;
      if (attempts <= failures) throw new Error("Compute library failed creation");
      return { label: "ok" };
    },
  };
  return { device: device as unknown as GPUDevice, attempts: () => attempts };
}

describe("a pipeline that fails to build", () => {
  it("says which pipeline it was", async () => {
    const { device } = flakyDevice(1);
    const cache = new ComputePipelineCache(device);
    await expect(cache.get("block:triangle:incoming:59:project-gate", "fn main() {}"))
      .rejects.toThrow("Pipeline block:triangle:incoming:59:project-gate failed: "
        + "Compute library failed creation");
  });

  it("is not handed to the next caller, so a held device can retry it", async () => {
    // The page keeps its device across predictions. Caching the rejection
    // failed every later prediction with the first one's error.
    const { device, attempts } = flakyDevice(1);
    const cache = new ComputePipelineCache(device);
    await expect(cache.get("key", "fn main() {}")).rejects.toThrow();
    await expect(cache.get("key", "fn main() {}")).resolves.toEqual({ label: "ok" });
    expect(attempts()).toBe(2);
    expect(cache.size).toBe(1);
  });
});
