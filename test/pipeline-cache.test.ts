import { describe, expect, it } from "vitest";
import { ComputePipelineCache, usedOverrides } from "../src/runtime/pipeline-cache.js";

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

describe("the overrides a pipeline is given", () => {
  const PREAMBLE = `override L: u32 = 1u;
override PAIRS: u32 = L * L;
override WHOLE_STRIDE: u32 = 1u;
override BLOCK_ROWS: u32 = 1u;
override BLOCK_PAIRS: u32 = BLOCK_ROWS * L;
`;

  it("are the ones the source uses, and what their initializers name", () => {
    // BLOCK_PAIRS is used, so BLOCK_ROWS and L are too, through it.
    expect([...usedOverrides(`${PREAMBLE}fn main() { let n = BLOCK_PAIRS; }`)].sort())
      .toEqual(["BLOCK_PAIRS", "BLOCK_ROWS", "L"]);
    expect([...usedOverrides(`${PREAMBLE}fn main() { let n = PAIRS; }`)].sort()).toEqual(["L", "PAIRS"]);
  });

  it("do not count a name in a comment", () => {
    const source = `${PREAMBLE}// the row stride is WHOLE_STRIDE\n/* L */ fn main() {}`;
    expect([...usedOverrides(source)]).toEqual([]);
  });

  it("leave out one the kernel does not use, which Safari 26 rejects", async () => {
    // WebKit 7624's createLibrary fails a pipeline, with no message, for a
    // constant its entry point does not use. Dawn ignores it.
    let given: Record<string, number> | undefined;
    const device = {
      createShaderModule: () => ({ getCompilationInfo: async () => ({ messages: [] }) }),
      createComputePipelineAsync: async (descriptor: GPUComputePipelineDescriptor) => {
        given = descriptor.compute.constants as Record<string, number> | undefined;
        return {};
      },
    } as unknown as GPUDevice;
    await new ComputePipelineCache(device).get("statistics",
      `${PREAMBLE}fn main() { let n = PAIRS; }`, "main",
      { L: 59, PAIRS: 3481, WHOLE_STRIDE: 3584, BLOCK_ROWS: 59, BLOCK_PAIRS: 3481 });
    expect(given).toEqual({ L: 59, PAIRS: 3481 });
  });
});
