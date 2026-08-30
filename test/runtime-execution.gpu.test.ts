import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { WebGpuExecution } from "../src/runtime/execution.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

describe.skipIf(!enabled)("WebGPU bounded buffer reuse", () => {
  let gpu: GPU;
  let device: GPUDevice;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    device = await adapter.requestDevice();
  });

  afterAll(() => device?.destroy());

  it("matches an independent sum after binding a smaller logical range", async () => {
    const execution = new WebGpuExecution(device, { maxPooledBytes: 1024 });
    try {
      const dataUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
      const usage = dataUsage | GPUBufferUsage.COPY_DST;
      const workspace = execution.allocate("oversized-workspace", 16, usage);
      workspace.allocation.release();
      const base = execution.upload("base", new Float32Array([1, 2, 3, 4]), dataUsage);
      const update = execution.upload("update", new Float32Array([0.5, -1, 3, -1]));
      expect(base.allocation.buffer).toBe(workspace.allocation.buffer);
      expect(base.allocation.byteLength).toBe(16);

      const encoder = device.createCommandEncoder();
      device.pushErrorScope("validation");
      await execution.addInPlace(encoder, base, update, "bounded-reuse.add");
      const readback = execution.createReadback("bounded-reuse.readback", base, encoder);
      device.queue.submit([encoder.finish()]);
      const validationError = await device.popErrorScope();
      if (validationError !== null) throw new Error(validationError.message);

      expect(Array.from(await execution.mapFloat32(readback))).toEqual([1.5, 1, 6, 3]);
      expect(execution.snapshot().bufferCount).toBe(3);
    } finally {
      execution.release();
    }
  });
});
