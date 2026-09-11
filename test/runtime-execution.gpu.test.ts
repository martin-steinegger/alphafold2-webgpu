import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { WebGpuExecution } from "../src/runtime/execution.js";
import { testGpu } from "./support/gpu-instance.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

describe.skipIf(!enabled)("WebGPU bounded buffer reuse", () => {
  let gpu: GPU;
  let device: GPUDevice;

  beforeAll(async () => {
    gpu = testGpu();
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
      // Twice the upload's size: the pool reuses larger buffers up to that ratio.
      const workspace = execution.allocate("oversized-workspace", 8, usage);
      workspace.allocation.release();
      // Uploads only take over buffers retired before a submitted boundary,
      // because the queue write would land ahead of commands still encoding.
      execution.noteSubmitted();
      const base = execution.upload("base", new Float32Array([1, 2, 3, 4]), dataUsage);
      const update = execution.upload("update", new Float32Array([0.5, -1, 3, -1]));
      expect(base.allocation.buffer).toBe(workspace.allocation.buffer);
      expect(base.allocation.byteLength).toBe(16);

      const encoder = device.createCommandEncoder();
      device.pushErrorScope("validation");
      await execution.addInPlace(encoder, base, update, "bounded-reuse.add");
      const readback = execution.createReadback("bounded-reuse.readback", base, encoder);
      device.queue.submit([encoder.finish()]);
      execution.noteSubmitted();
      const validationError = await device.popErrorScope();
      if (validationError !== null) throw new Error(validationError.message);

      expect(Array.from(await execution.mapFloat32(readback))).toEqual([1.5, 1, 6, 3]);
      expect(execution.snapshot().bufferCount).toBe(3);
    } finally {
      execution.release();
    }
  });

  it("makes a bind group once for what it binds, and again for a different range", async () => {
    // A stack's blocks bind the same pooled buffers over and over; a new bind
    // group for each dispatch was 70% of them repeated, and each one costs the
    // host time Dawn and a browser charge for.
    const execution = new WebGpuExecution(device);
    const createBindGroup = device.createBindGroup.bind(device);
    let made = 0;
    device.createBindGroup = (descriptor) => { made += 1; return createBindGroup(descriptor); };
    try {
      const base = execution.upload("base", new Float32Array([1, 2, 3, 4]),
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      const update = execution.upload("update", new Float32Array([1, 1, 1, 1]));
      const half = { ...update, elements: 2 };
      const halfBase = { ...base, elements: 2 };
      const encoder = device.createCommandEncoder();
      device.pushErrorScope("validation");
      await execution.addInPlace(encoder, base, update, "cache.add");
      await execution.addInPlace(encoder, base, update, "cache.add");
      await execution.addInPlace(encoder, halfBase, half, "cache.add-half");
      execution.endComputePass(encoder);
      const readback = execution.createReadback("cache.readback", base, encoder);
      device.queue.submit([encoder.finish()]);
      const validationError = await device.popErrorScope();
      if (validationError !== null) throw new Error(validationError.message);

      expect(made).toBe(2);
      expect(Array.from(await execution.mapFloat32(readback))).toEqual([4, 5, 5, 6]);
    } finally {
      device.createBindGroup = createBindGroup;
      execution.release();
    }
  });
});
