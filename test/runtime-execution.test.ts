import { afterEach, describe, expect, it, vi } from "vitest";
import { WebGpuExecution } from "../src/runtime/execution.js";

describe("WebGPU logical tensor bindings", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("binds the logical range when a bounded pool reuses a larger buffer", () => {
    const physicalBuffer = { destroy: vi.fn() } as unknown as GPUBuffer;
    const pass = {
      setPipeline: vi.fn(), setBindGroup: vi.fn(), dispatchWorkgroups: vi.fn(),
      pushDebugGroup: vi.fn(), popDebugGroup: vi.fn(), end: vi.fn(),
    };
    const encoder = { beginComputePass: vi.fn(() => pass) } as unknown as GPUCommandEncoder;
    const createBindGroup = vi.fn((descriptor) => descriptor);
    const device = {
      limits: { maxStorageBufferBindingSize: 1024, minStorageBufferOffsetAlignment: 256 },
      createBuffer: vi.fn(() => physicalBuffer), createBindGroup,
    } as unknown as GPUDevice;
    const execution = new WebGpuExecution(device, { maxPooledBytes: 1024 });
    const large = execution.allocate("large", 8, 1);
    large.allocation.release();
    const logical = execution.allocate("logical", 4, 1);
    const pipeline = { getBindGroupLayout: vi.fn(() => ({})) } as unknown as GPUComputePipeline;

    execution.dispatch(encoder, pipeline, [logical], 1);

    expect(logical.allocation.buffer).toBe(physicalBuffer);
    expect(createBindGroup).toHaveBeenCalledWith(expect.objectContaining({
      entries: [{ binding: 0, resource: { buffer: physicalBuffer, offset: 0, size: 16 } }],
    }));
    execution.release();
  });

  it("maps only the logical tensor range from a larger pooled readback buffer", async () => {
    vi.stubGlobal("GPUMapMode", { READ: 1 });
    const backing = new Float32Array([1, 2, 3, 4, 99, 99, 99, 99]);
    const getMappedRange = vi.fn((offset: number, size: number) =>
      backing.buffer.slice(offset, offset + size));
    const physicalBuffer = {
      destroy: vi.fn(), mapAsync: vi.fn(async () => undefined), getMappedRange, unmap: vi.fn(),
    } as unknown as GPUBuffer;
    const device = {
      limits: { maxStorageBufferBindingSize: 1024 },
      createBuffer: vi.fn(() => physicalBuffer),
    } as unknown as GPUDevice;
    const execution = new WebGpuExecution(device, { maxPooledBytes: 1024 });
    const large = execution.allocate("large-readback", 8, 1);
    large.allocation.release();
    const logical = execution.allocate("logical-readback", 4, 1);

    expect(Array.from(await execution.mapFloat32(logical))).toEqual([1, 2, 3, 4]);
    expect(getMappedRange).toHaveBeenCalledWith(0, 16);
    execution.release();
  });
});
