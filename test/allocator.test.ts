import { describe, expect, it, vi } from "vitest";
import { GpuBufferAllocator } from "../src/runtime/allocator.js";

describe("GPU buffer pooling", () => {
  it("evicts the oldest released buffer when the resident pool cap is exceeded", () => {
    const buffers: { destroy: ReturnType<typeof vi.fn> }[] = [];
    const device = { createBuffer: vi.fn(() => {
      const buffer = { destroy: vi.fn() }; buffers.push(buffer); return buffer;
    }) } as unknown as GPUDevice;
    const allocator = new GpuBufferAllocator(device, true, 12);
    const first = allocator.allocate("first", 8, 1);
    const second = allocator.allocate("second", 8, 1);
    first.release(); second.release();
    expect(allocator.snapshot()).toMatchObject({
      currentBytes: 0, residentBytes: 8, pooledBytes: 8, peakResidentBytes: 16,
    });
    expect(buffers[0]!.destroy).toHaveBeenCalledOnce();
    expect(buffers[1]!.destroy).not.toHaveBeenCalled();
    const reused = allocator.allocate("reused", 8, 1);
    expect(device.createBuffer).toHaveBeenCalledTimes(2);
    expect(reused.buffer).toBe(buffers[1]);
    expect(allocator.snapshot().pooledBytes).toBe(0);
    reused.release(); allocator.destroyPooled();
    expect(allocator.snapshot()).toMatchObject({ currentBytes: 0, residentBytes: 0, pooledBytes: 0 });
  });

  it("does not retain a single buffer larger than the pool cap", () => {
    const buffer = { destroy: vi.fn() };
    const device = { createBuffer: vi.fn(() => buffer) } as unknown as GPUDevice;
    const allocator = new GpuBufferAllocator(device, true, 4);
    allocator.allocate("large", 8, 1).release();
    expect(buffer.destroy).toHaveBeenCalledOnce();
    expect(allocator.snapshot()).toMatchObject({ residentBytes: 0, pooledBytes: 0 });
  });

  it("uses the smallest compatible bounded allocation for a smaller logical tensor", () => {
    const buffers: { destroy: ReturnType<typeof vi.fn> }[] = [];
    const device = { createBuffer: vi.fn(() => {
      const buffer = { destroy: vi.fn() }; buffers.push(buffer); return buffer;
    }) } as unknown as GPUDevice;
    const allocator = new GpuBufferAllocator(device, true, 128);
    const large = allocator.allocate("large", 32, 1);
    const best = allocator.allocate("best", 24, 1);
    large.release(); best.release();

    const logical = allocator.allocate("logical", 16, 1);
    expect(logical.buffer).toBe(buffers[1]);
    expect(logical.byteLength).toBe(16);
    expect(device.createBuffer).toHaveBeenCalledTimes(2);
    expect(allocator.snapshot()).toMatchObject({
      currentBytes: 16, residentBytes: 56, pooledBytes: 32, bufferCount: 2,
    });
    logical.release();
    expect(allocator.snapshot()).toMatchObject({ currentBytes: 0, residentBytes: 56, pooledBytes: 56 });
    allocator.destroyPooled();
  });

  it("does not reuse an incompatible usage or change unbounded exact matching", () => {
    const buffers: { destroy: ReturnType<typeof vi.fn> }[] = [];
    const device = { createBuffer: vi.fn(() => {
      const buffer = { destroy: vi.fn() }; buffers.push(buffer); return buffer;
    }) } as unknown as GPUDevice;
    const bounded = new GpuBufferAllocator(device, true, 128);
    bounded.allocate("storage", 32, 1).release();
    expect(bounded.allocate("uniform", 16, 2).buffer).toBe(buffers[1]);

    const unbounded = new GpuBufferAllocator(device, true);
    unbounded.allocate("large", 32, 1).release();
    expect(unbounded.allocate("small", 16, 1).buffer).toBe(buffers[3]);
    expect(device.createBuffer).toHaveBeenCalledTimes(4);
    bounded.destroyPooled(); unbounded.destroyPooled();
  });
});
