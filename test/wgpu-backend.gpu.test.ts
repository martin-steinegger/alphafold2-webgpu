import { beforeAll, describe, expect, it } from "vitest";
import { requestWgpuAdapter } from "../src/runtime/wgpu/adapter.js";
import { wgpuAddonExists, wgpuNative } from "../src/runtime/wgpu/backend.js";
import { createTiledGemmShader } from "../src/runtime/gemm.js";
import { calibrateDialect, WGPU_MATRIX } from "../src/runtime/dialect.js";

// The addon is built out of tree, so a checkout without it skips rather than
// fails; npm run check must stay green on a machine that has never built it.
const enabled = process.env.AFWEBGPU_GPU_TESTS === "1" && wgpuAddonExists();

const GEMM = createTiledGemmShader({
  preamble: `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
struct P { rows: u32, inner: u32, columns: u32, pad: u32 };
@group(0) @binding(3) var<uniform> p: P;`,
  rows: "p.rows", inner: "p.inner", columns: "p.columns",
  sourceElement: "a[row * p.inner + k]", weightElement: "b[k * p.columns + column]",
  store: "o[row * p.columns + column] = element;",
  sourceArray: { array: "a", stride: "p.inner" },
  weightArray: { array: "b", stride: "p.columns" },
}, { precision: "f32", inner: 8 });

describe.skipIf(!enabled)("the wgpu backend", () => {
  const size = 128;
  let adapter: GPUAdapter;
  let device: GPUDevice;

  beforeAll(async () => {
    adapter = requestWgpuAdapter();
    device = await adapter.requestDevice({
      requiredFeatures: [...adapter.features] as GPUFeatureName[],
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      },
    });
  });

  it("reports an adapter the model can plan against", () => {
    expect(adapter.limits.maxBufferSize).toBeGreaterThanOrEqual(256 * 1024 ** 2);
    expect(adapter.limits.maxStorageBuffersPerShaderStage).toBeGreaterThanOrEqual(8);
    const info = adapter.info as unknown as { subgroupMinSize: number; subgroupMaxSize: number };
    expect(info.subgroupMinSize).toBeGreaterThan(0);
    expect(info.subgroupMinSize).toBeLessThanOrEqual(info.subgroupMaxSize);
  });

  it("grants the raised limits rather than the defaults", () => {
    // wgpu's own defaults are 16 KiB of workgroup storage and 256 invocations,
    // which is what a device opened before requestDevice would have been stuck
    // with. The matrix attention kernel needs more than the first.
    expect(device.limits.maxComputeWorkgroupStorageSize)
      .toBe(adapter.limits.maxComputeWorkgroupStorageSize);
  });

  it("settles on wgpu's matrix spelling, not Dawn's", async () => {
    const dialect = await calibrateDialect(device);
    expect(dialect.subgroupEnable).toBe("");
    if (adapter.features.has("wgpu-cooperative-matrix" as GPUFeatureName)) {
      expect(dialect.matrix).toBe(WGPU_MATRIX);
    }
  });

  it("runs a generated GEMM and reads the result back", async () => {
    const left = Float32Array.from({ length: size * size }, (_, i) => ((i % 17) - 8) / 8);
    const right = Float32Array.from({ length: size * size }, (_, i) => ((i % 13) - 6) / 6);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const upload = (data: ArrayBufferView, usage = storage): GPUBuffer => {
      const buffer = device.createBuffer({ size: data.byteLength, usage });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    const output = device.createBuffer({ size: size * size * 4, usage: storage });
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: GEMM }), entryPoint: "main" },
    });
    // The output is bound as a window, which is how the model binds a tensor
    // past the adapter's binding limit.
    const group = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: upload(left) } },
        { binding: 1, resource: { buffer: upload(right) } },
        { binding: 2, resource: { buffer: output, offset: 0, size: size * size * 4 } },
        { binding: 3, resource: { buffer: upload(new Uint32Array([size, size, size, 0]),
          GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST) } },
      ],
    });
    const readback = device.createBuffer(
      { size: size * size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.pushDebugGroup("gemm");
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(size / 128), Math.ceil(size / 64), 1);
    pass.popDebugGroup();
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, size * size * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    await readback.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    let expected = 0;
    for (let k = 0; k < size; k += 1) expected += left[k]! * right[k * size]!;
    expect(out[0]).toBeCloseTo(expected, 3);
    // A whole row, so a kernel that writes only its first element cannot pass.
    let last = 0;
    for (let k = 0; k < size; k += 1) last += left[k]! * right[k * size + size - 1]!;
    expect(out[size - 1]).toBeCloseTo(last, 3);
  });

  it("writes timestamps a pass can be profiled from", async () => {
    const timing = device.createQuerySet({ type: "timestamp", count: 2 });
    const resolved = device.createBuffer(
      { size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer(
      { size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.beginComputePass({ timestampWrites: {
      querySet: timing, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }).end();
    encoder.resolveQuerySet(timing, 0, 2, resolved, 0);
    encoder.copyBufferToBuffer(resolved, 0, readback, 0, 16);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const [begin, end] = new BigUint64Array(readback.getMappedRange().slice(0));
    readback.unmap();
    expect(begin).toBeGreaterThan(0n);
    expect(end).toBeGreaterThanOrEqual(begin!);
  });

  it("catches a rejected module in an error scope instead of throwing", async () => {
    device.pushErrorScope("validation");
    device.createShaderModule({ code: "@compute @workgroup_size(1) fn main() { this is not wgsl }" });
    const error = await device.popErrorScope();
    expect(error?.message ?? "").toMatch(/./);
  });

  it("frees the handles a caller destroys", () => {
    const native = wgpuNative();
    const before = native.handleCounts()[0]!;
    const buffer = device.createBuffer({ size: 256, usage: GPUBufferUsage.STORAGE });
    expect(native.handleCounts()[0]).toBe(before + 1);
    buffer.destroy();
    expect(native.handleCounts()[0]).toBe(before);
  });

  it("keeps a bind group alive until its commands are replayed", async () => {
    // The model writes setBindGroup(0, device.createBindGroup(...)), so the
    // group is garbage the moment the call returns, while the recorded command
    // holds only a handle and is replayed at finish. The encoder has to hold
    // it, or a collection between the two calls takes the native object.
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto",
      compute: { module: device.createShaderModule({ code: GEMM }), entryPoint: "main" },
    });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const make = (bytes: number, usage = storage): GPUBuffer =>
      device.createBuffer({ size: bytes, usage });
    const buffers = [make(1024), make(1024), make(1024),
      make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)];
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    pass.dispatchWorkgroups(1, 1, 1);
    pass.end();
    expect(() => device.queue.submit([encoder.finish()])).not.toThrow();
    await device.queue.onSubmittedWorkDone();
  });

  it("takes a buffer written through a creation mapping", async () => {
    const values = Uint32Array.from([7, 11, 13, 17]);
    const upload = device.createBuffer({ size: values.byteLength, mappedAtCreation: true,
      usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC });
    new Uint32Array(upload.getMappedRange()).set(values);
    upload.unmap();
    const readback = device.createBuffer({ size: values.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(upload, 0, readback, 0, values.byteLength);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const seen = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    expect([...seen]).toEqual([...values]);
  });

  it("refuses a buffer usage bit wgpu has no counterpart for", () => {
    // INDIRECT, which nothing here emits: the point is that an unknown bit is
    // refused rather than dropped by a truncating cast.
    expect(() => device.createBuffer({ size: 256, usage: 256 }))
      .toThrow(/unsupported buffer usage/);
  });
});
