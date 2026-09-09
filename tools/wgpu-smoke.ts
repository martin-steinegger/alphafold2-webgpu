/**
 * Runs one generated GEMM over the wgpu backend through the WebGPU shim, and
 * prints what the adapter reports.
 *
 * The path exercised is the one the model takes: the shim's adapter, the
 * shim's device, a pipeline with an automatic layout, a windowed binding, a
 * timestamped pass, a copy and a mapped readback.
 *
 * Usage: tsx tools/wgpu-smoke.ts [size]
 */
import { requestWgpuAdapter } from "../src/runtime/wgpu/adapter.js";
import { createTiledGemmShader } from "../src/runtime/gemm.js";
import { calibrateDialect } from "../src/runtime/dialect.js";

const size = Number(process.argv[2] ?? "256");
const adapter = requestWgpuAdapter();
console.log(JSON.stringify((adapter as unknown as { report: unknown }).report, null, 1));

const device = await adapter.requestDevice({
  requiredFeatures: [...adapter.features] as GPUFeatureName[],
  requiredLimits: {
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
    maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
  },
});
console.log("dialect:", JSON.stringify(await calibrateDialect(device)));

const code = createTiledGemmShader({
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

const left = Float32Array.from({ length: size * size }, (_, i) => ((i % 17) - 8) / 8);
const right = Float32Array.from({ length: size * size }, (_, i) => ((i % 13) - 6) / 6);

const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
const upload = (data: Float32Array): GPUBuffer => {
  const buffer = device.createBuffer({ size: data.byteLength, usage: storage });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};
const a = upload(left);
const b = upload(right);
const o = device.createBuffer({ size: size * size * 4, usage: storage });
const parameters = device.createBuffer(
  { size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(parameters, 0, new Uint32Array([size, size, size, 0]));

device.pushErrorScope("validation");
const pipeline = await device.createComputePipelineAsync({
  layout: "auto",
  compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
});
const compileError = await device.popErrorScope();
if (compileError !== null) throw new Error(compileError.message);

// A windowed binding on the output, which is how the model binds a tensor past
// the adapter's binding limit.
const group = device.createBindGroup({
  layout: pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: a } },
    { binding: 1, resource: { buffer: b } },
    { binding: 2, resource: { buffer: o, offset: 0, size: size * size * 4 } },
    { binding: 3, resource: { buffer: parameters } },
  ],
});

const timing = device.createQuerySet({ type: "timestamp", count: 2 });
const resolved = device.createBuffer(
  { size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const readback = device.createBuffer(
  { size: size * size * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass({
  timestampWrites: { querySet: timing, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
});
pass.pushDebugGroup("gemm");
pass.setPipeline(pipeline);
pass.setBindGroup(0, group);
pass.dispatchWorkgroups(Math.ceil(size / 128), Math.ceil(size / 64), 1);
pass.popDebugGroup();
pass.end();
encoder.resolveQuerySet(timing, 0, 2, resolved, 0);
encoder.copyBufferToBuffer(o, 0, readback, 0, size * size * 4);
device.queue.submit([encoder.finish()]);
await device.queue.onSubmittedWorkDone();

await readback.mapAsync(GPUMapMode.READ);
const out = new Float32Array(readback.getMappedRange().slice(0));
readback.unmap();

// The same product on the host, on one row, so a wrong kernel cannot pass.
let expected = 0;
for (let k = 0; k < size; k += 1) expected += left[k]! * right[k * size]!;
console.log(`out[0] ${out[0]!.toFixed(4)} expected ${expected.toFixed(4)}`);
if (Math.abs(out[0]! - expected) > 1e-3) throw new Error("the wgpu GEMM is wrong");
console.log("wgpu smoke passed");
