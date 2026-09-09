/**
 * The same generated GEMM through wgpu and through Dawn, in both precisions,
 * with the injected runtime checks on and off on both sides.
 *
 * The comparison this replaces had Dawn unclamped against wgpu fully checked,
 * which measures the checks rather than the backends. Every arm prints a
 * checksum over its output, so a faster wrong kernel cannot pass for a win.
 *
 * The native side opens one device a process, so one invocation measures one
 * wgpu arm: AFWEBGPU_WGPU_UNCHECKED=1 picks which.
 *
 * Usage: tsx tools/bench-wgpu-vs-dawn.ts [f32|matrix] [size] [repeats]
 */
import { create, globals } from "webgpu";
import { DAWN_MATRIX, WGPU_MATRIX, type MatrixSpelling } from "../src/runtime/dialect.js";
import { createTiledGemmShader } from "../src/runtime/gemm.js";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { requestWgpuAdapter } from "../src/runtime/wgpu/adapter.js";
import { wgpuNative } from "../src/runtime/wgpu/backend.js";

const precision = (process.argv[2] ?? "f32") === "matrix" ? "matrix" : "f32";
const size = Number(process.argv[3] ?? "1024");
const repeats = Number(process.argv[4] ?? "50");

const SHAPE = { M: 16, N: 16, K: 16,
  componentType: "f16" as const, resultComponentType: "f32" as const };

const SPEC = {
  preamble: `${precision === "matrix" ? "enable f16;\n" : ""}\
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> o: array<f32>;
struct P { rows: u32, inner: u32, columns: u32, pad: u32 };
@group(0) @binding(3) var<uniform> p: P;`,
  rows: "p.rows", inner: "p.inner", columns: "p.columns",
  sourceElement: "a[row * p.inner + k]", weightElement: "b[k * p.columns + column]",
  store: "o[row * p.columns + column] = element;",
  sourceArray: { array: "a", stride: "p.inner" },
  weightArray: { array: "b", stride: "p.columns" },
};

const variant = precision === "matrix"
  ? { precision: "matrix" as const, inner: 8 as const, matrix: SHAPE }
  : { precision: "f32" as const, inner: 8 as const };
const source = (spelling: MatrixSpelling): string =>
  createTiledGemmShader(SPEC, variant, spelling);

const TILE_ROWS = 64;
const TILE_COLUMNS = 128;
const groupsX = Math.ceil(size / TILE_COLUMNS);
const groupsY = Math.ceil(size / TILE_ROWS);

const left = Float32Array.from({ length: size * size }, (_, i) => ((i % 17) - 8) / 8);
const right = Float32Array.from({ length: size * size }, (_, i) => ((i % 13) - 6) / 6);
const parameters = new Uint32Array([size, size, size, 0]);
const outputBytes = size * size * 4;
const checkedElements = Math.min(size * size, 4096);

const checksum = (values: ArrayLike<number>): number => {
  let sum = 0;
  for (let i = 0; i < Math.min(values.length, checkedElements); i += 1) sum += values[i]!;
  return Math.round(sum * 100) / 100;
};

interface Arm { best: number; checksum: number }

/** The four buffers and the pipeline, built the same way on either backend. */
async function build(device: GPUDevice, code: string): Promise<{
  pipeline: GPUComputePipeline; group: GPUBindGroup; output: GPUBuffer;
}> {
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const upload = (data: ArrayBufferView, usage: number): GPUBuffer => {
    const buffer = device.createBuffer({ size: data.byteLength, usage });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const output = device.createBuffer({ size: outputBytes, usage: storage });
  const pipeline = await device.createComputePipelineAsync({
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [upload(left, storage), upload(right, storage), output,
      upload(parameters, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)]
      .map((buffer, binding) => ({ binding, resource: { buffer } })),
  });
  return { pipeline, group, output };
}

async function readOutput(device: GPUDevice, output: GPUBuffer): Promise<number> {
  const bytes = checkedElements * 4;
  const readback = device.createBuffer(
    { size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const sum = checksum(new Float32Array(readback.getMappedRange().slice(0)));
  readback.unmap();
  return sum;
}

async function runWgpu(unchecked: boolean): Promise<Arm> {
  const adapter = requestWgpuAdapter(unchecked);
  const device = await adapter.requestDevice({
    requiredFeatures: [...adapter.features] as GPUFeatureName[],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    },
  });
  const { pipeline, group, output } = await build(device, source(WGPU_MATRIX));
  // Timed inside one submission, which is what the Dawn arm below also does.
  const handles = pipeline as unknown as { handle: number };
  const native = wgpuNative();
  native.timeDispatches(handles.handle, (group as unknown as { handle: number }).handle,
    groupsX, groupsY, 1, 2);
  let best = Number.POSITIVE_INFINITY;
  for (let round = 0; round < 5; round += 1) {
    best = Math.min(best, native.timeDispatches(handles.handle,
      (group as unknown as { handle: number }).handle, groupsX, groupsY, 1, repeats) / repeats);
  }
  return { best, checksum: await readOutput(device, output) };
}

async function runDawn(unclamped: boolean): Promise<Arm> {
  Object.assign(globalThis, globals);
  const gpu = create(dawnInstanceFlags({ unclamped }));
  const adapter = (await gpu.requestAdapter({ powerPreference: "high-performance" }))!;
  const wanted = ["chromium-experimental-subgroup-matrix", "subgroups",
    "subgroup-size-control", "shader-f16"] as const;
  const device = await adapter.requestDevice({
    requiredFeatures: wanted.filter(
      (feature) => adapter.features.has(feature as GPUFeatureName)) as GPUFeatureName[],
    requiredLimits: {
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    },
  });
  const { pipeline, group, output } = await build(device, source(DAWN_MATRIX));
  const once = async (count: number): Promise<number> => {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    for (let i = 0; i < count; i += 1) pass.dispatchWorkgroups(groupsX, groupsY, 1);
    pass.end();
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - started) / count;
  };
  await once(2);
  let best = Number.POSITIVE_INFINITY;
  for (let round = 0; round < 5; round += 1) best = Math.min(best, await once(repeats));
  const sum = await readOutput(device, output);
  device.destroy();
  return { best, checksum: sum };
}

const flops = 2 * size * size * size;
const report = (label: string, arm: Arm): void => {
  console.log(`${label.padEnd(22)} ${arm.best.toFixed(3)} ms  `
    + `${(flops / arm.best / 1e6).toFixed(0)} GFLOP/s  checksum ${arm.checksum}`);
};

console.log(`${precision} ${size}^3, ${repeats} dispatches a round, best of 5`);
report("dawn checked", await runDawn(false));
report("dawn unclamped", await runDawn(true));
const unchecked = process.env.AFWEBGPU_WGPU_UNCHECKED === "1";
report(unchecked ? "wgpu unchecked" : "wgpu checked", await runWgpu(unchecked));
