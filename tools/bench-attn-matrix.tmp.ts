/** Differential and speed test of the matrix flash kernel vs the register one. */
import { create, globals } from "webgpu";
import { createAttentionRegisterFlashShader } from "../src/evoformer/attention.js";
import {
  attentionMatrixShape, createAttentionMatrixFlashShader, ATTENTION_MATRIX_QUERY_TILE,
} from "../src/evoformer/attention-matrix.js";
import { attentionMatrixStorageBytes } from "../src/evoformer/attention-matrix.js";
import type { SubgroupMatrixConfig } from "../src/runtime/gemm-selection.js";
Object.assign(globalThis, globals);

const f = process.env.AFWEBGPU_DAWN_FEATURES;
const gpu = create(f ? [`enable-dawn-features=${f}`] : []);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no adapter");
const configs = ((adapter.info as unknown as {
  subgroupMatrixConfigs?: readonly SubgroupMatrixConfig[] }).subgroupMatrixConfigs) ?? [];
const big = 2 * 1024 * 1024 * 1024 - 4;
const wanted = ["subgroups", "subgroup-size-control", "shader-f16",
  "chromium-experimental-subgroup-matrix"] as const;
const device = await adapter.requestDevice({
  requiredFeatures: wanted.filter((name) => adapter.features.has(name as GPUFeatureName)) as GPUFeatureName[],
  requiredLimits: {
    maxBufferSize: big, maxStorageBufferBindingSize: big,
    maxComputeWorkgroupStorageSize: Math.min(adapter.limits.maxComputeWorkgroupStorageSize,
      Math.max(16384, attentionMatrixStorageBytes(32))),
    maxComputeInvocationsPerWorkgroup: 256,
  },
});
console.log(`workgroup storage needed=${attentionMatrixStorageBytes(32)} B `
  + `granted=${device.limits.maxComputeWorkgroupStorageSize} B`);

const batch = Number(process.env.BENCH_BATCH ?? "64");
const queries = Number(process.env.BENCH_QUERIES ?? "800");
const heads = 4;
const headDim = 32;
const vectors = headDim / 4;
const elements = batch * queries * heads * headDim;

const shape = attentionMatrixShape(device, headDim, configs);
console.log(`configs=${configs.length} shape=${JSON.stringify(shape)}`);
if (shape === undefined) { console.log("no usable matrix configuration"); process.exit(0); }

let seed = 99;
const rand = (n: number): Float32Array => Float32Array.from({ length: n }, () => {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed / 0x100000000 - 0.5;
});
const q = rand(elements), k = rand(elements), v = rand(elements), g = rand(elements);
const mask = Float32Array.from({ length: batch * queries }, () => 1);
const bias = rand(heads * queries * queries);
const parameters = new Uint32Array(20);
parameters[0] = batch; parameters[1] = queries; parameters[2] = heads * headDim;
parameters[3] = heads; parameters[4] = headDim; parameters[5] = 0; parameters[6] = 1;
parameters[16] = 0; parameters[17] = batch; parameters[18] = queries;

const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
const make = (data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer => {
  const buffer = device.createBuffer({ size: data.byteLength, usage });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};
const qb = make(q, storage), kb = make(k, storage), vb = make(v, storage), gb = make(g, storage);
// The same keys and values, already half precision, for the variant that lets
// the units address them where they lie.
const toHalf = (source: Float32Array): Uint16Array => {
  const view = new DataView(new ArrayBuffer(4));
  return Uint16Array.from(source, (value) => {
    view.setFloat32(0, value);
    const bits = view.getUint32(0);
    const sign = (bits >>> 16) & 0x8000;
    let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
    let mantissa = bits & 0x7fffff;
    if (exponent <= 0) return sign;
    if (exponent >= 31) return sign | 0x7c00;
    // Round to nearest, ties to even, which is what the GPU conversion does.
    const round = (mantissa & 0x1000) !== 0
      && ((mantissa & 0x0fff) !== 0 || ((mantissa >>> 13) & 1) !== 0);
    mantissa >>>= 13;
    if (round) { mantissa += 1; if (mantissa === 0x400) { mantissa = 0; exponent += 1; } }
    return exponent >= 31 ? sign | 0x7c00 : sign | (exponent << 10) | mantissa;
  });
};
const khb = make(toHalf(k) as unknown as Uint32Array, storage);
const vhb = make(toHalf(v) as unknown as Uint32Array, storage);
const mb = make(mask, storage), bb = make(bias, storage);
const pb = make(parameters, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
const ob = device.createBuffer({ size: elements * 4,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
const rb = device.createBuffer({ size: elements * 4,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

interface Candidate {
  readonly label: string;
  readonly pipeline: GPUComputePipeline;
  readonly group: GPUBindGroup;
  readonly groupsX: number;
  best: number;
  output?: Float32Array;
}

async function build(label: string, code: string, tile: number, half = false): Promise<Candidate> {
  device.pushErrorScope("validation");
  const pipeline = device.createComputePipeline({ label, layout: "auto",
    compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" } });
  const failure = await device.popErrorScope();
  if (failure !== null) throw new Error(`${label}: ${failure.message}`);
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
    entries: [qb, half ? khb : kb, half ? vhb : vb, gb, mb, bb, pb, ob]
      .map((buffer, binding) => ({ binding, resource: { buffer } })) });
  return { label, pipeline, group, groupsX: Math.ceil(queries / tile), best: Number.POSITIVE_INFINITY };
}

async function dispatch(candidate: Candidate): Promise<number> {
  const encoder = device.createCommandEncoder({ label: candidate.label });
  const pass = encoder.beginComputePass();
  pass.setPipeline(candidate.pipeline);
  pass.setBindGroup(0, candidate.group);
  pass.dispatchWorkgroups(candidate.groupsX, batch, heads);
  pass.end();
  const started = performance.now();
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - started;
}

async function readOutput(): Promise<Float32Array> {
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(ob, 0, rb, 0, elements * 4);
  device.queue.submit([encoder.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const output = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap();
  return output;
}

const candidates = [
  await build("register", createAttentionRegisterFlashShader(headDim, 1, "f32"), 64),
  await build("matrix", createAttentionMatrixFlashShader(headDim, shape), ATTENTION_MATRIX_QUERY_TILE),
  await build("matrix-f16kv",
    createAttentionMatrixFlashShader(headDim, shape, true), ATTENTION_MATRIX_QUERY_TILE, true),
];

// The card idles at 180 MHz and takes a moment to reach its boost clock, so
// whichever kernel is measured first otherwise pays the whole ramp: measured
// cold, the unchanged register kernel read 8.04 ms against 1.64 ms warm.
for (let warm = 0; warm < 12; warm += 1) {
  for (const candidate of candidates) await dispatch(candidate);
}
// Interleaved, so a clock or a neighbour moving across the sweep lands on
// every candidate rather than ranking whichever one it happened to hit.
for (let round = 0; round < 6; round += 1) {
  for (const candidate of candidates) {
    candidate.best = Math.min(candidate.best, await dispatch(candidate));
  }
}
for (const candidate of candidates) {
  await dispatch(candidate);
  candidate.output = await readOutput();
}

const reference = candidates[0]!;
const flops = 2 * 2 * batch * heads * queries * queries * headDim;
console.log(`batch=${batch} queries=${queries} heads=${heads} headDim=${headDim}`);
let scale = 0;
for (const value of reference.output!) scale = Math.max(scale, Math.abs(value));
for (const candidate of candidates) {
  let worst = 0;
  for (let i = 0; i < reference.output!.length; i += 1) {
    worst = Math.max(worst, Math.abs(reference.output![i]! - candidate.output![i]!));
  }
  console.log(`  ${candidate.label.padEnd(13)} ${candidate.best.toFixed(2).padStart(7)} ms  `
    + `${(flops / candidate.best / 1e9).toFixed(1).padStart(6)} TFLOP/s  `
    + `${(reference.best / candidate.best).toFixed(2)}x  relerr ${(worst / scale).toExponential(2)}`);
}
device.destroy(); process.exit(0);
