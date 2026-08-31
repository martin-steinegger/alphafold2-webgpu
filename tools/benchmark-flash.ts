/**
 * Isolated microbenchmark for the flash-attention kernel.
 *
 * The whole-operator benchmark is dominated by uploads and readback, which
 * hides the kernel differences it is meant to expose. This encodes the flash
 * dispatch alone, many times per submission, against persistent buffers.
 */
import { create, globals } from "webgpu";
import {
  createAttentionParameters, selectAttentionFlashKernel,
  type AttentionFlashVariant, type AttentionInput,
} from "../src/evoformer/attention.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

Object.assign(globalThis, globals);

const VARIANTS: readonly AttentionFlashVariant[] = [
  "register", "subgroup-4x8", "subgroup-key32", "subgroup-8x16", "subgroup-8x32", "subgroup-8x64",
  "subgroup-16x64", "subgroup-32x64", "subgroup-64x64",
];

interface Shape {
  readonly name: string; readonly batch: number; readonly queries: number;
  readonly heads: number; readonly headDim: number; readonly pairBias: boolean;
}

// Main Evoformer, 59 residues, 508 clustered rows (+4 template rows in Multimer).
const SHAPES: readonly Shape[] = [
  { name: "msa-column L=59 N=512", batch: 59, queries: 512, heads: 8, headDim: 32, pairBias: false },
  { name: "msa-row    L=59 N=512", batch: 512, queries: 59, heads: 8, headDim: 32, pairBias: true },
  { name: "msa-column L=256 N=512", batch: 256, queries: 512, heads: 8, headDim: 32, pairBias: false },
  { name: "msa-row    L=256 N=512", batch: 512, queries: 256, heads: 8, headDim: 32, pairBias: true },
];

const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter is available");
const device = await requestAlphaFoldDevice(adapter);

function randomFloats(count: number, scale = 1): Float32Array {
  const values = new Float32Array(count);
  let state = 0x2545f491;
  for (let index = 0; index < count; index += 1) {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    values[index] = ((state / 0x7fffffff) - 0.5) * scale;
  }
  return values;
}

function upload(label: string, data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({ label, size: data.byteLength, usage, mappedAtCreation: true });
  new (data instanceof Float32Array ? Float32Array : Uint32Array)(buffer.getMappedRange()).set(data as never);
  buffer.unmap();
  return buffer;
}

const ITERATIONS = 20;
const REPEATS = 5;
const results: Record<string, Record<string, number>> = {};

for (const shape of SHAPES) {
  const { batch, queries, heads, headDim } = shape;
  const channels = heads * headDim;
  const rows = batch * queries;
  const storage = GPUBufferUsage.STORAGE;
  const query = upload("q", randomFloats(rows * channels), storage);
  const key = upload("k", randomFloats(rows * channels), storage);
  const value = upload("v", randomFloats(rows * channels), storage);
  const gate = upload("g", randomFloats(rows * channels), storage);
  const mask = upload("m", new Float32Array(rows).fill(1), storage);
  const pairBias = upload("pb", randomFloats(shape.pairBias ? heads * queries * queries : 1), storage);
  const output = upload("o", new Float32Array(rows * channels),
    storage | GPUBufferUsage.COPY_SRC);
  const reference = device.createBuffer({
    size: rows * channels * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  const input = {
    batch, queryLength: queries, channels, heads,
    ...(shape.pairBias ? { pairBias: { source: "normalized-input", projectionWeight: new Float32Array(channels * heads) } } : {}),
  } as unknown as AttentionInput;
  const parameters = upload("p", createAttentionParameters(input, new Array(16).fill(0)), GPUBufferUsage.UNIFORM);
  const buffers = [query, key, value, gate, mask, pairBias, parameters, output];

  console.error(`shape ${shape.name} buffers ready`);
  results[shape.name] = {};
  let baseline: Float32Array | undefined;
  for (const variant of VARIANTS) {
    let kernel;
    try { kernel = selectAttentionFlashKernel(device, headDim, variant); }
    catch { continue; }
    console.error(`  variant ${variant}`);
    const pipeline = device.createComputePipeline({
      label: kernel.cacheKey, layout: "auto",
      compute: { module: device.createShaderModule({ code: kernel.shader }), entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const groupsX = Math.ceil(queries / kernel.queryTile);
    const encodeRun = (iterations: number): void => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      for (let iteration = 0; iteration < iterations; iteration += 1) {
        pass.dispatchWorkgroups(groupsX, batch, heads);
      }
      pass.end();
      device.queue.submit([encoder.finish()]);
    };
    encodeRun(2);
    await device.queue.onSubmittedWorkDone();
    let best = Number.POSITIVE_INFINITY;
    for (let repeat = 0; repeat < REPEATS; repeat += 1) {
      const start = performance.now();
      encodeRun(ITERATIONS);
      await device.queue.onSubmittedWorkDone();
      best = Math.min(best, (performance.now() - start) / ITERATIONS);
    }
    results[shape.name]![variant] = best;
    // Correctness: every variant must agree with the first one measured.
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(output, 0, reference, 0, rows * channels * 4);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    await reference.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(reference.getMappedRange().slice(0));
    reference.unmap();
    if (baseline === undefined) baseline = values;
    else {
      let maximum = 0;
      for (let index = 0; index < values.length; index += 1) {
        maximum = Math.max(maximum, Math.abs(values[index]! - baseline[index]!));
      }
      if (maximum > 2e-3) results[shape.name]![`${variant}!MISMATCH`] = maximum;
    }
  }
  for (const buffer of [...buffers, reference]) buffer.destroy();
}

for (const [shape, entry] of Object.entries(results)) {
  console.log(`\n== ${shape} ==`);
  const ordered = Object.entries(entry).filter(([name]) => !name.includes("!"))
    .sort((a, b) => a[1] - b[1]);
  const fastest = ordered[0]![1];
  for (const [variant, milliseconds] of ordered) {
    console.log(`  ${variant.padEnd(18)} ${milliseconds.toFixed(3)} ms  (${(milliseconds / fastest).toFixed(2)}x)`);
  }
  for (const [name, value] of Object.entries(entry)) {
    if (name.includes("!")) console.log(`  ${name} maxAbs=${value}`);
  }
}
device.destroy();
