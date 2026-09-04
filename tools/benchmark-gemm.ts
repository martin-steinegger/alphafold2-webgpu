/**
 * Microbenchmark for the dense projections that dominate the Evoformer.
 *
 * Every AlphaFold projection is row-major A[M,K] x row-major W[K,N] + bias,
 * optionally with ReLU. This measures candidate kernels for that one shape
 * family at the sizes the model actually runs, and checks them against the
 * kernel currently in production.
 */
import { create, globals } from "webgpu";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { CANDIDATES, SHAPES } from "./gemm-candidates.js";

Object.assign(globalThis, globals);

const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter is available");
const device = await requestAlphaFoldDevice(adapter);
device.addEventListener("uncapturederror", (event) => {
  console.error("UNCAPTURED", (event as GPUUncapturedErrorEvent).error.message);
});

function pseudoRandom(count: number): Float32Array {
  const values = new Float32Array(count);
  let state = 0x9e3779b9;
  for (let index = 0; index < count; index += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values[index] = (state / 0xffffffff - 0.5) * 0.5;
  }
  return values;
}

function uploadBuffer(data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer {
  const buffer = device.createBuffer({ size: data.byteLength, usage, mappedAtCreation: true });
  if (data instanceof Float32Array) new Float32Array(buffer.getMappedRange()).set(data);
  else new Uint32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();
  return buffer;
}

for (const shape of SHAPES) {
  const { rows, inner, columns } = shape;
  const source = uploadBuffer(pseudoRandom(rows * inner), GPUBufferUsage.STORAGE);
  const weightValues = new Float32Array(inner * columns + columns);
  weightValues.set(pseudoRandom(inner * columns + columns));
  const weights = uploadBuffer(weightValues, GPUBufferUsage.STORAGE);
  const parameters = uploadBuffer(
    new Uint32Array([rows, inner, columns, 0, inner * columns, 1, 0, 0]), GPUBufferUsage.UNIFORM);
  const output = uploadBuffer(new Float32Array(rows * columns),
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const buffers = [source, weights, parameters, output];
  console.log(`\n== ${shape.name} ==`);
  const timings: [string, number][] = [];
  // Timing only: dawn-node aborts or hangs when mapAsync interleaves with
  // pipeline creation. Numerical agreement is gated by the GPU differential tests.
  for (const candidate of CANDIDATES) {
    if (candidate.requiresF16 === true && !device.features.has("shader-f16")) continue;
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({
      label: candidate.name, layout: "auto",
      compute: { module: device.createShaderModule({ code: candidate.shader }), entryPoint: "main" },
    });
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const groupsX = Math.ceil(columns / candidate.tileColumns);
    const groupsY = Math.ceil(rows / candidate.tileRows);
    const run = async (): Promise<number> => {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(groupsX, groupsY, 1);
      pass.end();
      const start = performance.now();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return performance.now() - start;
    };
    await run();
    const error = await device.popErrorScope();
    if (error !== null) { console.log(`  ${candidate.name.padEnd(16)} VALIDATION ${error.message.slice(0, 120)}`); continue; }
    let best = Number.POSITIVE_INFINITY;
    for (let repeat = 0; repeat < 6; repeat += 1) best = Math.min(best, await run());
    const gflops = 2 * rows * inner * columns / (best / 1000) / 1e9;
    timings.push([candidate.name, best]);
    console.log(`  ${candidate.name.padEnd(16)} ${best.toFixed(3)} ms  ${(gflops / 1000).toFixed(2)} TFLOP/s`);
  }
  const fastest = timings.sort((a, b) => a[1] - b[1])[0]!;
  const baseline = timings.find(([name]) => name === "current-64x128k8");
  if (baseline !== undefined) {
    console.log(`  -> fastest ${fastest[0]} (${(baseline[1] / fastest[1]).toFixed(2)}x over current)`);
  }
  for (const buffer of buffers) buffer.destroy();
}
device.destroy();
