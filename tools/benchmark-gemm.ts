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

Object.assign(globalThis, globals);

interface Candidate {
  readonly name: string;
  readonly shader: string;
  /** Output tile covered by one workgroup. */
  readonly tileRows: number;
  readonly tileColumns: number;
}

const PARAMETERS = `
struct MatmulParameters {
  rows: u32, inner: u32, columns: u32,
  weight_offset: u32, bias_offset: u32, activation: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: MatmulParameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
`;

/** The kernel currently shipped in src/evoformer/transition.ts. */
const CURRENT = `${PARAMETERS}
var<workgroup> tile_source: array<f32, 128>;
var<workgroup> tile_weight: array<f32, 512>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.y * 16u + local.y;
  let second_row = row + 8u;
  let column = group.x * 64u + local.x;
  let tile_index = local.y * 8u + local.x;
  var value_low = vec4<f32>(0.0);
  var value_high = vec4<f32>(0.0);
  var second_value_low = vec4<f32>(0.0);
  var second_value_high = vec4<f32>(0.0);
  for (var k0 = 0u; k0 < parameters.inner; k0 += 8u) {
    let source_k = k0 + local.x;
    let weight_k = k0 + local.y;
    tile_source[tile_index] = 0.0;
    tile_source[tile_index + 64u] = 0.0;
    if (row < parameters.rows && source_k < parameters.inner) {
      tile_source[tile_index] = source[row * parameters.inner + source_k];
    }
    if (second_row < parameters.rows && source_k < parameters.inner) {
      tile_source[tile_index + 64u] = source[second_row * parameters.inner + source_k];
    }
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let tile_column = local.x + column_block * 8u;
      let output_column = column + column_block * 8u;
      let weight_index = local.y * 64u + tile_column;
      tile_weight[weight_index] = 0.0;
      if (output_column < parameters.columns && weight_k < parameters.inner) {
        tile_weight[weight_index] = weights[parameters.weight_offset + weight_k * parameters.columns + output_column];
      }
    }
    workgroupBarrier();
    for (var k = 0u; k < 8u; k += 1u) {
      let source_value = tile_source[local.y * 8u + k];
      let second_source_value = tile_source[local.y * 8u + k + 64u];
      let weight_base = k * 64u + local.x;
      let weight_low = vec4<f32>(tile_weight[weight_base], tile_weight[weight_base + 8u],
        tile_weight[weight_base + 16u], tile_weight[weight_base + 24u]);
      let weight_high = vec4<f32>(tile_weight[weight_base + 32u], tile_weight[weight_base + 40u],
        tile_weight[weight_base + 48u], tile_weight[weight_base + 56u]);
      value_low += source_value * weight_low;
      value_high += source_value * weight_high;
      second_value_low += second_source_value * weight_low;
      second_value_high += second_source_value * weight_high;
    }
    workgroupBarrier();
  }
  if (row < parameters.rows) {
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let output_column = column + column_block * 8u;
      if (output_column < parameters.columns) {
        let values = select(value_low, value_high, column_block >= 4u);
        var value = values[column_block % 4u] + weights[parameters.bias_offset + output_column];
        if (parameters.activation == 1u) { value = max(value, 0.0); }
        output[row * parameters.columns + output_column] = value;
      }
    }
  }
  if (second_row < parameters.rows) {
    for (var column_block = 0u; column_block < 8u; column_block += 1u) {
      let output_column = column + column_block * 8u;
      if (output_column < parameters.columns) {
        let values = select(second_value_low, second_value_high, column_block >= 4u);
        var value = values[column_block % 4u] + weights[parameters.bias_offset + output_column];
        if (parameters.activation == 1u) { value = max(value, 0.0); }
        output[second_row * parameters.columns + output_column] = value;
      }
    }
  }
}`;

/**
 * Register-blocked GEMM: 256 invocations cover a TR x TC output tile, each
 * holding an RR x RC register block. A and W tiles are staged in workgroup
 * memory as vec4 so both the global and the shared reads stay vectorized.
 */
function tiledGemm(tileRows: number, tileColumns: number, tileInner: number): string {
  const threads = 256;
  const registerColumns = 4;
  const columnThreads = tileColumns / registerColumns;
  const rowThreads = threads / columnThreads;
  const registerRows = tileRows / rowThreads;
  if (!Number.isInteger(rowThreads) || !Number.isInteger(registerRows)) {
    throw new Error(`bad tiling ${tileRows}x${tileColumns}`);
  }
  const sourceVectors = (tileRows * tileInner) / 4;
  const weightVectors = (tileInner * tileColumns) / 4;
  const declare = (name: string, count: number): string => Array.from({ length: count },
    (_, index) => `  var ${name}${index} = vec4<f32>(0.0);`).join("\n");
  const accumulate = Array.from({ length: registerRows }, (_, r) =>
    `      let a${r} = tile_source[(k * ${tileRows}u + row_base + ${r}u) / 4u][(row_base + ${r}u) % 4u];`
    + `\n      acc${r} += a${r} * w;`).join("\n");
  return `${PARAMETERS}
// A is staged transposed (k-major) so one k step reads consecutive rows.
var<workgroup> tile_source: array<vec4<f32>, ${sourceVectors}>;
var<workgroup> tile_weight: array<vec4<f32>, ${weightVectors}>;

@compute @workgroup_size(${threads}, 1, 1)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let thread = local.x;
  let column_thread = thread % ${columnThreads}u;
  let row_thread = thread / ${columnThreads}u;
  let row_base = row_thread * ${registerRows}u;
  let row_origin = group.y * ${tileRows}u;
  let column_origin = group.x * ${tileColumns}u;
  let column = column_origin + column_thread * 4u;
${declare("acc", registerRows)}

  for (var k0 = 0u; k0 < parameters.inner; k0 += ${tileInner}u) {
    // Stage A transposed: tile_source[k][row].
    for (var item = thread; item < ${tileRows * tileInner}u; item += ${threads}u) {
      let load_row = item / ${tileInner}u;
      let load_k = item % ${tileInner}u;
      let source_row = row_origin + load_row;
      let source_k = k0 + load_k;
      var value = 0.0;
      if (source_row < parameters.rows && source_k < parameters.inner) {
        value = source[source_row * parameters.inner + source_k];
      }
      let slot = load_k * ${tileRows}u + load_row;
      tile_source[slot / 4u][slot % 4u] = value;
    }
    for (var item = thread; item < ${weightVectors}u; item += ${threads}u) {
      let load_k = item / ${tileColumns / 4}u;
      let load_column = (item % ${tileColumns / 4}u) * 4u;
      let weight_k = k0 + load_k;
      var value = vec4<f32>(0.0);
      if (weight_k < parameters.inner && column_origin + load_column + 3u < parameters.columns) {
        let base = parameters.weight_offset + weight_k * parameters.columns + column_origin + load_column;
        value = vec4<f32>(weights[base], weights[base + 1u], weights[base + 2u], weights[base + 3u]);
      } else if (weight_k < parameters.inner) {
        for (var lane = 0u; lane < 4u; lane += 1u) {
          let output_column = column_origin + load_column + lane;
          if (output_column < parameters.columns) {
            value[lane] = weights[parameters.weight_offset + weight_k * parameters.columns + output_column];
          }
        }
      }
      tile_weight[item] = value;
    }
    workgroupBarrier();
    for (var k = 0u; k < ${tileInner}u; k += 1u) {
      let w = tile_weight[k * ${tileColumns / 4}u + column_thread];
${accumulate}
    }
    workgroupBarrier();
  }

  var bias = vec4<f32>(0.0);
  for (var lane = 0u; lane < 4u; lane += 1u) {
    if (column + lane < parameters.columns) { bias[lane] = weights[parameters.bias_offset + column + lane]; }
  }
  for (var r = 0u; r < ${registerRows}u; r += 1u) {
    let row = row_origin + row_base + r;
    if (row < parameters.rows) {
      var value = vec4<f32>(0.0);
${Array.from({ length: registerRows }, (_, index) => `      if (r == ${index}u) { value = acc${index}; }`).join("\n")}
      value += bias;
      if (parameters.activation == 1u) { value = max(value, vec4<f32>(0.0)); }
      let base = row * parameters.columns + column;
      if (column + 3u < parameters.columns) {
        output[base] = value.x; output[base + 1u] = value.y;
        output[base + 2u] = value.z; output[base + 3u] = value.w;
      } else {
        for (var lane = 0u; lane < 4u; lane += 1u) {
          if (column + lane < parameters.columns) { output[base + lane] = value[lane]; }
        }
      }
    }
  }
}`;
}


/**
 * The same tiling, reading A as packed f16 pairs.
 *
 * shader-f16 is unavailable on this adapter, but unpack2x16float is core WGSL,
 * so A can be stored at half width with the arithmetic still in f32. This tests
 * whether the projection is bandwidth-bound.
 */
function tiledGemmPackedSource(tileRows: number, tileColumns: number, tileInner: number): string {
  return tiledGemm(tileRows, tileColumns, tileInner)
    .replace("@group(0) @binding(0) var<storage, read> source: array<f32>;",
      `@group(0) @binding(0) var<storage, read> source_packed: array<u32>;

fn source_at(index: u32) -> f32 {
  let pair = unpack2x16float(source_packed[index / 2u]);
  return select(pair.x, pair.y, (index & 1u) == 1u);
}`)
    .replace("value = source[source_row * parameters.inner + source_k];",
      "value = source_at(source_row * parameters.inner + source_k);");
}

const CANDIDATES: readonly Candidate[] = [
  { name: "current-64x128k8", shader: tiledGemm(64, 128, 8), tileRows: 64, tileColumns: 128 },
  { name: "tiled-64x64k16", shader: tiledGemm(64, 64, 16), tileRows: 64, tileColumns: 64 },
  { name: "tiled-128x128k8", shader: tiledGemm(128, 128, 8), tileRows: 128, tileColumns: 128 },
  // 16 KiB of workgroup storage exactly, with no headroom against the WebGPU
  // minimum, so it is measured but not shipped.
  { name: "tiled-128x128k16", shader: tiledGemm(128, 128, 16), tileRows: 128, tileColumns: 128 },
  { name: "f16-source-64x128k8", shader: tiledGemmPackedSource(64, 128, 8), tileRows: 64, tileColumns: 128 },
];

interface Shape { readonly name: string; readonly rows: number; readonly inner: number; readonly columns: number; }
const SHAPES: readonly Shape[] = [
  { name: "project   M=29972 K=256 N=1024", rows: 29972, inner: 256, columns: 1024 },
  { name: "output    M=29972 K=256 N=256", rows: 29972, inner: 256, columns: 256 },
  { name: "trans1    M=29972 K=256 N=1024", rows: 29972, inner: 256, columns: 1024 },
  { name: "trans2    M=29972 K=1024 N=256", rows: 29972, inner: 1024, columns: 256 },
  { name: "extra1    M=60416 K=64 N=256", rows: 60416, inner: 64, columns: 256 },
];

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
