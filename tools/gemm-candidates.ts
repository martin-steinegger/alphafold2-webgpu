import { createLinearShader } from "../src/evoformer/transition.js";
import { GEMM_TILE_COLUMNS, GEMM_TILE_ROWS } from "../src/runtime/gemm.js";

/**
 * The dense-projection kernels, and the shapes the model runs them at.
 *
 * Every AlphaFold projection is row-major A[M,K] x row-major W[K,N] + bias,
 * optionally with ReLU, and one hand-tiled kernel serves all of them. Which
 * tile is fastest depends on the shape and on the GPU, so the candidates live
 * here where both the Node benchmark and the in-browser one can measure them:
 * a tile chosen on a workstation says little about a laptop.
 */
export interface Candidate {
  readonly name: string;
  readonly shader: string;
  /** Output tile covered by one workgroup. */
  readonly tileRows: number;
  readonly tileColumns: number;
  /** Skipped unless the device has the shader-f16 feature. */
  readonly requiresF16?: boolean;
  /**
   * Skipped unless the adapter offers a matrix unit of this component type and
   * shape, and unless the shape's M, N and K are all multiples of `alignment`.
   */
  readonly requiresSubgroupMatrix?: { readonly componentType: string; readonly size: number };
  readonly alignment?: number;
  /** Measures arithmetic only: no bias, no activation, so it is not verified. */
  readonly throughputOnly?: boolean;
  /** What the kernel's bindings expect, which is not always plain f32. */
  readonly sourceFormat?: "f32" | "f16";
  readonly weightFormat?: "f32" | "f16";
  readonly outputFormat?: "f32" | "f16";
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
function tiledGemm(
  tileRows: number, tileColumns: number, tileInner: number,
  precision: "f32" | "f16" | "f16-mixed" | "f16-chunked" = "f32",
): string {
  // Half precision is generated, not patched in: the staged tiles, the
  // products and the accumulators are all f16, and only the loads from and
  // stores to the f32 buffers convert. Apple issues f16 multiply-accumulate
  // at twice the f32 rate and stages half the bytes.
  // "f16-mixed" multiplies in half precision and accumulates in single: the
  // multiply is where the rate doubles, while a K-long reduction in f16 is
  // where the error grows, so the two can be bought separately.
  // "f16-chunked" buys both: the reduction stays in f16 for the depth of one
  // staged tile and folds into an f32 running sum once per tile, so the error
  // grows with the square root of the tile depth rather than of K while every
  // multiply and all but one add per tile stay in half precision.
  const half = precision !== "f32";
  const scalar = half ? "f16" : "f32";
  const accumulator = precision === "f16" ? "f16" : "f32";
  const vec = `vec4<${accumulator}>`;
  const cast = (value: string): string => half ? `${scalar}(${value})` : value;
  const threads = 256;
  const registerColumns = 4;
  const columnThreads = tileColumns / registerColumns;
  const rowThreads = threads / columnThreads;
  const registerRows = tileRows / rowThreads;
  if (!Number.isInteger(rowThreads) || !Number.isInteger(registerRows)) {
    throw new Error(`bad tiling ${tileRows}x${tileColumns}`);
  }
  const weightVectors = (tileInner * tileColumns) / 4;
  const declare = (name: string, count: number): string => Array.from({ length: count },
    (_, index) => `  var ${name}${index} = ${vec}(0.0);`).join("\n");
  const chunked = precision === "f16-chunked";
  const product = (r: number): string => precision === "f16-mixed"
    ? `vec4<f32>(a${r} * w)` : `a${r} * w`;
  const accumulate = Array.from({ length: registerRows }, (_, r) =>
    `      let a${r} = tile_source[k * ${tileRows}u + row_base + ${r}u];`
    + `\n      ${chunked ? `chunk${r} += a${r} * w` : `acc${r} += ${product(r)}`};`).join("\n");
  const declareChunks = chunked
    ? Array.from({ length: registerRows },
      (_, index) => `    var chunk${index} = vec4<f16>(0.0);`).join("\n") + "\n"
    : "";
  const foldChunks = chunked
    ? "\n" + Array.from({ length: registerRows },
      (_, index) => `    acc${index} += vec4<f32>(chunk${index});`).join("\n")
    : "";
  return `${half ? "enable f16;\n" : ""}${PARAMETERS}
// A is staged transposed (k-major) so one k step reads consecutive rows, and
// as scalars rather than vectors: adjacent elements are written by different
// invocations, and assigning separate lanes of one workgroup vec4
// concurrently is a data race that Metal lowers to a clobbering
// read-modify-write of the whole vector. The shared GEMM fixed this once
// already; these candidates were extracted from the code before that.
var<workgroup> tile_source: array<${scalar}, ${tileRows * tileInner}>;
var<workgroup> tile_weight: array<vec4<${scalar}>, ${weightVectors}>;

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
      tile_source[load_k * ${tileRows}u + load_row] = ${cast("value")};
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
      tile_weight[item] = ${half ? `vec4<${scalar}>(value)` : "value"};
    }
    workgroupBarrier();
${declareChunks}    for (var k = 0u; k < ${tileInner}u; k += 1u) {
      let w = tile_weight[k * ${tileColumns / 4}u + column_thread];
${accumulate}
    }${foldChunks}
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
${Array.from({ length: registerRows }, (_, index) =>
  `      if (r == ${index}u) { value = ${accumulator === "f16" ? `vec4<f32>(acc${index})` : `acc${index}`}; }`).join("\n")}
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

/**
 * The hardware matrix units, through Chromium's experimental subgroup matrix.
 *
 * Apple Metal 3 offers 8x8x8 tiles in f32 and f16; this machine's NVIDIA
 * adapter offers only integer ones, so nothing here can be compiled or
 * checked locally. One subgroup owns one output tile and walks K, which is
 * the simplest arrangement that uses the units at all: bias, activation and
 * bounds are left out, so it measures throughput and nothing else.
 */
function subgroupMatrixGemm(componentType: "f32" | "f16", size: number): string {
  const out = componentType === "f16" ? "f16" : "f32";
  const enables = `${componentType === "f16" ? "enable f16;\n" : ""}enable chromium_experimental_subgroup_matrix;\n`;
  const buffer = componentType === "f16" ? "f16" : "f32";
  return `${enables}${PARAMETERS.replace("array<f32>;\n@group(0) @binding(1) var<storage, read> weights: array<f32>;",
    `array<${buffer}>;\n@group(0) @binding(1) var<storage, read> weights: array<${buffer}>;`)
    .replace("var<storage, read_write> output: array<f32>;", `var<storage, read_write> output: array<${out}>;`)}
@compute @workgroup_size(32, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>) {
  let row_origin = group.y * ${size}u;
  let column_origin = group.x * ${size}u;
  var acc = subgroup_matrix_result<${componentType}, ${size}, ${size}>();
  for (var k0 = 0u; k0 < parameters.inner; k0 += ${size}u) {
    let left = subgroupMatrixLoad<subgroup_matrix_left<${componentType}, ${size}, ${size}>>(
      &source, row_origin * parameters.inner + k0, false, parameters.inner);
    let right = subgroupMatrixLoad<subgroup_matrix_right<${componentType}, ${size}, ${size}>>(
      &weights, parameters.weight_offset + k0 * parameters.columns + column_origin, false, parameters.columns);
    acc = subgroupMatrixMultiplyAccumulate(left, right, acc);
  }
  subgroupMatrixStore(&output, row_origin * parameters.columns + column_origin, acc,
    false, parameters.columns);
}`;
}

export const CANDIDATES: readonly Candidate[] = [
  // The kernel the model actually runs, so the baseline is not a
  // reimplementation of it that may have drifted.
  { name: "production", shader: createLinearShader(false), tileRows: GEMM_TILE_ROWS,
    tileColumns: GEMM_TILE_COLUMNS },
  { name: "current-64x128k8", shader: tiledGemm(64, 128, 8), tileRows: 64, tileColumns: 128 },
  { name: "tiled-64x64k16", shader: tiledGemm(64, 64, 16), tileRows: 64, tileColumns: 64 },
  { name: "tiled-64x128k16", shader: tiledGemm(64, 128, 16), tileRows: 64, tileColumns: 128 },
  { name: "tiled-128x128k8", shader: tiledGemm(128, 128, 8), tileRows: 128, tileColumns: 128 },
  // 16 KiB of workgroup storage exactly, with no headroom against the WebGPU
  // minimum, so it is measured but not shipped.
  { name: "tiled-128x128k16", shader: tiledGemm(128, 128, 16), tileRows: 128, tileColumns: 128 },
  // Reads the source as packed half words, the way the model stores it.
  { name: "f16-source-64x128k8", shader: tiledGemmPackedSource(64, 128, 8), tileRows: 64, tileColumns: 128,
    sourceFormat: "f16" },
  { name: "f16-math-64x128k8", shader: tiledGemm(64, 128, 8, "f16"), tileRows: 64, tileColumns: 128,
    requiresF16: true },
  { name: "f16-math-64x128k16", shader: tiledGemm(64, 128, 16, "f16"), tileRows: 64, tileColumns: 128,
    requiresF16: true },
  { name: "f16-math-64x64k16", shader: tiledGemm(64, 64, 16, "f16"), tileRows: 64, tileColumns: 64,
    requiresF16: true },
  // Half-precision products, single-precision accumulation.
  { name: "f16-mixed-64x128k8", shader: tiledGemm(64, 128, 8, "f16-mixed"), tileRows: 64, tileColumns: 128,
    requiresF16: true },
  { name: "f16-mixed-64x64k16", shader: tiledGemm(64, 64, 16, "f16-mixed"), tileRows: 64, tileColumns: 64,
    requiresF16: true },
  // Half-precision products and accumulation within a staged tile, folded
  // into a single-precision running sum once per tile.
  { name: "f16-chunked-64x128k8", shader: tiledGemm(64, 128, 8, "f16-chunked"), tileRows: 64,
    tileColumns: 128, requiresF16: true },
  { name: "f16-chunked-64x128k16", shader: tiledGemm(64, 128, 16, "f16-chunked"), tileRows: 64,
    tileColumns: 128, requiresF16: true },
  { name: "f16-chunked-64x64k16", shader: tiledGemm(64, 64, 16, "f16-chunked"), tileRows: 64,
    tileColumns: 64, requiresF16: true },
  { name: "matrix-f32-8x8x8", shader: subgroupMatrixGemm("f32", 8), tileRows: 8, tileColumns: 8,
    requiresSubgroupMatrix: { componentType: "f32", size: 8 }, alignment: 8, throughputOnly: true },
  // An f16 matrix produces an f16 result, which only an f16 buffer can hold.
  { name: "matrix-f16-8x8x8", shader: subgroupMatrixGemm("f16", 8), tileRows: 8, tileColumns: 8,
    requiresF16: true, requiresSubgroupMatrix: { componentType: "f16", size: 8 },
    alignment: 8, throughputOnly: true, sourceFormat: "f16", weightFormat: "f16",
    outputFormat: "f16" },
];

export interface Shape { readonly name: string; readonly rows: number; readonly inner: number; readonly columns: number; }
export const SHAPES: readonly Shape[] = [
  // The outer product mean's contraction, the largest single kernel in a
  // block: few rows, a very wide output, repeated once per residue block.
  { name: "opm-contract M=128 K=508 N=32000", rows: 128, inner: 508, columns: 32000 },
  // The outer product mean's output projection: one column tile wide, so the
  // grid is as many workgroups as it has row tiles and no more.
  { name: "opm-out   M=4096 K=1024 N=128", rows: 4096, inner: 1024, columns: 128 },
  { name: "opm-out2  M=3000 K=1024 N=128", rows: 3000, inner: 1024, columns: 128 },
  { name: "project   M=29972 K=256 N=1024", rows: 29972, inner: 256, columns: 1024 },
  { name: "output    M=29972 K=256 N=256", rows: 29972, inner: 256, columns: 256 },
  { name: "trans1    M=29972 K=256 N=1024", rows: 29972, inner: 256, columns: 1024 },
  { name: "trans2    M=29972 K=1024 N=256", rows: 29972, inner: 1024, columns: 256 },
  { name: "extra1    M=60416 K=64 N=256", rows: 60416, inner: 64, columns: 256 },
];

