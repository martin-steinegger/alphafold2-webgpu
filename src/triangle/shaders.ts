import type { Precision, TriangleShape } from "./types.js";
import type { WeightOffsets } from "./weights.js";
import { createTiledGemmShader } from "../runtime/gemm.js";

export interface TriangleShaders {
  /** Mean and inverse standard deviation of every pair row, computed once. */
  readonly inputStatistics: string;
  /**
   * The output gate for one block of pair rows, from the raw pair and the
   * statistics; the output projection multiplies it in.
   */
  readonly projectGate: string;
  /**
   * The two contraction inputs, each a gated projection of the normalized
   * pair. Incoming blocks the contraction index, so both are block-sized;
   * outgoing blocks the first residue index, so only `a` is, and `b` is
   * filled block by block into a whole tensor before any block contracts.
   */
  readonly projectA: string;
  readonly projectB: string;
  readonly contract: string;
  readonly normalizeHidden: string;
  readonly projectOutput: string;
}

export type TriangleDirection = "outgoing" | "incoming";

const declaration = (precision: Precision): string => precision === "f16" ? "enable f16;\n" : "";
const scalar = (precision: Precision): "f16" | "f32" => precision;
const read = (precision: Precision, expression: string): string =>
  precision === "f16" ? `f32(${expression})` : expression;

/**
 * @param blockRows Rows of the blocked residue axis one step covers. PAIRS
 * stays the whole pair count; BLOCK_PAIRS is one step's worth.
 */
function prelude(
  shape: TriangleShape, precision: Precision, offsets: WeightOffsets, epsilon: number,
  blockRows = shape.length,
): string {
  const offsetConstants = Object.entries(offsets)
    .map(([name, offset]) => `const W_${name.toUpperCase()}: u32 = ${offset}u;`)
    .join("\n");
  return `${declaration(precision)}
const L: u32 = ${shape.length}u;
const CZ: u32 = ${shape.cZ}u;
const CH: u32 = ${shape.cHidden}u;
const PAIRS: u32 = L * L;
const BLOCK_ROWS: u32 = ${blockRows}u;
const BLOCK_PAIRS: u32 = BLOCK_ROWS * L;
const LINEAR_GRID_WIDTH: u32 = 32768u;
const EPSILON: f32 = ${epsilon.toPrecision(9)};
${offsetConstants}

fn logistic(value: f32) -> f32 {
  return 1.0 / (1.0 + exp(-value));
}
`;
}

export function createTriangleShaders(
  shape: TriangleShape,
  precision: Precision,
  offsets: WeightOffsets,
  epsilon = 1e-5,
  direction: TriangleDirection = "outgoing",
  blockRows = shape.length,
): TriangleShaders {
  const common = prelude(shape, precision, offsets, epsilon, blockRows);
  const t = scalar(precision);

  // One workgroup per pair row, so the channel reads of a row are contiguous
  // across lanes. The statistics let every later consumer normalize the raw
  // pair on the fly instead of materializing a normalized copy.
  const inputStatistics = `${common}
@group(0) @binding(0) var<storage, read> source: array<${t}>;
@group(0) @binding(1) var<storage, read_write> statistics: array<f32>;
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * LINEAR_GRID_WIDTH;
  let base = row * CZ;
  var sum = 0.0;
  if (row < PAIRS) {
    for (var c = local.x; c < CZ; c += 64u) { sum += ${read(precision, "source[base + c]")}; }
  }
  partial[local.x] = sum;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  let mean = partial[0] / f32(CZ);
  workgroupBarrier();
  var squared = 0.0;
  if (row < PAIRS) {
    for (var c = local.x; c < CZ; c += 64u) {
      let centered = ${read(precision, "source[base + c]")} - mean;
      squared += centered * centered;
    }
  }
  partial[local.x] = squared;
  workgroupBarrier();
  for (var stride = 32u; stride > 0u; stride >>= 1u) {
    if (local.x < stride) { partial[local.x] += partial[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u && row < PAIRS) {
    statistics[2u * row] = mean;
    statistics[2u * row + 1u] = inverseSqrt(partial[0] / f32(CZ) + EPSILON);
  }
}`;

  // The output gate is a projection of the normalized input, so like the
  // contraction inputs it normalizes the raw pair while loading it.
  const projectGate = createTiledGemmShader({
    preamble: `${common}
@group(0) @binding(0) var<storage, read> z: array<${t}>;
@group(0) @binding(1) var<storage, read> weights: array<${t}>;
@group(0) @binding(2) var<storage, read> statistics: array<f32>;
@group(0) @binding(3) var<storage, read_write> gate: array<f32>;
// x is the first row of this block within the whole pair tensor, y the
// number of rows it spans.
@group(0) @binding(4) var<uniform> block: vec4<u32>;

fn normalized_input(pair_row: u32, k: u32) -> f32 {
  return (${read(precision, "z[pair_row * CZ + k]")} - statistics[2u * pair_row]) * statistics[2u * pair_row + 1u]
    * ${read(precision, "weights[W_LAYERNORMINWEIGHT + k]")} + ${read(precision, "weights[W_LAYERNORMINBIAS + k]")};
}`,
    rows: "block.y",
    inner: "CZ",
    columns: "CZ",
    sourceElement: "normalized_input(block.x + row, k)",
    weightElement: read(precision, "weights[W_LINEARGWEIGHT + column * CZ + k]"),
    store: `gate[row * CZ + column] = logistic(element + ${read(precision, "weights[W_LINEARGBIAS + column]")});`,
  });

  /**
   * One gated projection as a tiled GEMM over the raw pair rows of a block.
   *
   * The normalization is applied while loading the operand, from the row
   * statistics. Even output columns are the projection and odd ones its gate,
   * so the four adjacent columns an invocation holds pair each channel with its
   * gate and the epilogue can combine them with the mask. The result is stored
   * channel-major, which is how the contraction consumes it.
   */
  const project = (operand: "a" | "b", stride: string, offset: string): string => {
    const upper = operand.toUpperCase();
    const weight = (kind: "P" | "G"): string =>
      read(precision, `weights[W_LINEAR${upper}${kind}WEIGHT + (column >> 1u) * CZ + k]`);
    const bias = (kind: "P" | "G", channel: string): string =>
      read(precision, `weights[W_LINEAR${upper}${kind}BIAS + ${channel}]`);
    return createTiledGemmShader({
      preamble: `${common}
@group(0) @binding(0) var<storage, read> z: array<${t}>;
@group(0) @binding(1) var<storage, read> mask: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${t}>;
@group(0) @binding(3) var<storage, read> statistics: array<f32>;
@group(0) @binding(4) var<storage, read_write> ${operand}: array<f32>;
// x is the first row of this block within the whole pair tensor, y the
// number of rows it spans.
@group(0) @binding(5) var<uniform> block: vec4<u32>;

fn normalized_input(pair_row: u32, k: u32) -> f32 {
  return (${read(precision, "z[pair_row * CZ + k]")} - statistics[2u * pair_row]) * statistics[2u * pair_row + 1u]
    * ${read(precision, "weights[W_LAYERNORMINWEIGHT + k]")} + ${read(precision, "weights[W_LAYERNORMINBIAS + k]")};
}`,
      rows: "block.y",
      inner: "CZ",
      columns: "2u * CH",
      sourceElement: "normalized_input(block.x + row, k)",
      weightElement: `select(${weight("G")}, ${weight("P")}, (column & 1u) == 0u)`,
      store: "",
      storeVector: `let pair_mask = mask[block.x + row];
      let h = column >> 1u;
      if (h < CH) {
        ${operand}[h * ${stride} + ${offset} + row] = pair_mask
          * (values[0] + ${bias("P", "h")}) * logistic(values[1] + ${bias("G", "h")});
      }
      if (h + 1u < CH) {
        ${operand}[(h + 1u) * ${stride} + ${offset} + row] = pair_mask
          * (values[2] + ${bias("P", "h + 1u")}) * logistic(values[3] + ${bias("G", "h + 1u")});
      }`,
    });
  };

  // out[h][i][j] = sum_k A[h][i][k] * B[h][j][k], one independent matrix per
  // hidden channel, dispatched along z.
  //
  // The two directions contract over different indices, so they block
  // differently. Outgoing contracts over the second index, so blocking the
  // first leaves one projection whole and writes its own slice of the output.
  // Incoming contracts over the first index, so blocking it shrinks both
  // projections but every block contributes to every output element, and the
  // output is accumulated across blocks instead of partitioned.
  const contract = direction === "outgoing"
    ? createTiledGemmShader({
      preamble: `${common}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// x is the first pair row of the block, w the residue count it spans.
@group(0) @binding(3) var<uniform> block: vec4<u32>;`,
      rows: "block.w",
      inner: "L",
      columns: "L",
      sourceElement: "a[group.z * BLOCK_PAIRS + row * L + k]",
      weightElement: "b[group.z * PAIRS + column * L + k]",
      store: "output[group.z * BLOCK_PAIRS + row * L + column] = element;",
    })
    : createTiledGemmShader({
      preamble: `${common}
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// x is the first pair row of the block, w the contraction rows it spans.
@group(0) @binding(3) var<uniform> block: vec4<u32>;`,
      rows: "L",
      inner: "block.w",
      columns: "L",
      sourceElement: "b[group.z * BLOCK_PAIRS + k * L + row]",
      weightElement: "a[group.z * BLOCK_PAIRS + k * L + column]",
      // z marks the first block, which overwrites rather than accumulating, so
      // the output needs no separate clear and no copy-destination usage.
      store: `let index = group.z * PAIRS + row * L + column;
          output[index] = select(output[index], 0.0, block.z == 1u) + element;`,
    });

  const contracted = direction === "outgoing"
    ? { stride: "BLOCK_PAIRS", offset: "" } : { stride: "PAIRS", offset: "block.x + " };
  const normalizeHidden = `${common}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<${t}>;
@group(0) @binding(2) var<storage, read_write> normalized: array<f32>;
// x is the first row of this block; the contraction is whole for the incoming
// direction and blocked for the outgoing one.
@group(0) @binding(3) var<uniform> block: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= block.y) { return; }
  let base = row * CH;
  var mean = 0.0;
  for (var h = 0u; h < CH; h += 1u) { mean += source[h * ${contracted.stride} + ${contracted.offset}row]; }
  mean /= f32(CH);
  var variance = 0.0;
  for (var h = 0u; h < CH; h += 1u) {
    let centered = source[h * ${contracted.stride} + ${contracted.offset}row] - mean;
    variance += centered * centered;
  }
  let inverse_std = inverseSqrt(variance / f32(CH) + EPSILON);
  for (var h = 0u; h < CH; h += 1u) {
    var value = (source[h * ${contracted.stride} + ${contracted.offset}row] - mean) * inverse_std;
    value = value * ${read(precision, "weights[W_LAYERNORMOUTWEIGHT + h]")}
      + ${read(precision, "weights[W_LAYERNORMOUTBIAS + h]")};
    normalized[base + h] = value;
  }
}`;

  // The output is the projected, gated hidden block; `output` is a view of the
  // rows this block produces.
  const projectOutput = createTiledGemmShader({
    preamble: `${common}
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${t}>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> block: vec4<u32>;`,
    rows: "block.y",
    inner: "CH",
    columns: "CZ",
    sourceElement: "x[row * CH + k]",
    weightElement: read(precision, "weights[W_LINEARZWEIGHT + column * CH + k]"),
    store: `let index = row * CZ + column;
          output[index] = (element + ${read(precision, "weights[W_LINEARZBIAS + column]")}) * gate[index];`,
  });

  const projectA = project("a", "BLOCK_PAIRS", "0u");
  const projectB = direction === "outgoing"
    ? project("b", "PAIRS", "block.x") : project("b", "BLOCK_PAIRS", "0u");
  return { inputStatistics, projectGate, projectA, projectB, contract, normalizeHidden, projectOutput };
}
