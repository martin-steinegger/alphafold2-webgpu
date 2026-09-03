import type { Precision, TriangleShape } from "./types.js";
import type { WeightOffsets } from "./weights.js";
import { createTiledGemmShader } from "../runtime/gemm.js";
import { type ActivationStorage, storageArray, storedElement } from "../runtime/storage.js";
import {
  planShards, shardBindings, shardLoader, shardStorer, shardWordLoader, type ShardLayout,
} from "../runtime/sharded.js";

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
   * pair. `a` is projected one output block at a time; `b` is filled block
   * by block into a whole tensor before any block contracts.
   */
  readonly projectBlockOperand: string;
  readonly projectWholeOperand: string;
  readonly contract: string;
  /** Mean and inverse standard deviation of each contracted row over the hidden channels. */
  readonly hiddenStatistics: string;
  /** The gated output projection, normalizing the contraction while loading it. */
  readonly projectOutput: string;
}

export type TriangleDirection = "outgoing" | "incoming";

/**
 * Storage of the whole projection. `f16` packs two half-precision values per
 * 32-bit word with `pack2x16float`, needing no device feature; it halves the
 * largest scratch tensor of the trunk and rounds the contraction inputs to
 * about three significant digits, so it is not exact.
 */
export type TriangleWholeStorage = "f32" | "f16";

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
  const pairs = shape.length * shape.length;
  return `${declaration(precision)}
const L: u32 = ${shape.length}u;
const CZ: u32 = ${shape.cZ}u;
const CH: u32 = ${shape.cHidden}u;
const PAIRS: u32 = L * L;
// Channel stride of the whole projection, padded so a packed pair of values
// never spans two channels.
const WHOLE_STRIDE: u32 = ${pairs + (pairs % 2)}u;
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
  wholeStorage: TriangleWholeStorage = "f32",
  pairStorage: ActivationStorage = "f32",
  residualOutput = false,
  /** How the pair is spread over bindings; one shard is the whole tensor. */
  pairShards: ShardLayout = planShards(shape.length * shape.length * shape.cZ, shape.cZ,
    Number.MAX_SAFE_INTEGER, 4),
): TriangleShaders {
  if (pairStorage === "f16" && precision !== "f32") {
    throw new RangeError("a packed pair needs f32 weight precision: both would claim the same halves of a word");
  }
  const common = prelude(shape, precision, offsets, epsilon, blockRows);
  const t = scalar(precision);
  // The pair may be stored packed, whatever precision the weights are in, and
  // it may be too large for one binding, in which case it arrives as several.
  const pairBindings = (name: string, first: number, writable: boolean): string =>
    shardBindings(pairShards, name, pairStorage === "f16" ? "f16" : "f32", first, writable);
  const pairAccessors = (name: string): string =>
    shardLoader(pairShards, name, pairStorage === "f16" ? "f16" : "f32");
  const pairSlots = pairShards.count;
  const pairElement = (index: string): string => `z_load(${index})`;
  const outgoing = direction === "outgoing";
  // Outgoing contracts over the second residue index and blocks the output by
  // rows i: its block operand a holds pair rows (i, k) of the block. Incoming
  // contracts over the first index and blocks the output by columns j: its
  // block operand a holds pairs (k, j) with j in the block, entry r being
  // (r / count, offset + r % count). Either way a finished block only
  // overwrites pair entries no later block reads, so the output can be the
  // pair itself, and the whole operand b is a plain projection in pair-row
  // order.
  const blockPairRow = outgoing
    ? "block.x + row" : "(row / block.w) * L + block.x / L + row % block.w";

  // One workgroup per pair row, so the channel reads of a row are contiguous
  // across lanes. The statistics let every later consumer normalize the raw
  // pair on the fly instead of materializing a normalized copy.
  const inputStatistics = `${common}
${pairBindings("source", 0, false)}
@group(0) @binding(${pairSlots}) var<storage, read_write> statistics: array<f32>;
${shardLoader(pairShards, "source", pairStorage === "f16" ? "f16" : "f32")}
var<workgroup> partial: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) local: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>) {
  let row = group.x + group.y * LINEAR_GRID_WIDTH;
  let base = row * CZ;
  var sum = 0.0;
  if (row < PAIRS) {
    for (var c = local.x; c < CZ; c += 64u) { sum += source_load(base + c); }
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
      let centered = source_load(base + c) - mean;
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
${pairBindings("z", 0, false)}
@group(0) @binding(${pairSlots}) var<storage, read> weights: array<${t}>;
@group(0) @binding(${pairSlots + 1}) var<storage, read> statistics: array<f32>;
@group(0) @binding(${pairSlots + 2}) var<storage, read_write> gate: array<f32>;
// x is the first row of this block within the whole pair tensor, y the
// number of rows it spans.
@group(0) @binding(${pairSlots + 3}) var<uniform> block: vec4<u32>;
${pairAccessors("z")}

fn pair_row_of(row: u32) -> u32 { return ${blockPairRow}; }

fn normalized_input(pair_row: u32, k: u32) -> f32 {
  return (${pairElement("pair_row * CZ + k")} - statistics[2u * pair_row]) * statistics[2u * pair_row + 1u]
    * ${read(precision, "weights[W_LAYERNORMINWEIGHT + k]")} + ${read(precision, "weights[W_LAYERNORMINBIAS + k]")};
}`,
    rows: "block.y",
    inner: "CZ",
    columns: "CZ",
    sourceElement: "normalized_input(pair_row_of(row), k)",
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
  const project = (
    operand: "a" | "b", stride: string, pairRow: string, storeRow: string, packed: boolean,
  ): string => {
    const upper = operand.toUpperCase();
    const weight = (kind: "P" | "G"): string =>
      read(precision, `weights[W_LINEAR${upper}${kind}WEIGHT + (column >> 1u) * CZ + k]`);
    const bias = (kind: "P" | "G", channel: string): string =>
      read(precision, `weights[W_LINEAR${upper}${kind}BIAS + ${channel}]`);
    return createTiledGemmShader({
      preamble: `${common}
${pairBindings("z", 0, false)}
@group(0) @binding(${pairSlots}) var<storage, read> mask: array<f32>;
@group(0) @binding(${pairSlots + 1}) var<storage, read> weights: array<${t}>;
@group(0) @binding(${pairSlots + 2}) var<storage, read> statistics: array<f32>;
@group(0) @binding(${pairSlots + 3}) var<storage, read_write> ${operand}: array<${packed ? "u32" : "f32"}>;
// x is the first pair row of this block, y the number of pair rows it spans,
// w the residues it covers.
@group(0) @binding(${pairSlots + 4}) var<uniform> block: vec4<u32>;
${pairAccessors("z")}

fn pair_row_of(row: u32) -> u32 { return ${pairRow}; }

fn normalized_input(pair_row: u32, k: u32) -> f32 {
  return (${pairElement("pair_row * CZ + k")} - statistics[2u * pair_row]) * statistics[2u * pair_row + 1u]
    * ${read(precision, "weights[W_LAYERNORMINWEIGHT + k]")} + ${read(precision, "weights[W_LAYERNORMINBIAS + k]")};
}`,
      rows: "block.y",
      inner: "CZ",
      columns: "2u * CH",
      sourceElement: "normalized_input(pair_row_of(row), k)",
      weightElement: `select(${weight("G")}, ${weight("P")}, (column & 1u) == 0u)`,
      store: "",
      // The contraction reads the projection channel-major, so a direct store
      // from the row-major tile would scatter every write across the whole
      // tensor. Each invocation drops its channel/gate pairs into a staged
      // transpose instead, thirty-two channels at a time so the staging stays
      // inside the portable workgroup storage, and the tile is then written out
      // with adjacent lanes on adjacent pair rows.
      stageElements: 32 * 64,
      epilogue: `
  for (var half = 0u; half < 2u; half += 1u) {
    if (column_thread >= half * 16u && column_thread < half * 16u + 16u) {
      let h_local = (column_thread - half * 16u) * 2u;
      let h = column_origin / 2u + half * 32u + h_local;
      let bias_p0 = ${bias("P", "h")}; let bias_g0 = ${bias("G", "h")};
      let bias_p1 = ${bias("P", "h + 1u")}; let bias_g1 = ${bias("G", "h + 1u")};
${Array.from({ length: 8 }, (_, index) => `      {
        let r_local = row_thread * 8u + ${index}u;
        let row = tile_row_origin + r_local;
        var pair_mask = 0.0;
        if (row < gemm_rows) { pair_mask = mask[pair_row_of(row)]; }
        gemm_stage[h_local * 64u + r_local] = pair_mask * (acc${index}[0] + bias_p0) * logistic(acc${index}[1] + bias_g0);
        gemm_stage[(h_local + 1u) * 64u + r_local] = pair_mask * (acc${index}[2] + bias_p1) * logistic(acc${index}[3] + bias_g1);
      }`).join("\n")}
    }
    workgroupBarrier();
${packed ? `    // Two consecutive pair rows share a word. Blocks start on even pair rows
    // and the channel stride is even, so a pair never spans two words.
    for (var item = 0u; item < 4u; item += 1u) {
      let element = (thread + item * 256u) * 2u;
      let h_local = element / 64u;
      let r_local = element % 64u;
      let row = tile_row_origin + r_local;
      let h = column_origin / 2u + half * 32u + h_local;
      if (row < gemm_rows && h < CH) {
        let second = select(0.0, gemm_stage[element + 1u], row + 1u < gemm_rows);
        ${operand}[(h * ${stride} + ${storeRow}) >> 1u] = pack2x16float(vec2<f32>(gemm_stage[element], second));
      }
    }` : `    for (var item = 0u; item < 8u; item += 1u) {
      let element = thread + item * 256u;
      let h_local = element / 64u;
      let r_local = element % 64u;
      let row = tile_row_origin + r_local;
      let h = column_origin / 2u + half * 32u + h_local;
      if (row < gemm_rows && h < CH) { ${operand}[h * ${stride} + ${storeRow}] = gemm_stage[element]; }
    }`}
    workgroupBarrier();
  }`,
    });
  };

  // One independent matrix per hidden channel, dispatched along z. Outgoing:
  // out[i][j] = sum_k a[i][k] b[j][k] over a block of rows i. Incoming:
  // out[i][j] = sum_k b[k][i] a[k][j] over a block of columns j, with the
  // GEMM rows being the block's columns and its columns every i.
  const packedWhole = wholeStorage === "f16";
  const wholeElement = (index: string): string => packedWhole
    ? `unpack2x16float(whole[(${index}) >> 1u])[(${index}) & 1u]` : `whole[${index}]`;
  const contract = createTiledGemmShader({
    preamble: `${common}
@group(0) @binding(0) var<storage, read> blocked: array<f32>;
@group(0) @binding(1) var<storage, read> whole: array<${packedWhole ? "u32" : "f32"}>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// x is the first pair row of the block, w the residue count it spans.
@group(0) @binding(3) var<uniform> block: vec4<u32>;`,
    rows: "block.w",
    inner: "L",
    columns: "L",
    sourceElement: outgoing
      ? "blocked[group.z * BLOCK_PAIRS + row * L + k]" : "blocked[group.z * BLOCK_PAIRS + k * block.w + row]",
    weightElement: outgoing
      ? wholeElement("group.z * WHOLE_STRIDE + column * L + k") : wholeElement("group.z * WHOLE_STRIDE + k * L + column"),
    // The block's output entries are enumerated like its operand: by pair row
    // (i, j) outgoing, by (i, block column j) incoming.
    store: outgoing
      ? "output[group.z * BLOCK_PAIRS + row * L + column] = element;"
      : "output[group.z * BLOCK_PAIRS + column * block.w + row] = element;",
  });

  const contracted = { stride: "BLOCK_PAIRS", offset: "" };
  // One invocation per contracted row: adjacent invocations read adjacent
  // addresses of the channel-major contraction.
  const hiddenStatistics = `${common}
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> statistics: array<f32>;
// x is the first row of this block; the contraction is whole for the incoming
// direction and blocked for the outgoing one.
@group(0) @binding(2) var<uniform> block: vec4<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  if (row >= block.y) { return; }
  var mean = 0.0;
  for (var h = 0u; h < CH; h += 1u) { mean += source[h * ${contracted.stride} + ${contracted.offset}row]; }
  mean /= f32(CH);
  var variance = 0.0;
  for (var h = 0u; h < CH; h += 1u) {
    let centered = source[h * ${contracted.stride} + ${contracted.offset}row] - mean;
    variance += centered * centered;
  }
  statistics[2u * row] = mean;
  statistics[2u * row + 1u] = inverseSqrt(variance / f32(CH) + EPSILON);
}`;

  // The output is the projected, gated hidden block, written at the pair rows
  // the block covers; `output` is the whole pair-shaped tensor.
  const projectOutput = createTiledGemmShader({
    preamble: `${common}
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> contracted: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<${t}>;
@group(0) @binding(3) var<storage, read> statistics: array<f32>;
${shardBindings(pairShards, "output", pairStorage, 4, true)}
@group(0) @binding(${4 + pairSlots}) var<uniform> block: vec4<u32>;
${shardStorer(pairShards, "output", pairStorage)}
${pairStorage === "f16" ? shardWordLoader(pairShards, "output")
  : (residualOutput ? shardLoader(pairShards, "output", pairStorage) : "")}

fn pair_row_of(row: u32) -> u32 { return ${blockPairRow}; }

fn normalized_hidden(row: u32, h: u32) -> f32 {
  return (contracted[h * ${contracted.stride} + ${contracted.offset}row] - statistics[2u * row]) * statistics[2u * row + 1u]
    * ${read(precision, "weights[W_LAYERNORMOUTWEIGHT + h]")} + ${read(precision, "weights[W_LAYERNORMOUTBIAS + h]")};
}`,
    rows: "block.y",
    inner: "CH",
    columns: "CZ",
    sourceElement: "normalized_hidden(row, k)",
    weightElement: read(precision, "weights[W_LINEARZWEIGHT + column * CH + k]"),
    // A packed pair is written a word at a time, so the four adjacent columns
    // an invocation holds become two words; an unpacked one keeps the scalar
    // store, which callers may still rewrite into a residual add.
    ...(pairStorage === "f16" ? {
      storeVector: `let index = pair_row_of(row) * CZ + column;
      let gates = vec4<f32>(gate[row * CZ + column], gate[row * CZ + column + 1u],
        gate[row * CZ + column + 2u], gate[row * CZ + column + 3u]);
      let biases = vec4<f32>(weights[W_LINEARZBIAS + column], weights[W_LINEARZBIAS + column + 1u],
        weights[W_LINEARZBIAS + column + 2u], weights[W_LINEARZBIAS + column + 3u]);
      var stored = (values + biases) * gates;
      let word = index >> 1u;
      ${residualOutput
        ? `stored += vec4<f32>(unpack2x16float(output_load_word(word)), unpack2x16float(output_load_word(word + 1u)));`
        : ""}
      output_store(word, pack2x16float(stored.xy));
      output_store(word + 1u, pack2x16float(stored.zw));`,
      store: "",
    } : {
      store: `let index = pair_row_of(row) * CZ + column;
          let written = (element + ${read(precision, "weights[W_LINEARZBIAS + column]")}) * gate[row * CZ + column];
          output_store(index, ${residualOutput ? "output_load(index) + written" : "written"});`,
    }),
  });

  // The block operand is stored block-relative, the whole operand at its pair row.
  // Outgoing contracts a's rows i against b's rows j; incoming contracts a's
  // columns j against b's columns i. In both, a is the block operand.
  const projectBlockOperand = project("a", "BLOCK_PAIRS", blockPairRow, "row", false);
  const projectWholeOperand = project("b", "WHOLE_STRIDE", "block.x + row", "block.x + row", packedWhole);
  return {
    inputStatistics, projectGate, projectBlockOperand, projectWholeOperand, contract, hiddenStatistics, projectOutput,
  };
}
