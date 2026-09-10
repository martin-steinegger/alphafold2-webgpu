import type { Precision, TriangleShape } from "./types.js";
import type { WeightOffsets } from "./weights.js";
import { createTiledGemmShader, GEMM_TILE_COLUMNS, GEMM_TILE_ROWS } from "../runtime/gemm.js";
import { type ActivationStorage, storageArray, storedElement } from "../runtime/storage.js";
import {
  planShards, shardBindings, shardLoader, shardStorer, shardWordLoader, type ShardLayout,
} from "../runtime/sharded.js";
import type { MatrixSpelling } from "../runtime/dialect.js";

/** The whole operand in a single binding, which is the common case. */
const WHOLE_OPERAND_UNSHARDED: ShardLayout = {
  count: 1, shardElements: Number.MAX_SAFE_INTEGER, totalElements: 0,
};

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
   * pair. a is projected one output block at a time; b is filled block
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
 * Storage of the whole projection. f16 packs two half-precision values per
 * 32-bit word with pack2x16float, needing no device feature; it halves the
 * largest scratch tensor of the trunk and rounds the contraction inputs to
 * about three significant digits, so it is not exact.
 */
export type TriangleWholeStorage = "f32" | "f16";

/**
 * Tile rows the projection epilogue below is written for.
 *
 * It names eight accumulators an invocation and stages its transpose 64 rows
 * wide, both of which follow from a 64-row tile rather than being read off it.
 * A different tile row count leaves it addressing the wrong accumulators and
 * the wrong slots, and nothing says so: the model still runs, and comes back
 * at 70 pLDDT where it gave 88. So the coupling is checked rather than
 * commented on.
 */
const ROWS_THE_EPILOGUE_IS_WRITTEN_FOR = 64;
/**
 * Tile columns it is written for, for the same reason.
 *
 * It splits the tile into two halves of sixteen column threads and stages
 * thirty-two channels of each, which follows from 128 columns over four-wide
 * invocations. A 256-column tile silently returned 70 pLDDT for it.
 */
const COLUMNS_THE_EPILOGUE_IS_WRITTEN_FOR = 128;
if (GEMM_TILE_ROWS !== ROWS_THE_EPILOGUE_IS_WRITTEN_FOR
  || GEMM_TILE_COLUMNS !== COLUMNS_THE_EPILOGUE_IS_WRITTEN_FOR) {
  throw new RangeError(`the triangle projection epilogue is written for a `
    + `${ROWS_THE_EPILOGUE_IS_WRITTEN_FOR}x${COLUMNS_THE_EPILOGUE_IS_WRITTEN_FOR} GEMM tile, `
    + `and this build tiles ${GEMM_TILE_ROWS}x${GEMM_TILE_COLUMNS}`);
}

const declaration = (precision: Precision): string => precision === "f16" ? "enable f16;\n" : "";
const scalar = (precision: Precision): "f16" | "f32" => precision;
const read = (precision: Precision, expression: string): string =>
  precision === "f16" ? `f32(${expression})` : expression;

/**
 * @param blockRows Rows of the blocked residue axis one step covers. PAIRS
 * stays the whole pair count; BLOCK_PAIRS is one step's worth.
 */
/**
 * The length-dependent constants, supplied when a pipeline is created.
 *
 * They are overrides rather than literals so that one shader module serves
 * every sequence length: without this each length compiled its own copy of all
 * seven triangle kernels, 14 modules at about 16 ms each. They still fold --
 * L is the k bound of the contraction's loop and a runtime bound there
 * measured 4.7x, while an override measured 0.129 ms against a literal's 0.129
 * on that kernel.
 */
export function triangleOverrides(
  shape: TriangleShape, blockRows = shape.length,
): Record<string, number> {
  return { L: shape.length, WHOLE_STRIDE: wholeProjectionStride(shape.length), BLOCK_ROWS: blockRows };
}

/**
 * Elements one hidden channel of the whole projection occupies.
 *
 * The pair count itself would do, but the contraction is dispatched a channel
 * group at a time and each group is bound at its own offset, which WebGPU
 * requires to be a multiple of 256 bytes. Padding a channel to 128 elements
 * makes every channel boundary such an offset in both storages: 128 halves are
 * 256 bytes and 128 words are 512. The padding is at most 127 elements a
 * channel, which is 32 KiB over the whole tensor at any length.
 */
export function wholeProjectionStride(length: number): number {
  const pairs = length * length;
  return pairs + (WHOLE_CHANNEL_ALIGNMENT - pairs % WHOLE_CHANNEL_ALIGNMENT) % WHOLE_CHANNEL_ALIGNMENT;
}

/** Elements a channel is padded to, which is 256 bytes of the narrower storage. */
export const WHOLE_CHANNEL_ALIGNMENT = 128;

function prelude(
  shape: TriangleShape, precision: Precision, offsets: WeightOffsets, epsilon: number,
): string {
  const offsetConstants = Object.entries(offsets)
    .map(([name, offset]) => `const W_${name.toUpperCase()}: u32 = ${offset}u;`)
    .join("\n");
  return `${declaration(precision)}
// Length-dependent, and supplied at pipeline creation. See triangleOverrides.
override L: u32 = 1u;
const CZ: u32 = ${shape.cZ}u;
const CH: u32 = ${shape.cHidden}u;
override PAIRS: u32 = L * L;
// Channel stride of the whole projection, padded so a packed pair of values
// never spans two channels.
override WHOLE_STRIDE: u32 = 1u;
override BLOCK_ROWS: u32 = 1u;
override BLOCK_PAIRS: u32 = BLOCK_ROWS * L;
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
  /** How the whole operand is spread over bindings; it outgrows one first. */
  wholeShards: ShardLayout = planShards(
    (shape.length * shape.length + (shape.length * shape.length) % 2) * shape.cHidden, 2,
    Number.MAX_SAFE_INTEGER, 4),

  /**
   * Whether the pair arrives as the block's own rows in one binding.
   *
   * Only where those rows are contiguous, which the caller decides: the
   * outgoing direction and, in both directions, the whole operand.
   */
  pairWindow = false,

  spelling?: MatrixSpelling,
): TriangleShaders {
  if (pairStorage === "f16" && precision !== "f32") {
    throw new RangeError("a packed pair needs f32 weight precision: both would claim the same halves of a word");
  }
  const common = prelude(shape, precision, offsets, epsilon);
  const t = scalar(precision);
  // The pair may be stored packed, whatever precision the weights are in, and
  // it may be too large for one binding, in which case it arrives as several.
  const pairElementStorage: ActivationStorage = pairStorage === "f16" ? "f16" : "f32";
  // A kernel that reads only the pair rows of its block can be given those
  // rows as one binding, which the shard chain is not. That is worth 2.4x on
  // the kernel: a chain puts a branch between the loop and the array, and the
  // compiler then cannot form the wide contiguous loads a staged matrix tile
  // wants. Removing the divide the chain also does changed nothing, measured,
  // so it is the branch. Only kernels whose pair rows are block.x + row may
  // ask; the incoming direction reads a column window of every row and has to
  // take the chain. See pairWindow in block.ts for the binding.
  const pairLayout = (windowed: boolean): ShardLayout =>
    windowed ? WHOLE_OPERAND_UNSHARDED : pairShards;
  const pairBindings = (name: string, first: number, writable: boolean, windowed = false): string =>
    shardBindings(pairLayout(windowed), name, pairElementStorage, first, writable);
  const pairAccessors = (name: string, windowed = false): string =>
    shardLoader(pairLayout(windowed), name, pairElementStorage);
  const pairSlotsOf = (windowed: boolean): number => windowed ? 1 : pairShards.count;
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
  // Only the outgoing direction reads its block as a run of pair rows.
  const gateWindowed = pairWindow && outgoing;
  const gateSlots = pairSlotsOf(gateWindowed);
  const projectGate = createTiledGemmShader({
    preamble: `${common}
${pairBindings("z", 0, false, gateWindowed)}
@group(0) @binding(${gateSlots}) var<storage, read> weights: array<${t}>;
@group(0) @binding(${gateSlots + 1}) var<storage, read> statistics: array<f32>;
@group(0) @binding(${gateSlots + 2}) var<storage, read_write> gate: array<f32>;
// x is the first row of this block within the whole pair tensor, y the
// number of rows it spans.
@group(0) @binding(${gateSlots + 3}) var<uniform> block: vec4<u32>;
${pairAccessors("z", gateWindowed)}

fn pair_row_of(row: u32) -> u32 { return ${blockPairRow}; }
// Where the row starts in the binding, which is the block for a window.
fn pair_base_of(row: u32) -> u32 { return ${gateWindowed ? "row" : "pair_row_of(row)"} * CZ; }

fn normalized_input(row: u32, k: u32) -> f32 {
  let pair_row = pair_row_of(row);
  return (${pairElement("pair_base_of(row) + k")} - statistics[2u * pair_row]) * statistics[2u * pair_row + 1u]
    * ${read(precision, "weights[W_LAYERNORMINWEIGHT + k]")} + ${read(precision, "weights[W_LAYERNORMINBIAS + k]")};
}`,
    rows: "block.y",
    inner: "CZ",
    columns: "CZ",
    sourceElement: "normalized_input(row, k)",
    // Every weight here is stored channel-major, so k runs contiguously and a
    // lane staging the tile should vary k rather than column. The default is
    // the other layout, and reading it that way costs bandwidth silently.
    weightContiguous: "k",
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
    shards: ShardLayout = WHOLE_OPERAND_UNSHARDED, windowed = false,
  ): string => {
    const pairSlots = pairSlotsOf(windowed);
    const upper = operand.toUpperCase();
    const wholeStorage: ActivationStorage = packed ? "f16" : "f32";
    const weight = (kind: "P" | "G"): string =>
      read(precision, `weights[W_LINEAR${upper}${kind}WEIGHT + (column >> 1u) * CZ + k]`);
    const bias = (kind: "P" | "G", channel: string): string =>
      read(precision, `weights[W_LINEAR${upper}${kind}BIAS + ${channel}]`);
    return createTiledGemmShader({
      preamble: `${common}
${pairBindings("z", 0, false, windowed)}
@group(0) @binding(${pairSlots}) var<storage, read> mask: array<f32>;
@group(0) @binding(${pairSlots + 1}) var<storage, read> weights: array<${t}>;
@group(0) @binding(${pairSlots + 2}) var<storage, read> statistics: array<f32>;
${shardBindings(shards, operand, wholeStorage, pairSlots + 3, true)}
// x is the first pair row of this block, y the number of pair rows it spans,
// w the residues it covers.
@group(0) @binding(${pairSlots + 3 + shards.count}) var<uniform> block: vec4<u32>;
${pairAccessors("z", windowed)}
${shardStorer(shards, operand, wholeStorage)}

fn pair_row_of(row: u32) -> u32 { return ${pairRow}; }
// Where the row starts in the binding, which is the block for a window.
fn pair_base_of(row: u32) -> u32 { return ${windowed ? "row" : "pair_row_of(row)"} * CZ; }

fn normalized_input(row: u32, k: u32) -> f32 {
  let pair_row = pair_row_of(row);
  return (${pairElement("pair_base_of(row) + k")} - statistics[2u * pair_row]) * statistics[2u * pair_row + 1u]
    * ${read(precision, "weights[W_LAYERNORMINWEIGHT + k]")} + ${read(precision, "weights[W_LAYERNORMINBIAS + k]")};
}`,
      rows: "block.y",
      inner: "CZ",
      columns: "2u * CH",
      sourceElement: "normalized_input(row, k)",
      // Channel-major, as the gate and the output projection are.
      weightContiguous: "k",
      weightElement: `select(${weight("G")}, ${weight("P")}, (column & 1u) == 0u)`,
      store: "",
      // What the epilogue below does, for the matrix kernel, which has no
      // acc{n} to write it against. Its accumulator arrives transposed, so
      // this tile is already channel-major and a lane walking local_row walks
      // consecutive pair rows of one channel, which is what the store wants.
      // The transpose costs nothing: the instruction does it while the result
      // is still spread over the lanes.
      matrixEpilogue: `
    let pairs = tile_columns >> 1u;
    let rows_each = tile_rows${packed ? " >> 1u" : ""};
    for (var item = in_subgroup; item < pairs * rows_each; item += tile_lanes) {
      let h = (tile_column_first >> 1u) + item / rows_each;
      let r = (item % rows_each)${packed ? " * 2u" : ""};
      let row = tile_row_first + r;
      if (row >= gemm_rows || h >= CH) { continue; }
      let at = tile_base + ((item / rows_each) << 1u) * tile_stride + r;
      let bias_p = ${bias("P", "h")};
      let bias_g = ${bias("G", "h")};
      let gated = mask[pair_row_of(row)]
        * (gemm_matrix_out[at] + bias_p) * logistic(gemm_matrix_out[at + tile_stride] + bias_g);
${packed ? `      // Two consecutive pair rows share a word, and a tile starts on an even
      // one, so this lane owns both halves and no neighbour has to be asked.
      var second = 0.0;
      if (row + 1u < gemm_rows) {
        let next_row = row + 1u;
        second = mask[pair_row_of(next_row)]
          * (gemm_matrix_out[at + 1u] + bias_p)
          * logistic(gemm_matrix_out[at + tile_stride + 1u] + bias_g);
      }
      ${operand}_store((h * ${stride} + ${storeRow}) >> 1u,
        pack2x16float(vec2<f32>(gated, second)));` : `      ${operand}_store(h * ${stride} + ${storeRow}, gated);`}
    }`,
      // The contraction reads the projection channel-major, so a direct store
      // from the row-major tile would scatter every write across the whole
      // tensor. Each invocation drops its channel/gate pairs into a staged
      // transpose instead, thirty-two channels at a time so the staging stays
      // inside the portable workgroup storage, and the tile is then written out
      // with adjacent lanes on adjacent pair rows.
      stageElements: 32 * ROWS_THE_EPILOGUE_IS_WRITTEN_FOR,
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
        ${operand}_store((h * ${stride} + ${storeRow}) >> 1u, pack2x16float(vec2<f32>(gemm_stage[element], second)));
      }
    }` : `    for (var item = 0u; item < 8u; item += 1u) {
      let element = thread + item * 256u;
      let h_local = element / 64u;
      let r_local = element % 64u;
      let row = tile_row_origin + r_local;
      let h = column_origin / 2u + half * 32u + h_local;
      if (row < gemm_rows && h < CH) { ${operand}_store(h * ${stride} + ${storeRow}, gemm_stage[element]); }
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
  const wholeElement = (index: string): string => `whole_load(${index})`;
  // The projection reaches this kernel as one binding however large it is,
  // because all three of its operands are channel-major and the dispatch
  // covers a group of channels that fits. A shard chain here would put an
  // integer divide and a branch in the inner loop of the hottest kernel of the
  // trunk, and stop the operand being staged as a contiguous tile with it: at
  // 1,650 residues that measured 2.5x on this kernel alone. See
  // wholeProjectionStride, and encodeTriangleMultiplication for the grouping.
  // The channel the dispatch starts at, which the two operands bound whole
  // still need; the projection is bound at the group, so its own index is
  // local.
  const channel = "(channels.x + group.z)";
  const contract = createTiledGemmShader({
    preamble: `${common}
@group(0) @binding(0) var<storage, read> blocked: array<f32>;
${shardBindings(WHOLE_OPERAND_UNSHARDED, "whole", packedWhole ? "f16" : "f32", 1, false)}
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
// x is the first pair row of the block, w the residue count it spans.
@group(0) @binding(3) var<uniform> block: vec4<u32>;
// x is the first hidden channel of the group this dispatch covers.
@group(0) @binding(4) var<uniform> channels: vec4<u32>;
${shardLoader(WHOLE_OPERAND_UNSHARDED, "whole", packedWhole ? "f16" : "f32")}`,
    rows: "block.w",
    inner: "L",
    columns: "L",
    sourceElement: outgoing
      ? `blocked[${channel} * BLOCK_PAIRS + row * L + k]` : `blocked[${channel} * BLOCK_PAIRS + k * block.w + row]`,
    weightElement: outgoing
      ? wholeElement("group.z * WHOLE_STRIDE + column * L + k") : wholeElement("group.z * WHOLE_STRIDE + k * L + column"),
    // Which index of each operand is contiguous, so a staged tile is fetched
    // along memory rather than across it.
    sourceContiguous: outgoing ? "k" : "row",
    weightContiguous: outgoing ? "k" : "column",
    // The block's output entries are enumerated like its operand: by pair row
    // (i, j) outgoing, by (i, block column j) incoming.
    store: outgoing
      ? `output[${channel} * BLOCK_PAIRS + row * L + column] = element;`
      : `output[${channel} * BLOCK_PAIRS + column * block.w + row] = element;`,
  }, undefined, spelling);

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
  // the block covers; output is the whole pair-shaped tensor.
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
    // Every weight here is stored channel-major, so k runs contiguously and a
    // lane staging the tile should vary k rather than column. The default is
    // the other layout, and reading it that way costs bandwidth silently.
    weightContiguous: "k",
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
  const projectBlockOperand = project("a", "BLOCK_PAIRS", blockPairRow, "row", false,
    WHOLE_OPERAND_UNSHARDED, pairWindow && outgoing);
  // The whole operand is written at its pair row in both directions, so it
  // reads the block's own rows either way and always takes the window.
  const projectWholeOperand = project("b", "WHOLE_STRIDE", "block.x + row", "block.x + row", packedWhole,
    wholeShards, pairWindow);
  return {
    inputStatistics, projectGate, projectBlockOperand, projectWholeOperand, contract, hiddenStatistics, projectOutput,
  };
}
