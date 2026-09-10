/**
 * Flash attention over the hardware matrix units.
 *
 * The register-resident kernel this competes with keeps one query's whole
 * accumulator in registers and reads every key and value itself, which costs
 * no cross-lane traffic and is why it beats every subgroup variant here: those
 * reduce each query-key dot product with subgroupAdd, once per key. A matrix
 * unit does that reduction in hardware, so this kernel pays neither the
 * cross-lane traffic nor the per-key reduction, and reads each staged key tile
 * once for sixteen queries instead of once per query.
 *
 * The shape is the one both reference implementations use: 16x16x16 tiles with
 * f16 operands and an f32 accumulator, which is what Nvidia's units implement
 * and what subgroupMatrixConfigs reports. Operands must live in workgroup or
 * storage memory as the component type, so the query, key and value tiles are
 * converted into workgroup f16 once per tile rather than per multiply.
 *
 * The online softmax cannot rescale an accumulator in place: its correction is
 * per query row and subgroupMatrixScalarMultiply takes one uniform scalar.
 * So P V accumulates into a freshly zeroed result each key tile, is stored to
 * workgroup memory, and the running output is rescaled there in plain f32 —
 * which is exactly what the reference CUDA kernel does for the same reason.
 */
import { attentionMatrixConfig, type MatrixUnitShape } from "../runtime/gemm.js";
import type { MatrixSpelling } from "../runtime/dialect.js";

/**
 * Subgroups per workgroup, each owning sixteen queries.
 *
 * Two, and the reason is occupancy rather than reuse. Every arrangement of
 * this kernel ranks by how much workgroup storage it declares, monotonically:
 * 6.3 KiB reads 1.66x, 10.3 KiB 1.84x, 18.2 KiB 1.77x, 22.5 KiB 1.63x and
 * 40.4 KiB 1.39x. Wider tiles reuse a staged key across more queries and lose
 * anyway, because what a multiprocessor can hold matters more here than what a
 * workgroup can reuse. One subgroup is past the other side of it: too little
 * work in flight to cover the latency of its own staging.
 *
 * Measured against the stack rather than a standalone dispatch, four and eight
 * subgroups still lose: the flash dispatches of an 800-residue stack read
 * 51.0 ms at two, 54.1 ms at four and 62.9 ms at eight. A standalone
 * dispatch ranks them the other way round, and is the wrong instrument here.
 */
/** Lanes a subgroup, which every index and shuffle in this kernel assumes. */
export const ATTENTION_MATRIX_SUBGROUP_SIZE = 32;
const MATRIX_LANES = ATTENTION_MATRIX_SUBGROUP_SIZE;

/**
 * Subgroups a workgroup, which is also how many queries share one staged key
 * tile.
 *
 * Four rather than two, swept whole-fold on an RTX PRO 6000 with the card
 * pinned and the arms alternated, because this machine drifts by more between
 * repeats than the difference being measured. Milliseconds a recycle at 825
 * residues over three pairs: 4313/4414/4611 at two subgroups against
 * 4229/4247/4460 at four, so four wins every pair by about 3%. At 512 it is
 * 1847 against 1745 and at 256 it is 1041 against 1047, a wash. pLDDT is
 * 86.76, 86.93 and 84.25 at those three lengths whichever is used.
 *
 * The key tile stays 32: 64 measured 1.067 ms against 0.954 for the isolated
 * attention at two subgroups and 1.056 against 0.936 at four.
 */
const SUBGROUPS = 4;
/** The M of every tile, fixed by the unit shape the device reports. */
const UNIT = 16;
/** Queries one workgroup owns. */
export const ATTENTION_MATRIX_QUERY_TILE = SUBGROUPS * UNIT;
/**
 * Keys staged per pass, shared by every subgroup.
 *
 * Two units wide. One unit is too narrow: the pass then stages a key tile and
 * runs the whole online softmax over it for four multiplies, and the staging
 * and the barriers around it cost more than the multiplies. Two units halve
 * that overhead per multiply and still fit the storage a multiprocessor wants
 * back, and the stack's flash dispatches fall from 51.0 ms to 47.5 ms over two
 * runs at 800 residues, and the main stack from 25.07 s to 24.42 s at 1,650.
 * Three and four units cross the occupancy cliff and read 69.6 ms and 70.4 ms.
 *
 * The earlier note here said the opposite, on a microbenchmark whose key and
 * value tensors were small enough to sit in L2. The model's are not: at 1,650
 * residues triangle attention alone streams 1.4 GB of each. Measure this
 * against the stack, not against a standalone dispatch.
 */
const KEY_TILE = 32;
const LANES = SUBGROUPS * 32;

/**
 * Padding on the staged score and probability rows.
 *
 * Both are indexed by lane and by matrix stride at once, and a power-of-two
 * row length puts every row in the same bank. The reference kernel pads the
 * f32 rows by four and the f16 rows by eight, which is one bank of each.
 */
/**
 * Row lengths of everything the units address, chosen against the banks.
 *
 * Workgroup memory is thirty-two banks of four bytes, so lane i reading
 * element i * stride lands in bank (i * stride) % 32 for f32 and
 * (i * stride / 2) % 32 for f16. A stride sharing a factor with thirty-two
 * collapses the lanes onto few banks and serialises the read; a stride coprime
 * with it spreads them across all thirty-two.
 *
 * The head width is thirty-two, which is the worst case of all: the staged key
 * tile is read column-major, so consecutive lanes are one row apart, and at a
 * stride of thirty-two f16 that is (16i) % 32 — two banks for thirty-two
 * lanes, a sixteen-way conflict on the hottest read in the kernel. Thirty-four
 * gives (17i) % 32, and seventeen is coprime with thirty-two, so no two
 * lanes collide. The f32 rows take the same treatment with an odd stride.
 */
/**
 * Head widths this kernel takes, and the tile width each one spans.
 *
 * A head narrower than the unit is padded up to it: the channels past the head
 * are never written, and workgroup memory starts at zero, so a contraction
 * over the padded width is the contraction over the head. It costs multiplies
 * that this kernel is not short of — the extra-MSA stack's eight-channel heads
 * read a key for eight channels of work, and wait for memory rather than for
 * the units.
 */
const paddedHeadDim = (headDim: number): number => Math.max(headDim, UNIT);

/**
 * Row lengths of the staged tiles, padded past what they hold.
 *
 * The padding is what the matrix instructions want, not what avoids bank
 * conflicts. A tile a matrix load reads wants a row length that is a multiple
 * of eight halves; one a matrix store writes wants a multiple of four words.
 * These were one and two before, which is the right padding for the banks and
 * the wrong alignment for the units: it cost the four flash kernels 21.2 ms of
 * a main block against 15.2, and a whole fold 11%.
 *
 * ColabFold's Volta kernels say the same in a comment, which is where this
 * came from: "SS_LD = BK + 4, f32 ldm: multiple of 4 for wmma stores" and
 * "PS_LD = BK + 8, f16 ldm: multiple of 8 for wmma loads".
 */
const tileStride = (headDim: number): number => paddedHeadDim(headDim) + 8;
const SCORE_STRIDE = KEY_TILE + 4;
const WEIGHTED_STRIDE = 36;
const PROBABILITY_STRIDE = KEY_TILE + 8;

/** Whether this device can run it, and with which unit shape. */
export function attentionMatrixShape(
  device: GPUDevice, headDim: number,
  configs: readonly { componentType: string; resultComponentType: string;
    M: number; N: number; K: number }[],
): MatrixUnitShape | undefined {
  if (headDim % 4 !== 0 || headDim > 32 || headDim < 4) return undefined;
  return attentionMatrixConfig(device, configs, UNIT);
}

/**
 * Workgroup bytes the kernel declares, which a device must permit.
 *
 * Every array is sized by the same offset + stride * rows reach the loads
 * and stores claim, not by the last element they touch, so this matches what
 * the shader actually declares.
 */
export function attentionMatrixStorageBytes(headDim: number, storedHalf = false): number {
  const rows = ATTENTION_MATRIX_QUERY_TILE;
  const unit = UNIT;
  const keyTiles = KEY_TILE / unit;
  const channelTiles = paddedHeadDim(headDim) / unit;
  const TILE_STRIDE = tileStride(headDim);
  const reach = (maxOffset: number, stride: number, count: number): number =>
    maxOffset + stride * count;
  const queries = reach((rows - unit) * TILE_STRIDE, TILE_STRIDE, unit);
  const keys = reach((keyTiles - 1) * unit * TILE_STRIDE, TILE_STRIDE, unit);
  const values = reach((keyTiles - 1) * unit * TILE_STRIDE + (channelTiles - 1) * unit,
    TILE_STRIDE, unit);
  const scores = Math.max(
    reach((rows - unit) * SCORE_STRIDE + (keyTiles - 1) * unit, SCORE_STRIDE, unit),
    reach((rows - unit) * WEIGHTED_STRIDE + (channelTiles - 1) * unit, WEIGHTED_STRIDE, unit));
  const probabilities = reach(
    (rows - unit) * PROBABILITY_STRIDE + (keyTiles - 1) * unit, PROBABILITY_STRIDE, unit);
  return KEY_TILE * 4                             // staged mask
    + queries * 2
    + (storedHalf ? 0 : (keys + values) * 2)
    + scores * 4
    + probabilities * 2
    + 3 * rows * 4;                               // max, sum, rescale
}

export function createAttentionMatrixFlashShader(
  headDim: number, unit: MatrixUnitShape, m: MatrixSpelling, storedHalf = false,
): string {
  if (headDim % 4 !== 0 || headDim > 32 || headDim < 4) {
    throw new RangeError("matrix attention takes a head of four to thirty-two channels");
  }
  const TILE_STRIDE = tileStride(headDim);
  const { M, N, K } = unit;
  // Two lanes share a softmax row and take half the staged keys each, and the
  // units walk the pass in whole tiles.
  if (KEY_TILE % 2 !== 0 || KEY_TILE % N !== 0) {
    throw new RangeError(`a ${KEY_TILE}-key pass cannot be halved into whole ${N}-wide units`);
  }
  const vectors = headDim / 4;
  const keyTiles = KEY_TILE / N;
  // The tiles span the padded width; the loads and stores that reach the
  // model's own tensors keep the head's true width.
  const channelTiles = paddedHeadDim(headDim) / N;
  const contractions = paddedHeadDim(headDim) / K;
  // A matrix load or store reaches offset + stride * rows elements, not the
  // last element it actually touches. An array sized to the last element is
  // out of bounds by the extension's own rule, which is undefined behaviour
  // however valid every index in it is.
  const reach = (maxOffset: number, stride: number, rows: number): number =>
    maxOffset + stride * rows;
  const rows = ATTENTION_MATRIX_QUERY_TILE;
  const lines = (count: number, body: (index: number) => string): string =>
    Array.from({ length: count }, (_, index) => body(index)).join(String.fromCharCode(10));
  // Each lane carries a fixed share of the staged key and value tiles between
  // passes, so the fetch for the next one overlaps this one's multiplies.
  //
  // The share rounds up rather than having to divide: a narrow head makes the
  // tile smaller than the workgroup, and refusing that geometry cost the
  // kernel every four-subgroup arrangement at a head of eight, where a 32-key
  // tile is 64 vector items against 128 lanes. The lanes past the end sit the
  // fetch and the staging out; their prefetch registers keep the zero they
  // were declared with, which is what an absent key contributes anyway.
  const keyItems = KEY_TILE * (headDim / 4);
  const keyPerLane = Math.ceil(keyItems / LANES);
  // The guard is emitted only where it is needed, so a geometry whose tile
  // does fill the lanes generates exactly the shader it did before.
  const keyBound = (body: string): string => (keyItems % LANES === 0
    ? body.replace(/^\n/, "")
    : `    if (item < ${keyItems}u) {${body}
    }`.replace(/^ {4}if/, "if"));
  if ((rows * (headDim / 4)) % LANES !== 0) {
    throw new RangeError("the output tile must divide evenly among the lanes");
  }
  const outPerLane = (rows * (headDim / 4)) / LANES;
  // A head narrower than a matrix unit is contracted over the padded width, so
  // the units read channels past the head that no staging loop writes. They
  // have to be zero, and a device is not obliged to hand out zeroed workgroup
  // memory: Dawn does it and wgpu does it only when asked, which is why the
  // same kernel folded to a pLDDT of 96.48 on one and 50 on the other. Written
  // once, because nothing else ever writes them.
  const padVectors = (paddedHeadDim(headDim) - headDim) / 4;
  const padTile = (name: string, tileRows: number, type: string): string => padVectors === 0
    ? "" : `  for (var item = lane; item < ${tileRows * padVectors}u; item += ${LANES}u) {
    let base = (item / ${padVectors}u) * ${TILE_STRIDE}u
      + (${vectors}u + item % ${padVectors}u) * 4u;
${lines(4, (c) => `    ${name}[base + ${c}u] = ${type}(0);`)}
  }
`;
  const padChannels = padTile("queries_tile", rows, "f16")
    + (storedHalf ? "" : padTile("keys_tile", KEY_TILE, "f16"));
  const fetchKeyValue = (at: string): string => lines(keyPerLane, (i) => `  {
    let item = lane + ${i * LANES}u;
${keyBound(`
    let global_key = ${at} + item / ${headDim / 4}u;
    let at_index = ((batch_index * p.queries + global_key) * p.heads + head)
      * ${headDim / 4}u + item % ${headDim / 4}u;
    // Under an if rather than a select, which evaluates both of its arms and so
    // reads the tensor past its end for a key that does not exist. WebGPU
    // clamps that read, but a device may be asked not to.
    next_k_${i} = vec4<f32>(0.0);
    next_v_${i} = vec4<f32>(0.0);
    if (global_key < p.queries) {
      next_k_${i} = key[at_index];
      next_v_${i} = value[at_index];
    }`)}
  }`);
  const queriesLength = reach((rows - M) * TILE_STRIDE, TILE_STRIDE, M);
  const keysLength = reach((keyTiles - 1) * N * TILE_STRIDE, TILE_STRIDE, N);
  const valuesLength = reach((keyTiles - 1) * K * TILE_STRIDE + (channelTiles - 1) * N,
    TILE_STRIDE, K);
  const scoresLength = Math.max(
    reach((rows - M) * SCORE_STRIDE + (keyTiles - 1) * N, SCORE_STRIDE, M),
    reach((rows - M) * WEIGHTED_STRIDE + (channelTiles - 1) * N, WEIGHTED_STRIDE, M));
  const probabilitiesLength = reach(
    (rows - M) * PROBABILITY_STRIDE + (keyTiles - 1) * K, PROBABILITY_STRIDE, M);
  return `${m.prelude}enable f16;
struct Parameters {
  batch: u32, queries: u32, channels: u32, heads: u32,
  head_dim: u32, transpose: u32, has_pair_bias: u32,
  query_weight: u32, key_weight: u32, value_weight: u32,
  gating_weight: u32, gating_bias: u32, output_weight: u32,
  output_bias: u32, pair_weight: u32, pair_channels: u32,
  batch_offset: u32, batch_total: u32, bias_stride: u32, padding: u32,
};
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<${storedHalf ? "f16" : "vec4<f32>"}>;
@group(0) @binding(2) var<storage, read> value: array<${storedHalf ? "f16" : "vec4<f32>"}>;
@group(0) @binding(3) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
// Read four at a time: the count of these reads is what the softmax pays, not
// the bytes, and the rows are padded to four so that a vector read of one is
// aligned. See attentionPairBiasStride.
@group(0) @binding(5) var<storage, read> pair_bias: array<vec4<f32>>;
@group(0) @binding(6) var<uniform> p: Parameters;
@group(0) @binding(7) var<storage, read_write> output: array<vec4<f32>>;

fn mask_index(batch: u32, key_index: u32) -> u32 {
  let b = p.batch_offset + batch;
  if (p.transpose == 0u) { return b * p.queries + key_index; }
  return key_index * p.batch_total + b;
}

var<workgroup> queries_tile: array<f16, ${queriesLength}>;
${storedHalf ? "" : `var<workgroup> keys_tile: array<f16, ${keysLength}>;
var<workgroup> values_tile: array<f16, ${valuesLength}>;`}
// Scores while the softmax runs, then the weighted output the matrix units
// write; a barrier separates the last read of one from the first write of the
// other, and the scores are the larger, so one buffer serves both.
var<workgroup> scores: array<f32, ${scoresLength}>;
var<workgroup> probabilities: array<f16, ${probabilitiesLength}>;
var<workgroup> running_max: array<f32, ${ATTENTION_MATRIX_QUERY_TILE}>;
var<workgroup> running_sum: array<f32, ${ATTENTION_MATRIX_QUERY_TILE}>;
var<workgroup> rescale: array<f32, ${ATTENTION_MATRIX_QUERY_TILE}>;
// The mask varies only by key, so reading it inside the softmax read it once
// per row: sixty-four times more often than it changes. Staged once per pass.
var<workgroup> mask_tile: array<f32, ${KEY_TILE}>;

// Pinned, because every index here counts on thirty-two lanes a subgroup.
@compute @workgroup_size(${LANES}, 1, 1)${m.subgroupSize(MATRIX_LANES)}
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  // Dividing the local index by the subgroup width gives the same number, but
  // the uniformity analysis cannot prove that and rejects every matrix offset
  // built from it. This builtin is the sanctioned way to say it.
  @builtin(subgroup_id) subgroup: u32,
) {
  let lane = local.x;
${padChannels}
  // Which sixteen queries this subgroup owns.
  let rows_at = subgroup * ${UNIT}u;
  // The batch is the fastest-varying dimension, so that neighbouring
  // workgroups share the block of pair bias they read. See batchFirst.
  let query_origin = group.y * ${ATTENTION_MATRIX_QUERY_TILE}u;
  // The running output stays in registers: read and written once a pass in
  // workgroup memory it was a third of the traffic of the pass, and the array
  // it needed is workgroup storage that occupancy wants back. Each lane owns a
  // fixed set of four-channel groups, so the indices are computed once.
${lines(outPerLane, (j) => `  let own_row_${j} = (lane + ${j * LANES}u) / ${vectors}u;
  let own_vector_${j} = (lane + ${j * LANES}u) % ${vectors}u;
  var out_${j} = vec4<f32>(0.0);`)}
  let batch_index = group.x;
  let head = group.z;

  // Q is read once for the whole key loop, so it is staged once.
  for (var item = lane; item < ${ATTENTION_MATRIX_QUERY_TILE * vectors}u; item += ${LANES}u) {
    let row = item / ${vectors}u;
    let vector = item % ${vectors}u;
    let global_query = query_origin + row;
    var held = vec4<f32>(0.0);
    if (global_query < p.queries) {
      held = query[((batch_index * p.queries + global_query) * p.heads + head) * ${vectors}u + vector];
    }
    let base = row * ${TILE_STRIDE}u + vector * 4u;
    queries_tile[base] = f16(held.x);
    queries_tile[base + 1u] = f16(held.y);
    queries_tile[base + 2u] = f16(held.z);
    queries_tile[base + 3u] = f16(held.w);
  }
  if (lane < ${ATTENTION_MATRIX_QUERY_TILE}u) {
    running_max[lane] = -1e30;
    running_sum[lane] = 0.0;
  }
${storedHalf ? "" : `${lines(keyPerLane, (i) => `  var next_k_${i} = vec4<f32>(0.0);
  var next_v_${i} = vec4<f32>(0.0);`)}
${fetchKeyValue("0u")}`}
  workgroupBarrier();

  for (var key_origin = 0u; key_origin < p.queries; key_origin += ${KEY_TILE}u) {
    // Whether this pass needs any bounds test at all. Uniform, so the whole
    // workgroup takes one branch; the tail pass is the only checked one.
    let whole_pass = key_origin + ${KEY_TILE}u <= p.queries
      && query_origin + ${ATTENTION_MATRIX_QUERY_TILE}u <= p.queries;
${storedHalf ? "" : `${lines(keyPerLane, (i) => `    {
      let item = lane + ${i * LANES}u;
${keyBound(`
      let base = (item / ${vectors}u) * ${TILE_STRIDE}u + (item % ${vectors}u) * 4u;
      keys_tile[base] = f16(next_k_${i}.x); keys_tile[base + 1u] = f16(next_k_${i}.y);
      keys_tile[base + 2u] = f16(next_k_${i}.z); keys_tile[base + 3u] = f16(next_k_${i}.w);
      values_tile[base] = f16(next_v_${i}.x); values_tile[base + 1u] = f16(next_v_${i}.y);
      values_tile[base + 2u] = f16(next_v_${i}.z); values_tile[base + 3u] = f16(next_v_${i}.w);`)}
    }`)}
    workgroupBarrier();
    // The next pass's keys and values are fetched while the units still work
    // on these, so a global load is never what the multiplies wait for.
${fetchKeyValue(`key_origin + ${KEY_TILE}u`)}`}

    // S = Q K^T. The key tile is [key][channel], so a column-major load of it
    // is the transpose the right operand wants, at no cost.
    for (var tile = 0u; tile < ${keyTiles}u; tile += 1u) {
      var product = ${m.zero("f32", N, M)};
      for (var step = 0u; step < ${contractions}u; step += 1u) {
        let left = ${m.load(m.left("f16", K, M), "queries_tile",
    `rows_at * ${TILE_STRIDE}u + step * ${K}u`, `${TILE_STRIDE}u`)};
        let right = ${storedHalf
    ? m.load(m.right("f16", N, K), "key",
      `((batch_index * p.queries + key_origin + tile * ${N}u) * p.heads + head) * ${headDim}u`
        + ` + step * ${K}u`, `p.heads * ${headDim}u`, "col")
    : m.load(m.right("f16", N, K), "keys_tile",
      `tile * ${N} * ${TILE_STRIDE}u + step * ${K}u`, `${TILE_STRIDE}u`, "col")};
        product = ${m.multiplyAccumulate("left", "right", "product")};
      }
      ${m.store("scores", `rows_at * ${SCORE_STRIDE}u + tile * ${N}u`,
    "product", `${SCORE_STRIDE}u`)};
    }
    for (var column = lane; column < ${KEY_TILE}u; column += ${LANES}u) {
      let global_key = key_origin + column;
      var held_mask = 0.0;
      if (global_key < p.queries) { held_mask = mask[mask_index(batch_index, global_key)]; }
      mask_tile[column] = held_mask;
    }
    workgroupBarrier();

    // Online softmax over the staged scores. Every lane works: a subgroup owns
    // sixteen rows and has thirty-two lanes, so two lanes share a row and take
    // half its keys each. They meet through subgroupShuffleXor by one, which
    // is a register exchange inside the subgroup and needs no barrier: the
    // same split the reference CUDA kernel makes with a shuffle-xor.
    {
      let in_subgroup = lane % 32u;
      let row = rows_at + in_subgroup / 2u;
      let first = (in_subgroup % 2u) * ${KEY_TILE / 2}u;
      let global_query = query_origin + row;
      let previous_max = running_max[row];
      var next_max = previous_max;
      // Both invariants of the inner loop, hoisted out of it.
      let live_query = global_query < p.queries;
      // A whole number of vectors: the row length is a multiple of four, the
      // key origin a multiple of the key tile, and the half-row offset even.
      let bias_row = ((head * p.queries + global_query) * p.bias_stride
        + key_origin + first) / 4u;
${lines(KEY_TILE / 8, (v) => `      var bias_${v} = vec4<f32>(0.0);`)}
      if (p.has_pair_bias != 0u && live_query) {
${lines(KEY_TILE / 8, (v) => `        bias_${v} = pair_bias[bias_row + ${v}u];`)}
      }
      let score_row = row * ${SCORE_STRIDE}u;
      // The row's own share of the logits, held between the two sweeps over
      // it. Written back to the staged scores instead, each of the sixteen
      // costs a workgroup write here and a workgroup read below, and the pass
      // walks the tile three times over rather than once.
${lines(KEY_TILE / 2, (j) => `      var held_${j} = 0.0;`)}
      if (whole_pass) {
${lines(KEY_TILE / 2, (j) => `        {
          let column = first + ${j}u;
          var logit = scores[score_row + column] + 1e9 * (mask_tile[column] - 1.0);
          logit += bias_${Math.floor(j / 4)}[${j % 4}];
          // Scaled once here so the exponential below is the hardware's exp2.
          held_${j} = clamp(logit, -1e8, 1e8) * 1.44269504088896340736;
          next_max = max(next_max, held_${j});
        }`)}
      } else {
${lines(KEY_TILE / 2, (j) => `        {
          let column = first + ${j}u;
          var logit = -1e9;
          if (live_query && key_origin + column < p.queries) {
            logit = scores[score_row + column] + 1e9 * (mask_tile[column] - 1.0);
            logit += bias_${Math.floor(j / 4)}[${j % 4}];
            logit = clamp(logit, -1e8, 1e8);
          }
          held_${j} = logit * 1.44269504088896340736;
          next_max = max(next_max, held_${j});
        }`)}
      }
      next_max = max(next_max, subgroupShuffleXor(next_max, 1u));
      var total = 0.0;
${lines(KEY_TILE / 2, (j) => `      {
        let weight = exp2(held_${j} - next_max);
        probabilities[row * ${PROBABILITY_STRIDE}u + first + ${j}u] = f16(weight);
        total += weight;
      }`)}
      total += subgroupShuffleXor(total, 1u);
      let previous_scale = exp2(previous_max - next_max);
      // One of the pair writes the row's carried state; both computed it.
      if (in_subgroup % 2u == 0u) {
        rescale[row] = previous_scale;
        running_max[row] = next_max;
        running_sum[row] = running_sum[row] * previous_scale + total;
      }
    }
    workgroupBarrier();

    // O = O * previous_scale + P V, with P V into a fresh accumulator because
    // the rescale is per row and a matrix result cannot be scaled row-wise.
    for (var channel = 0u; channel < ${channelTiles}u; channel += 1u) {
      var product = ${m.zero("f32", N, M)};
      for (var tile = 0u; tile < ${keyTiles}u; tile += 1u) {
        let left = ${m.load(m.left("f16", K, M), "probabilities",
    `rows_at * ${PROBABILITY_STRIDE}u + tile * ${K}u`, `${PROBABILITY_STRIDE}u`)};
        let right = ${storedHalf
    ? m.load(m.right("f16", N, K), "value",
      `((batch_index * p.queries + key_origin + tile * ${K}u) * p.heads + head) * ${headDim}u`
        + ` + channel * ${N}u`, `p.heads * ${headDim}u`)
    : m.load(m.right("f16", N, K), "values_tile",
      `tile * ${K} * ${TILE_STRIDE}u + channel * ${N}u`, `${TILE_STRIDE}u`)};
        product = ${m.multiplyAccumulate("left", "right", "product")};
      }
      ${m.store("scores", `rows_at * ${WEIGHTED_STRIDE}u + channel * ${N}u`,
    "product", `${WEIGHTED_STRIDE}u`)};
    }
    workgroupBarrier();
${lines(outPerLane, (j) => `    {
      let staged = own_row_${j} * ${WEIGHTED_STRIDE}u + own_vector_${j} * 4u;
      out_${j} = out_${j} * rescale[own_row_${j}] + vec4<f32>(scores[staged],
        scores[staged + 1u], scores[staged + 2u], scores[staged + 3u]);
    }`)}
    workgroupBarrier();
  }

${lines(outPerLane, (j) => `  {
    let global_query = query_origin + own_row_${j};
    if (global_query < p.queries) {
      let at = ((batch_index * p.queries + global_query) * p.heads + head)
        * ${vectors}u + own_vector_${j};
      output[at] = (out_${j} / running_sum[own_row_${j}]) * gate[at];
    }
  }`)}
}`;
}
