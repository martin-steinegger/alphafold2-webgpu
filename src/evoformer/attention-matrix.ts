/**
 * Flash attention over the hardware matrix units.
 *
 * The register-resident kernel this competes with keeps one query's whole
 * accumulator in registers and reads every key and value itself, which costs
 * no cross-lane traffic and is why it beats every subgroup variant here: those
 * reduce each query-key dot product with `subgroupAdd`, once per key. A matrix
 * unit does that reduction in hardware, so this kernel pays neither the
 * cross-lane traffic nor the per-key reduction, and reads each staged key tile
 * once for sixteen queries instead of once per query.
 *
 * The shape is the one both reference implementations use: 16x16x16 tiles with
 * f16 operands and an f32 accumulator, which is what Nvidia's units implement
 * and what `subgroupMatrixConfigs` reports. Operands must live in workgroup or
 * storage memory as the component type, so the query, key and value tiles are
 * converted into workgroup f16 once per tile rather than per multiply.
 *
 * The online softmax cannot rescale an accumulator in place: its correction is
 * per query row and `subgroupMatrixScalarMultiply` takes one uniform scalar.
 * So `P V` accumulates into a freshly zeroed result each key tile, is stored to
 * workgroup memory, and the running output is rescaled there in plain f32 —
 * which is exactly what the reference CUDA kernel does for the same reason.
 */
import { attentionMatrixConfig, type MatrixUnitShape } from "../runtime/gemm.js";

/**
 * Subgroups per workgroup, each owning sixteen queries.
 *
 * One subgroup was the obvious first shape and it loses: sixteen queries per
 * workgroup against the register kernel's sixty-four is a quarter of the work
 * per staging and per barrier, and it measured 0.62x. Four subgroups stage the
 * key and value tiles once for sixty-four queries, which is what makes reading
 * a key once for sixteen queries into reading it once for sixty-four.
 */
const SUBGROUPS = 4;
/** The M of every tile, fixed by the unit shape the device reports. */
const UNIT = 16;
/** Queries one workgroup owns. */
export const ATTENTION_MATRIX_QUERY_TILE = SUBGROUPS * UNIT;
/**
 * Keys staged per pass, shared by every subgroup.
 *
 * Each pass costs four barriers whatever its width, so a wider tile buys back
 * barriers and amortises the conversion of the staged tiles. Sixty-four is
 * what fits once the scores and the weighted output share their storage.
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
 * Workgroup memory is thirty-two banks of four bytes, so lane `i` reading
 * element `i * stride` lands in bank `(i * stride) % 32` for f32 and
 * `(i * stride / 2) % 32` for f16. A stride sharing a factor with thirty-two
 * collapses the lanes onto few banks and serialises the read; a stride coprime
 * with it spreads them across all thirty-two.
 *
 * The head width is thirty-two, which is the worst case of all: the staged key
 * tile is read column-major, so consecutive lanes are one row apart, and at a
 * stride of thirty-two f16 that is `(16i) % 32` — two banks for thirty-two
 * lanes, a sixteen-way conflict on the hottest read in the kernel. Thirty-four
 * gives `(17i) % 32`, and seventeen is coprime with thirty-two, so no two
 * lanes collide. The f32 rows take the same treatment with an odd stride.
 */
const TILE_STRIDE = 34;
const SCORE_STRIDE = KEY_TILE + 1;
const WEIGHTED_STRIDE = 33;
const PROBABILITY_STRIDE = KEY_TILE + 2;

/** Whether this device can run it, and with which unit shape. */
export function attentionMatrixShape(
  device: GPUDevice, headDim: number,
  configs: readonly { componentType: string; resultComponentType: string;
    M: number; N: number; K: number }[],
): MatrixUnitShape | undefined {
  if (headDim !== 32) return undefined;
  return attentionMatrixConfig(device, configs, UNIT);
}

/**
 * Workgroup bytes the kernel declares, which a device must permit.
 *
 * Every array is sized by the same `offset + stride * rows` reach the loads
 * and stores claim, not by the last element they touch, so this matches what
 * the shader actually declares.
 */
export function attentionMatrixStorageBytes(headDim: number, storedHalf = false): number {
  const rows = ATTENTION_MATRIX_QUERY_TILE;
  const unit = UNIT;
  const keyTiles = KEY_TILE / unit;
  const channelTiles = headDim / unit;
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
    + rows * headDim * 4                          // accumulated
    + 3 * rows * 4;                               // max, sum, rescale
}

export function createAttentionMatrixFlashShader(
  headDim: number, unit: MatrixUnitShape, storedHalf = false,
): string {
  if (headDim !== 32) throw new RangeError("matrix attention is written for a 32-channel head");
  const { M, N, K } = unit;
  // Two lanes share a softmax row and take half the staged keys each. A half
  // narrower than one unit measured wrong — sixteen keys per pass returned a
  // relative error of 3.7 where thirty-two, forty-eight and sixty-four all
  // return 4.6e-4 — so the shape is refused rather than shipped unexplained.
  if (KEY_TILE % 2 !== 0 || KEY_TILE / 2 < N || KEY_TILE % N !== 0) {
    throw new RangeError(`a ${KEY_TILE}-key pass cannot be halved into whole ${N}-wide units`);
  }
  const vectors = headDim / 4;
  const keyTiles = KEY_TILE / N;
  const channelTiles = headDim / N;
  const contractions = headDim / K;
  // A matrix load or store reaches `offset + stride * rows` elements, not the
  // last element it actually touches. An array sized to the last element is
  // out of bounds by the extension's own rule, which is undefined behaviour
  // however valid every index in it is.
  const reach = (maxOffset: number, stride: number, rows: number): number =>
    maxOffset + stride * rows;
  const rows = ATTENTION_MATRIX_QUERY_TILE;
  const queriesLength = reach((rows - M) * TILE_STRIDE, TILE_STRIDE, M);
  const keysLength = reach((keyTiles - 1) * N * TILE_STRIDE, TILE_STRIDE, N);
  const valuesLength = reach((keyTiles - 1) * K * TILE_STRIDE + (channelTiles - 1) * N,
    TILE_STRIDE, K);
  const scoresLength = Math.max(
    reach((rows - M) * SCORE_STRIDE + (keyTiles - 1) * N, SCORE_STRIDE, M),
    reach((rows - M) * WEIGHTED_STRIDE + (channelTiles - 1) * N, WEIGHTED_STRIDE, M));
  const probabilitiesLength = reach(
    (rows - M) * PROBABILITY_STRIDE + (keyTiles - 1) * K, PROBABILITY_STRIDE, M);
  return `enable chromium_experimental_subgroup_matrix;
enable f16;
enable subgroups;
struct Parameters {
  batch: u32, queries: u32, channels: u32, heads: u32,
  head_dim: u32, transpose: u32, has_pair_bias: u32,
  query_weight: u32, key_weight: u32, value_weight: u32,
  gating_weight: u32, gating_bias: u32, output_weight: u32,
  output_bias: u32, pair_weight: u32, pair_channels: u32,
  batch_offset: u32, batch_total: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> query: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> key: array<${storedHalf ? "f16" : "vec4<f32>"}>;
@group(0) @binding(2) var<storage, read> value: array<${storedHalf ? "f16" : "vec4<f32>"}>;
@group(0) @binding(3) var<storage, read> gate: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read> pair_bias: array<f32>;
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
var<workgroup> accumulated: array<f32, ${ATTENTION_MATRIX_QUERY_TILE * headDim}>;
var<workgroup> running_max: array<f32, ${ATTENTION_MATRIX_QUERY_TILE}>;
var<workgroup> running_sum: array<f32, ${ATTENTION_MATRIX_QUERY_TILE}>;
var<workgroup> rescale: array<f32, ${ATTENTION_MATRIX_QUERY_TILE}>;
// The mask varies only by key, so reading it inside the softmax read it once
// per row: sixty-four times more often than it changes. Staged once per pass.
var<workgroup> mask_tile: array<f32, ${KEY_TILE}>;

@compute @workgroup_size(${LANES}, 1, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  // Dividing the local index by the subgroup width gives the same number, but
  // the uniformity analysis cannot prove that and rejects every matrix offset
  // built from it. This builtin is the sanctioned way to say it.
  @builtin(subgroup_id) subgroup: u32,
) {
  let lane = local.x;
  // Which sixteen queries this subgroup owns.
  let rows_at = subgroup * ${UNIT}u;
  let query_origin = group.x * ${ATTENTION_MATRIX_QUERY_TILE}u;
  let batch_index = group.y;
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
  for (var item = lane; item < ${ATTENTION_MATRIX_QUERY_TILE * headDim}u; item += ${LANES}u) {
    accumulated[item] = 0.0;
  }
  if (lane < ${ATTENTION_MATRIX_QUERY_TILE}u) {
    running_max[lane] = -1e30;
    running_sum[lane] = 0.0;
  }
  workgroupBarrier();

  for (var key_origin = 0u; key_origin < p.queries; key_origin += ${KEY_TILE}u) {
    // Whether this pass needs any bounds test at all. Uniform, so the whole
    // workgroup takes one branch; the tail pass is the only checked one.
    let whole_pass = key_origin + ${KEY_TILE}u <= p.queries
      && query_origin + ${ATTENTION_MATRIX_QUERY_TILE}u <= p.queries;
${storedHalf ? "" : `    for (var item = lane; item < ${KEY_TILE * vectors}u; item += ${LANES}u) {
      let row = item / ${vectors}u;
      let vector = item % ${vectors}u;
      let global_key = key_origin + row;
      var k = vec4<f32>(0.0);
      var v = vec4<f32>(0.0);
      // Uniform across the workgroup: every key of a whole pass is in range,
      // so the common pass loads without testing each key.
      if (whole_pass || global_key < p.queries) {
        let at = ((batch_index * p.queries + global_key) * p.heads + head) * ${vectors}u + vector;
        k = key[at];
        v = value[at];
      }
      let base = row * ${TILE_STRIDE}u + vector * 4u;
      keys_tile[base] = f16(k.x); keys_tile[base + 1u] = f16(k.y);
      keys_tile[base + 2u] = f16(k.z); keys_tile[base + 3u] = f16(k.w);
      values_tile[base] = f16(v.x); values_tile[base + 1u] = f16(v.y);
      values_tile[base + 2u] = f16(v.z); values_tile[base + 3u] = f16(v.w);
    }
    workgroupBarrier();`}

    // S = Q K^T. The key tile is [key][channel], so a column-major load of it
    // is the transpose the right operand wants, at no cost.
    for (var tile = 0u; tile < ${keyTiles}u; tile += 1u) {
      var product = subgroup_matrix_result<f32, ${N}, ${M}>();
      for (var step = 0u; step < ${contractions}u; step += 1u) {
        let left = subgroupMatrixLoad<subgroup_matrix_left<f16, ${K}, ${M}>>(
          &queries_tile, rows_at * ${TILE_STRIDE}u + step * ${K}u, false, ${TILE_STRIDE}u);
        let right = ${storedHalf
          ? `subgroupMatrixLoad<subgroup_matrix_right<f16, ${N}, ${K}>>(&key,
          ((batch_index * p.queries + key_origin + tile * ${N}u) * p.heads + head) * ${headDim}u
            + step * ${K}u, true, p.heads * ${headDim}u)`
          : `subgroupMatrixLoad<subgroup_matrix_right<f16, ${N}, ${K}>>(
          &keys_tile, tile * ${N} * ${TILE_STRIDE}u + step * ${K}u, true, ${TILE_STRIDE}u)`};
        product = subgroupMatrixMultiplyAccumulate(left, right, product);
      }
      subgroupMatrixStore(&scores, rows_at * ${SCORE_STRIDE}u + tile * ${N}u,
        product, false, ${SCORE_STRIDE}u);
    }
    for (var column = lane; column < ${KEY_TILE}u; column += ${LANES}u) {
      let global_key = key_origin + column;
      mask_tile[column] = select(0.0, mask[mask_index(batch_index, global_key)],
        global_key < p.queries);
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
      let bias_row = (head * p.queries + global_query) * p.queries + key_origin;
      let score_row = row * ${SCORE_STRIDE}u;
      if (whole_pass) {
        for (var column = first; column < first + ${KEY_TILE / 2}u; column += 1u) {
          var logit = scores[score_row + column] + 1e9 * (mask_tile[column] - 1.0);
          if (p.has_pair_bias != 0u) {
            logit += pair_bias[bias_row + column];
          }
          // Scaled once here so the exponential below is the hardware's exp2.
          let scaled = clamp(logit, -1e8, 1e8) * 1.44269504088896340736;
          scores[score_row + column] = scaled;
          next_max = max(next_max, scaled);
        }
      } else {
        for (var column = first; column < first + ${KEY_TILE / 2}u; column += 1u) {
          var logit = -1e9;
          if (live_query && key_origin + column < p.queries) {
            logit = scores[score_row + column] + 1e9 * (mask_tile[column] - 1.0);
            if (p.has_pair_bias != 0u) {
              logit += pair_bias[bias_row + column];
            }
            logit = clamp(logit, -1e8, 1e8);
          }
          let scaled = logit * 1.44269504088896340736;
          scores[score_row + column] = scaled;
          next_max = max(next_max, scaled);
        }
      }
      next_max = max(next_max, subgroupShuffleXor(next_max, 1u));
      var total = 0.0;
      for (var column = first; column < first + ${KEY_TILE / 2}u; column += 1u) {
        let weight = exp2(scores[score_row + column] - next_max);
        probabilities[row * ${PROBABILITY_STRIDE}u + column] = f16(weight);
        total += weight;
      }
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
      var product = subgroup_matrix_result<f32, ${N}, ${M}>();
      for (var tile = 0u; tile < ${keyTiles}u; tile += 1u) {
        let left = subgroupMatrixLoad<subgroup_matrix_left<f16, ${K}, ${M}>>(
          &probabilities, rows_at * ${PROBABILITY_STRIDE}u + tile * ${K}u,
          false, ${PROBABILITY_STRIDE}u);
        let right = ${storedHalf
          ? `subgroupMatrixLoad<subgroup_matrix_right<f16, ${N}, ${K}>>(&value,
          ((batch_index * p.queries + key_origin + tile * ${K}u) * p.heads + head) * ${headDim}u
            + channel * ${N}u, false, p.heads * ${headDim}u)`
          : `subgroupMatrixLoad<subgroup_matrix_right<f16, ${N}, ${K}>>(
          &values_tile, tile * ${K} * ${TILE_STRIDE}u + channel * ${N}u, false, ${TILE_STRIDE}u)`};
        product = subgroupMatrixMultiplyAccumulate(left, right, product);
      }
      subgroupMatrixStore(&scores, rows_at * ${WEIGHTED_STRIDE}u + channel * ${N}u,
        product, false, ${WEIGHTED_STRIDE}u);
    }
    workgroupBarrier();
    for (var item = lane; item < ${ATTENTION_MATRIX_QUERY_TILE * headDim}u; item += ${LANES}u) {
      let row = item / ${headDim}u;
      accumulated[item] = accumulated[item] * rescale[row]
        + scores[row * ${WEIGHTED_STRIDE}u + item % ${headDim}u];
    }
    workgroupBarrier();
  }

  for (var item = lane; item < ${ATTENTION_MATRIX_QUERY_TILE * vectors}u; item += ${LANES}u) {
    let row = item / ${vectors}u;
    let vector = item % ${vectors}u;
    let global_query = query_origin + row;
    if (global_query < p.queries) {
      let at = ((batch_index * p.queries + global_query) * p.heads + head) * ${vectors}u + vector;
      let base = row * ${headDim}u + vector * 4u;
      let sum = running_sum[row];
      let held = vec4<f32>(accumulated[base], accumulated[base + 1u],
        accumulated[base + 2u], accumulated[base + 3u]) / sum;
      output[at] = held * gate[at];
    }
  }
}`;
}
