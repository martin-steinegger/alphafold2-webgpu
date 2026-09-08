/**
 * The register-blocked dense projection shared by every AlphaFold module.
 *
 * All of them compute row-major A[rows, inner] x row-major W[inner, columns]
 * and differ only in where the operands come from and what the epilogue does
 * with a result, so the tiling lives here once and callers supply WGSL
 * fragments for the rest.
 *
 * One workgroup of 256 invocations covers a 64x128 output tile. Each invocation
 * keeps eight contiguous rows by four columns in registers, so a single k step
 * costs two vector reads of A plus one of W and performs 32 fused
 * multiply-adds. A is staged k-major, which keeps both its global loads and its
 * shared loads contiguous. Workgroup storage stays at 6 KiB, well inside the
 * 16 KiB every WebGPU implementation guarantees.
 */

/**
 * Workgroups one dispatch dimension may hold, which every WebGPU device grants
 * and none here grants more of.
 *
 * The row tiles of a tall contraction outgrow it: the extra-MSA global
 * attention projects `sequences * length` rows, which at 800 residues passes it
 * at 5,242 extra sequences, and at the multimer's 1,152 at 3,640 residues. It
 * announced itself as a validation error rather than a wrong answer, but it
 * stopped the prediction. `gemmGrid` folds the excess into the column
 * dimension and the kernels take it apart again.
 */
export const GEMM_GRID_LIMIT = 65_535;

export const GEMM_TILE_ROWS = 64;
export const GEMM_TILE_COLUMNS = 128;
const GEMM_TILE_INNER = 8;
const GEMM_THREADS = 256;
/**
 * Workgroup storage every WebGPU implementation guarantees. A deeper k tile
 * stages more of both operands, and an epilogue reserves `gemm_stage` on top,
 * so the depth is only taken when the three together still fit.
 */
const GEMM_WORKGROUP_BYTES = 16384;

/** The region one subgroup owns and its width. */
export const MATRIX_REGION = 32;
/** Lanes a subgroup. Every index and tile placement in the kernels assumes it. */
export const MATRIX_LANES = 32;

/**
 * One hardware matrix configuration: `Accum[M][N] += A[M][K] * B[K][N]`.
 *
 * A device advertises the shapes and component types its units actually
 * implement, and they differ by vendor rather than by preference. Apple offers
 * f32 at 8x8x8, which is what this kernel was first written against. Nvidia
 * offers no f32 configuration at all: its units take f16 operands, at 16x16x16
 * and two narrower shapes, and accumulate into f32. So the shape cannot be a
 * constant, and a device that reports nothing usable keeps the hand-tiled
 * kernel as before.
 *
 * `resultType` is always f32 here. The units can accumulate in f16 as well and
 * this deliberately does not: the whole reason the matrix path was exempt from
 * the half-precision margin is that its reduction stays single precision, and
 * a contraction over a deep MSA is exactly where an f16 accumulator fails.
 */
export interface MatrixUnitShape {
  readonly componentType: "f32" | "f16";
  readonly M: number;
  readonly N: number;
  readonly K: number;
}

/**
 * Sixteen-deep steps staged before the units walk them.
 *
 * Each staging costs two barriers whatever its depth, so a deeper one buys
 * them back and grows only the operands, which are small beside the result.
 */
const GEMM_STAGE_STEPS = 1;

/**
 * Workgroup bytes the f16 projection kernel declares.
 *
 * It stages both operands and the result, which is more than the 16 KiB every
 * WebGPU implementation guarantees. A device that grants only the baseline
 * cannot run it, and the candidate is withheld rather than left to fail
 * validation inside the measurement.
 */
export function matrixGemmStorageBytes(
  unit: MatrixUnitShape, tileColumns = GEMM_TILE_COLUMNS, kStep = unit.K,
): number {
  if (unit.componentType !== "f16") return 0;
  const { M, N, K } = unit;
  const columnTiles = tileColumns / N;
  const columnGroups = columnTiles % 2 === 0 ? 2 : 1;
  const subgroups = (GEMM_TILE_ROWS / M) * columnGroups;
  const aStride = kStep + 2;
  const bStride = tileColumns + 8;
  const outStride = N + 1;
  const reach = (maxOffset: number, stride: number, count: number): number =>
    maxOffset + stride * count;
  return reach((GEMM_TILE_ROWS - M) * aStride, aStride, M) * 2
    + reach((columnTiles - 1) * N, bStride, kStep) * 2
    + reach((subgroups - 1) * M * outStride, outStride, M) * 4;
}

/** Apple's configuration, and what a variant naming no shape means. */
export const MATRIX_SHAPE_F32_8: MatrixUnitShape = { componentType: "f32", M: 8, N: 8, K: 8 };

/**
 * How the k loop computes, chosen per device rather than written down.
 *
 * `precision` is the arithmetic of the staged tiles, the products and the
 * accumulator; `inner` is the depth of one staged k tile. Neither is visible
 * to a caller: the accumulator still reaches the epilogue as `vec4<f32>`, and
 * the k depth does not enter the dispatch grid, which `gemmGrid` derives from
 * the output tile alone. That is what makes them safe to measure and switch
 * without touching a single call site.
 */
export interface GemmVariant {
  /**
   * Where half precision is used, which is not one choice but three.
   *
   * `f16` stages, multiplies and accumulates in half precision. It is the
   * fastest and it is not shippable: a contraction over a deep MSA overflows
   * the accumulator, which took a 508-row prediction from 96.8 pLDDT to 69.9
   * and its pTM to NaN.
   *
   * `f16-mixed` multiplies in half precision and accumulates in single. The
   * multiply is where Apple's rate doubles and the long reduction is where the
   * error grows, so the two can be bought separately.
   *
   * `f16-chunked` accumulates in half precision for the depth of one staged
   * tile and folds that into an f32 running sum once per tile, so the error
   * grows with the square root of 8 or 16 terms rather than of K while all but
   * one add per tile stays in half precision.
   *
   * `matrix` is not half precision at all: it is the hardware matrix units,
   * accumulating in f32, and it is both faster than any of the above and
   * exact. It serves only callers that declared their operands as arrays and
   * store one element at a time, and `inner` does not apply to it: the units
   * fix the depth of a multiply, and `matrixDepth` says how many of those a
   * step stages at once.
   */
  readonly precision: "f32" | "f16" | "f16-mixed" | "f16-chunked" | "matrix";
  readonly inner: 8 | 16;
  /**
   * Unit depths of the contraction one step stages, for the matrix precision.
   *
   * One is two barriers for as many multiplies as the tile is wide; two spend
   * them once for twice as many, and stage the same operands to do it. It
   * costs workgroup storage, which this kernel ranks by, so the depth is
   * offered to the calibration rather than written down. Absent means one.
   */
  readonly matrixDepth?: number;
  /**
   * What a caller that cannot reach the matrix units computes instead.
   *
   * Only some projections can use them: an operand that is a function call, or
   * four weight matrices selected by column, is not something
   * `subgroupMatrixLoad` can address. Those callers would otherwise drop all
   * the way back to f32 whenever the matrix units won, losing the
   * half-precision gain they did qualify for — which on this model is most of
   * the projection time, since the largest single shape is one of them.
   */
  readonly fallback?: "f32" | "f16-mixed" | "f16-chunked";
  /**
   * Which hardware configuration `matrix` means on this device.
   *
   * Absent is Apple's f32 8x8x8, which is what every existing caller and
   * every recorded measurement assumed. An f16 configuration computes the
   * products in half precision, so unlike the f32 one it is not exact and is
   * held to the half-precision margin rather than the matrix one.
   */
  readonly matrix?: MatrixUnitShape;
}

export const GEMM_VARIANT_F32: GemmVariant = { precision: "f32", inner: GEMM_TILE_INNER };

let selectedVariant: GemmVariant = GEMM_VARIANT_F32;

/** The variant every `createTiledGemmShader` uses unless told otherwise. */
export function gemmVariant(): GemmVariant {
  return selectedVariant;
}

/**
 * Installs the measured winner.
 *
 * `src/runtime/gemm-selection.ts` calls this once per device, from inside
 * `requestAlphaFoldDevice`, before any consumer can hold the device and so
 * before any projection shader exists. Setting it later would let one cache
 * key describe two different shaders, which `ComputePipelineCache` reports as
 * a collision rather than running.
 */
export function setGemmVariant(variant: GemmVariant): void {
  selectedVariant = variant;
}

export interface TiledGemmShader {
  /** WGSL emitted before the entry point: bindings, structs, and helpers. */
  readonly preamble: string;
  /** Expression for the number of output rows. */
  readonly rows: string;
  /** Expression for the contraction length. */
  readonly inner: string;
  /** Expression for the number of output columns. */
  readonly columns: string;
  /** Expression producing one A element, with `row` and `k` in scope. */
  readonly sourceElement: string;
  /** Expression producing one W element, with `k` and `column` in scope. */
  readonly weightElement: string;
  /** Statements storing one result, with `row`, `column` and `element` in scope. */
  readonly store: string;
  /**
   * Statements storing one invocation's four adjacent results at once, with
   * `row`, `column` (the first of the four, a multiple of four) and
   * `values: vec4<f32>` in scope. Replaces `store`; the caller bounds the
   * columns itself, which lets an epilogue combine neighbouring columns such
   * as a projection and its gate.
   */
  readonly storeVector?: string;
  /**
   * A whole-tile epilogue replacing the per-invocation stores. Runs after the
   * k loop with `acc0`..`acc{rows per invocation - 1}` (vec4<f32> each, four
   * adjacent columns of one row), `tile_row_origin`, `row_thread`,
   * `column_thread`, `column_origin`, `thread`, `gemm_rows` and `gemm_columns`
   * in scope, and `gemm_stage: array<f32, stageElements>` in workgroup storage
   * for transposing results before storing them. Barriers are permitted: every
   * invocation of the workgroup runs the epilogue.
   */
  readonly epilogue?: string;
  readonly stageElements?: number;
  /** Narrower tile for outputs that would otherwise waste most of a workgroup. */
  readonly tileColumns?: number;
  /**
   * The same operands again, as arrays rather than as expressions.
   *
   * `sourceElement` and `weightElement` are expressions because a caller may
   * unpack a half-precision word, window a tensor past a binding limit, or
   * index something else entirely. The hardware matrix units cannot consume an
   * expression: `subgroupMatrixLoad` takes an array, a base offset and a row
   * stride, and loads a whole 8x8 tile itself.
   *
   * A caller whose operands really are plain `array<f32>` in row-major order
   * says so here, and becomes eligible for the matrix kernel on a device that
   * has the units. Saying nothing is always safe and keeps the hand-tiled
   * kernel. Declaring this when it is not true is not detectable here and will
   * compute the wrong answer, so it is a claim, not a hint: element `[row][k]`
   * of A must live at `base + row * stride + k`, and `[k][column]` of W at
   * `base + k * stride + column`.
   */
  readonly sourceArray?: GemmOperandArray;
  readonly weightArray?: GemmOperandArray;
  /**
   * Which index of each operand runs contiguously in memory.
   *
   * Only the staged kernel reads this, and only to choose which index a lane
   * varies fastest when it fetches a tile and writes it into workgroup memory.
   * Getting it wrong costs bandwidth, not correctness — but the fetch and the
   * write have to agree, and when they did not the prediction came back at 26
   * pLDDT. The defaults are the common layouts — a source indexed `[row][k]`
   * and a weight indexed `[k][column]`. The triangle multiplication's incoming
   * direction has its source the other way round and read as though it did
   * not, its tile arrived one block row apart per lane and it cost 0.19s to
   * 0.27s of a recycle.
   */
  readonly sourceContiguous?: "k" | "row";
  readonly weightContiguous?: "column" | "k";
}

export interface GemmOperandArray {
  /** Name of an `array<f32>` binding in the preamble. */
  readonly array: string;
  /** Expression for the element index the operand starts at. Defaults to zero. */
  readonly base?: string;
  /** Expression for the distance in elements between consecutive rows. */
  readonly stride: string;
  /**
   * Whether the operand is stored with the contraction axis outermost.
   *
   * The default is `[row][k]`: element `[row][k]` at `base + row * stride + k`.
   * A contraction whose operand is accumulated over its sequences, as the
   * outer-product mean's is, has it the other way round — `[k][row]` at
   * `base + k * stride + row` — and transposing it to satisfy this would cost
   * a pass over the whole tensor. Only the f16 matrix kernel honours this; the
   * f32 one addresses the operand where it lies and has no such freedom, so a
   * caller declaring it is not offered that kernel.
   */
  readonly columnMajor?: boolean;
}

/**
 * Workgroup counts for a tiled GEMM dispatch.
 *
 * `tileColumns` must match the tile the shader was generated with; a grid
 * computed from a narrower tile silently leaves output columns unwritten.
 */
export function gemmGrid(
  rows: number, columns: number, tileColumns: number = GEMM_TILE_COLUMNS,
): readonly [number, number] {
  if (![rows, columns].every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError("GEMM dispatch dimensions must be positive safe integers");
  }
  const columnTiles = Math.ceil(columns / tileColumns);
  const rowTiles = Math.ceil(rows / GEMM_TILE_ROWS);
  const folds = Math.ceil(rowTiles / GEMM_GRID_LIMIT);
  return [columnTiles * folds, Math.min(rowTiles, GEMM_GRID_LIMIT)];
}

/**
 * Where a workgroup's output tile begins.
 *
 * With one fold this is `group.x` across the columns and `group.y` down the
 * rows, as it reads. With more, the folds ride in the high part of `group.x`,
 * which costs a division of a value the whole workgroup shares.
 */
const tileOrigins = (tileColumns: number): string => `  let gemm_column_tiles =
    (gemm_columns + ${tileColumns - 1}u) / ${tileColumns}u;
  let tile_column_origin = (group.x % gemm_column_tiles) * ${tileColumns}u;
  let tile_row_origin = (group.y + (group.x / gemm_column_tiles) * ${GEMM_GRID_LIMIT}u)
    * ${GEMM_TILE_ROWS}u;`;

/** Whether this shader and this variant can use the hardware matrix units. */
export function usesMatrixUnits(shader: TiledGemmShader, variant: GemmVariant): boolean {
  // The f32 kernel hands the units a pointer and a stride, so its operands
  // have to be plain arrays laid out the way it addresses them. The f16 one
  // cannot do that at all — the units want the component type, and these
  // operands are single precision — so it converts a tile into workgroup
  // memory first, and whatever produced that tile is free to be a packed
  // word, a windowed binding, or a choice between four weight blocks. The
  // element expressions every caller already supplies are enough for it.
  return variant.precision === "matrix"
    && (variant.matrix?.componentType === "f16"
      || (shader.sourceArray !== undefined && shader.weightArray !== undefined
        && shader.sourceArray.columnMajor !== true))
    // A whole-tile epilogue is written against `acc{n}` in the hand-tiled
    // thread mapping, which a matrix kernel does not have. `storeVector` is
    // fine: the result is staged in workgroup memory, so an invocation can
    // read four adjacent columns of it as easily as one.
    && shader.epilogue === undefined;
}

/**
 * The projection over the hardware matrix units.
 *
 * One subgroup owns the whole 64x128 output tile and walks it as eight 32x32
 * sub-regions, four left tiles by four right tiles at a time: sixteen
 * multiply-accumulates per eight loads, where the naive arrangement gets one
 * per two. Keeping the tile the hand-tiled kernel uses is what lets callers
 * opt in one at a time — `gemmGrid` does not know which shader is asking, so
 * two kernels with different tiles would hand one of them the wrong grid.
 *
 * `subgroupMatrixStore` requires a uniform offset and WGSL's uniformity
 * analysis is workgroup-scoped, so one subgroup per workgroup and every offset
 * from the workgroup id.
 *
 * The edges are the whole difficulty. A tile of columns that runs off the
 * right-hand side is harmless: the load stays inside the weights and only the
 * columns that do not exist come back wrong, and those are not stored. A tile
 * of rows that runs off the *end of the source* is not harmless, and not in
 * the way one would guess — it does not merely return zeros for the rows that
 * are missing, it returns nothing usable for the valid rows in the same tile.
 * At 81 rows that silently corrupted row 80 and nothing else, which end to end
 * moved the prediction by 6.5 pLDDT.
 *
 * So a region that would run past the end is loaded from an origin pulled back
 * far enough to fit, and the store maps each output row to wherever it landed
 * in the staged result. Rows are still written exactly once, by exactly one
 * workgroup, which matters because some callers accumulate into their output.
 * Below one region's worth of rows there is nowhere to pull back to, and that
 * case is computed without the units at all.
 */
/**
 * The projection over f16 matrix units, four subgroups to a workgroup.
 *
 * The f32 kernel below addresses the operands where they lie, because Apple's
 * units take the component type the tensors already are. An f16 unit cannot:
 * `subgroupMatrixLoad` reinterprets nothing, so the operands have to be f16 in
 * memory, and a staged tile is the only place they are.
 *
 * That staging is what the shape is chosen around. One subgroup owning the
 * whole 64x128 tile — which is what the f32 kernel does — leaves thirty-two
 * lanes to stage every tile and issue every multiply, and measured fifteen
 * times slower than the hand-tiled kernel. Four subgroups stage once for
 * sixty-four rows and take sixteen rows each, which is the same arrangement
 * that made the matrix attention kernel win.
 *
 * Every staged row is padded to an odd-ish length for the banks: workgroup
 * memory is thirty-two four-byte banks, and a stride sharing a factor with
 * thirty-two collapses the lanes of a column read onto a few of them.
 */
/**
 * The projection over the matrix units, staging `kStep` of the contraction.
 *
 * One unit of depth a step is two barriers for four multiplies, and the
 * staging around them is the kernel's cost rather than the multiplies: the
 * contraction reads 96 GB/s where the card gives 1.8 TB/s, and runs at a
 * ninth of what the units can do. A deeper step moves the same operands and
 * spends the barriers once for as many multiplies as it is deep. It costs
 * workgroup storage, which this kernel ranks by, so which depth wins is
 * measured per device rather than written down.
 */
function createMatrixGemmShaderF16(
  shader: TiledGemmShader, unit: MatrixUnitShape, kStep = unit.K,
): string {
  const { M, N, K } = unit;
  if (kStep % K !== 0) throw new RangeError("the staged depth must be whole units");
  const steps = kStep / K;
  const tileColumns = shader.tileColumns ?? GEMM_TILE_COLUMNS;
  const rowGroups = GEMM_TILE_ROWS / M;
  const columnTiles = tileColumns / N;
  if (!Number.isInteger(rowGroups) || !Number.isInteger(columnTiles)) {
    throw new RangeError(`a ${M}x${N} unit does not tile a ${GEMM_TILE_ROWS}x${tileColumns} output`);
  }
  // Subgroups are laid out across the columns as well as down the rows, so a
  // 64x128 tile is carried by 256 lanes rather than 128. The staging is what
  // wanted them — three thousand operand elements a step is the kernel's cost,
  // not the multiplies — and it halves the accumulators each subgroup holds.
  // Two, measured: one leaves 128 lanes staging three thousand elements a
  // step and costs 0.33 ms where two cost 0.19; four asks for 512 lanes, which
  // this driver would not run at all.
  const columnGroups = columnTiles % 2 === 0 ? 2 : 1;
  const subgroups = rowGroups * columnGroups;
  const lanes = subgroups * MATRIX_LANES;
  const groupTiles = columnTiles / columnGroups;
  const aStride = kStep + 2;
  const bStride = tileColumns + 8;
  const outStride = N + 1;
  // A load or store reaches `offset + stride * rows`, not the last element it
  // touches; an array sized to the latter is out of bounds by the extension's
  // own rule however valid every index in it is.
  const reach = (maxOffset: number, stride: number, count: number): number =>
    maxOffset + stride * count;
  const aLength = reach((GEMM_TILE_ROWS - M) * aStride, aStride, M);
  const bLength = reach((columnTiles - 1) * N, bStride, kStep);
  const outLength = reach((subgroups - 1) * M * outStride, outStride, M);
  const lines = (count: number, body: (index: number) => string): string =>
    Array.from({ length: count }, (_, index) => body(index)).join(String.fromCharCode(10));
  // Each lane carries a fixed share of both staged tiles between steps.
  if ((GEMM_TILE_ROWS * kStep) % lanes !== 0 || (kStep * tileColumns) % lanes !== 0) {
    throw new RangeError("the staged tiles must divide evenly among the lanes");
  }
  const aPerLane = (GEMM_TILE_ROWS * kStep) / lanes;
  const bPerLane = (kStep * tileColumns) / lanes;
  // The operands come from the caller's own element expressions, which name
  // `row` and `k` for the source and `k` and `column` for the weight. They are
  // read under an `if` rather than a `select` so an expression is never
  // evaluated for an index outside the operand.
  const rowFirst = shader.sourceContiguous === "row";
  // The staged slot is `[row][k]` however the fetch walked it, so the write
  // index has to be derived the same way the fetch was.
  const stagedA = (offset: number): string => rowFirst
    ? `((lane + ${offset}u) % ${GEMM_TILE_ROWS}u) * ${aStride}u + (lane + ${offset}u) / ${GEMM_TILE_ROWS}u`
    : `((lane + ${offset}u) / ${kStep}u) * ${aStride}u + (lane + ${offset}u) % ${kStep}u`;
  // A tile wholly inside the operand needs no test at all. The condition is
  // uniform across the workgroup, so the whole of it takes one branch, and
  // only the last tile of a row or of the contraction takes the checked one.
  const fetchA = (at: string, whole: boolean): string => lines(aPerLane, (i) => `  {
    let item = lane + ${i * lanes}u;
    let row = tile_row_origin + item ${rowFirst ? `% ${GEMM_TILE_ROWS}u` : `/ ${kStep}u`};
    let k = ${at} + item ${rowFirst ? `/ ${GEMM_TILE_ROWS}u` : `% ${kStep}u`};
    ${whole ? `next_a_${i} = ${shader.sourceElement};` : `var held = 0.0;
    if (row < gemm_rows && k < gemm_inner) { held = ${shader.sourceElement}; }
    next_a_${i} = held;`}
  }`);
  const kFirst = shader.weightContiguous === "k";
  const stagedB = (offset: number): string => kFirst
    ? `((lane + ${offset}u) % ${kStep}u) * ${bStride}u + (lane + ${offset}u) / ${kStep}u`
    : `((lane + ${offset}u) / ${tileColumns}u) * ${bStride}u + (lane + ${offset}u) % ${tileColumns}u`;
  const fetchB = (at: string, whole: boolean): string => lines(bPerLane, (i) => `  {
    let item = lane + ${i * lanes}u;
    let k = ${at} + item ${kFirst ? `% ${kStep}u` : `/ ${tileColumns}u`};
    let column = tile_column_origin + item ${kFirst ? `/ ${kStep}u` : `% ${tileColumns}u`};
    ${whole ? `next_b_${i} = ${shader.weightElement};` : `var held = 0.0;
    if (k < gemm_inner && column < gemm_columns) { held = ${shader.weightElement}; }
    next_b_${i} = held;`}
  }`);
  // Each subgroup stores and drains one unit-wide tile of its own, rather than
  // every subgroup staging the whole 64x128 output before any of it is read.
  // That buffer was 33 KiB of the kernel's 40 KiB, and this kernel ranks by
  // the workgroup storage it declares as plainly as the attention one does.
  const drain = (columnTile: number): string => shader.storeVector === undefined
    ? `    for (var item = in_subgroup; item < ${M * N}u; item += ${MATRIX_LANES}u) {
      let local_row = item / ${N}u;
      let local_column = item % ${N}u;
      let row = tile_row_origin + rows_at + local_row;
      let column = tile_column_origin + columns_at + ${columnTile * N}u + local_column;
      if (row < gemm_rows && column < gemm_columns) {
        let element = gemm_matrix_out[subgroup * ${M * outStride}u
          + local_row * ${outStride}u + local_column];
        ${shader.store}
      }
    }`
    : `    for (var item = in_subgroup * 4u; item < ${M * N}u; item += ${MATRIX_LANES * 4}u) {
      let local_row = item / ${N}u;
      let local_column = item % ${N}u;
      let row = tile_row_origin + rows_at + local_row;
      let column = tile_column_origin + columns_at + ${columnTile * N}u + local_column;
      if (row < gemm_rows) {
        let at = subgroup * ${M * outStride}u + local_row * ${outStride}u + local_column;
        let values = vec4<f32>(gemm_matrix_out[at], gemm_matrix_out[at + 1u],
          gemm_matrix_out[at + 2u], gemm_matrix_out[at + 3u]);
        ${shader.storeVector}
      }
    }`;
  return `enable chromium_experimental_subgroup_matrix;
enable f16;
enable subgroups;
enable subgroup_size_control;
${shader.preamble}

var<workgroup> gemm_matrix_a: array<f16, ${aLength}>;
var<workgroup> gemm_matrix_b: array<f16, ${bLength}>;
var<workgroup> gemm_matrix_out: array<f32, ${outLength}>;

// Pinned: one tile sits on one subgroup and every index counts on its width.
@compute @workgroup_size(${lanes}, 1, 1) @subgroup_size(${MATRIX_LANES})
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
  @builtin(subgroup_id) subgroup: u32,
) {
  let gemm_rows = ${shader.rows};
  let gemm_inner = ${shader.inner};
  let gemm_columns = ${shader.columns};
  let lane = local.x;
  let in_subgroup = lane % ${MATRIX_LANES}u;
  let rows_at = (subgroup / ${columnGroups}u) * ${M}u;
  let columns_at = (subgroup % ${columnGroups}u) * ${groupTiles * N}u;
${tileOrigins(tileColumns)}
${lines(groupTiles, (c) => `  var acc_${c} = subgroup_matrix_result<f32, ${N}, ${M}>();`)}

  // Software pipelined: the operands for the next step are fetched into
  // registers while the units still work on the staged ones, so a global load
  // is never what the multiplies wait for. One staging buffer is enough for
  // that; a second would double the workgroup memory to hide the same latency.
${lines(aPerLane, (i) => `  var next_a_${i} = 0.0;`)}
${lines(bPerLane, (i) => `  var next_b_${i} = 0.0;`)}
  // Whole in the operands' other index; the contraction is tested per step.
  let whole_tile = tile_row_origin + ${GEMM_TILE_ROWS}u <= gemm_rows
    && tile_column_origin + ${tileColumns}u <= gemm_columns;
  if (whole_tile && ${kStep}u <= gemm_inner) {
${fetchA("0u", true)}
${fetchB("0u", true)}
  } else {
${fetchA("0u", false)}
${fetchB("0u", false)}
  }
  for (var k0 = 0u; k0 < gemm_inner; k0 += ${kStep}u) {
${lines(aPerLane, (i) => `    gemm_matrix_a[${stagedA(i * lanes)}] = f16(next_a_${i});`)}
${lines(bPerLane, (i) => `    gemm_matrix_b[${stagedB(i * lanes)}] = f16(next_b_${i});`)}
    workgroupBarrier();
    if (whole_tile && k0 + ${2 * kStep}u <= gemm_inner) {
${fetchA(`k0 + ${kStep}u`, true)}
${fetchB(`k0 + ${kStep}u`, true)}
    } else {
${fetchA(`k0 + ${kStep}u`, false)}
${fetchB(`k0 + ${kStep}u`, false)}
    }
${lines(steps, (s) => `    {
      let left = subgroupMatrixLoad<subgroup_matrix_left<f16, ${K}, ${M}>, row_major>(
        &gemm_matrix_a, rows_at * ${aStride}u + ${s * K}u, ${aStride}u);
${lines(groupTiles, (c) => `      acc_${c} = subgroupMatrixMultiplyAccumulate(left,
        subgroupMatrixLoad<subgroup_matrix_right<f16, ${N}, ${K}>, row_major>(
          &gemm_matrix_b, ${s * K * bStride}u + columns_at + ${c * N}u, ${bStride}u), acc_${c});`)}
    }`)}
    // The next step refills the tiles this multiply just read.
    workgroupBarrier();
  }
${lines(groupTiles, (c) => `  subgroupMatrixStore<row_major>(&gemm_matrix_out, subgroup * ${M * outStride}u,
    acc_${c}, ${outStride}u);
  workgroupBarrier();
${drain(c)}
  workgroupBarrier();`)}
}`;
}

function createMatrixGemmShader(shader: TiledGemmShader, variant: GemmVariant): string {
  const unit = variant.matrix ?? MATRIX_SHAPE_F32_8;
  const { M, N, K } = unit;
  const half = unit.componentType === "f16";
  const region = MATRIX_REGION;
  const rowTiles = region / M;
  const columnTiles = region / N;
  const tileColumns = shader.tileColumns ?? GEMM_TILE_COLUMNS;
  const rowBlocks = GEMM_TILE_ROWS / region;
  const columnBlocks = tileColumns / region;
  if (!Number.isInteger(rowBlocks) || !Number.isInteger(columnBlocks)) {
    throw new RangeError(`matrix GEMM needs a tile in multiples of ${region}`);
  }
  if (!Number.isInteger(rowTiles) || !Number.isInteger(columnTiles)) {
    throw new RangeError(`matrix GEMM needs a unit shape dividing ${region}`);
  }
  const source = shader.sourceArray!;
  const weight = shader.weightArray!;
  const lines = (count: number, body: (index: number) => string): string =>
    Array.from({ length: count }, (_, index) => body(index)).join("\n");
  const base = (operand: GemmOperandArray): string => operand.base ?? "0u";
  // Where a tile is read from. With f32 components the units address the
  // storage buffers directly. With f16 they cannot: `subgroupMatrixLoad`
  // reinterprets nothing, so the array it reads must already hold the
  // component type, and the operands here are f32. The tile is therefore
  // converted once into workgroup storage and loaded from there, which is
  // also where a staged tile is read more than once — every left tile meets
  // every right tile — so the conversion is paid once per tile rather than
  // once per multiply.
  const leftArray = half ? "gemm_matrix_left" : source.array;
  const leftOffset = (r: number): string => half
    ? `${r * M * K}u`
    : `${base(source)} + (load_origin + ${r * M}u) * (${source.stride}) + k0`;
  const leftStride = half ? `${K}u` : source.stride;
  const rightArray = half ? "gemm_matrix_right" : weight.array;
  const rightOffset = (c: number): string => half
    ? `${c * N}u`
    : `${base(weight)} + k0 * (${weight.stride}) + column_origin + ${c * N}u`;
  const rightStride = half ? `${region}u` : weight.stride;
  // One lane-strided pass each, so the two staging loops touch consecutive
  // addresses. A k that runs past the contraction is written as zero rather
  // than read from the next row, which keeps a shape whose inner dimension is
  // not a multiple of the unit's K exact instead of merely unstored.
  const staging = half ? `
        for (var item = lane; item < ${region * K}u; item += ${MATRIX_LANES}u) {
          let stage_k = k0 + item % ${K}u;
          gemm_matrix_left[item] = select(f16(0.0),
            f16(${source.array}[${base(source)} + (load_origin + item / ${K}u) * (${source.stride}) + stage_k]),
            stage_k < gemm_inner);
        }
        for (var item = lane; item < ${K * region}u; item += ${MATRIX_LANES}u) {
          let stage_k = k0 + item / ${region}u;
          gemm_matrix_right[item] = select(f16(0.0),
            f16(${weight.array}[${base(weight)} + stage_k * (${weight.stride}) + column_origin + item % ${region}u]),
            stage_k < gemm_inner);
        }
        workgroupBarrier();` : "";
  return `enable chromium_experimental_subgroup_matrix;
enable subgroups;
enable subgroup_size_control;
${half ? "enable f16;\n" : ""}${shader.preamble}

var<workgroup> gemm_matrix_stage: array<f32, ${region * region}>;${half ? `
var<workgroup> gemm_matrix_left: array<f16, ${region * K}>;
var<workgroup> gemm_matrix_right: array<f16, ${K * region}>;` : ""}

// Pinned, for the same reason as the half-precision kernel above.
@compute @workgroup_size(${MATRIX_LANES}, 1, 1) @subgroup_size(${MATRIX_LANES})
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let gemm_rows = ${shader.rows};
  let gemm_inner = ${shader.inner};
  let gemm_columns = ${shader.columns};
  let lane = local.x;
${tileOrigins(tileColumns)}
  // Uniform: every invocation of the workgroup takes the same branch.
  if (gemm_rows >= ${region}u) {
    for (var row_block = 0u; row_block < ${rowBlocks}u; row_block += 1u) {
      for (var column_block = 0u; column_block < ${columnBlocks}u; column_block += 1u) {
        let row_origin = tile_row_origin + row_block * ${region}u;
        let column_origin = tile_column_origin + column_block * ${region}u;
        // Pulled back so the last tile still lies inside the source.
        let load_origin = min(row_origin, gemm_rows - ${region}u);
${lines(rowTiles, (r) => lines(columnTiles, (c) =>
    `        var acc_${r}_${c} = subgroup_matrix_result<f32, ${N}, ${M}>();`))}
        for (var k0 = 0u; k0 < gemm_inner; k0 += ${K}u) {${staging}
${lines(rowTiles, (r) => `          let left_${r} = subgroupMatrixLoad<subgroup_matrix_left<${unit.componentType}, ${K}, ${M}>, row_major>(
            &${leftArray}, ${leftOffset(r)},
            ${leftStride});`)}
${lines(columnTiles, (c) => `          let right_${c} = subgroupMatrixLoad<subgroup_matrix_right<${unit.componentType}, ${N}, ${K}>, row_major>(
            &${rightArray}, ${rightOffset(c)},
            ${rightStride});`)}
${lines(rowTiles, (r) => lines(columnTiles, (c) =>
    `          acc_${r}_${c} = subgroupMatrixMultiplyAccumulate(left_${r}, right_${c}, acc_${r}_${c});`))}${half ? `
          // The next k step refills the staging tiles this multiply just read.
          workgroupBarrier();` : ""}
        }
        workgroupBarrier();
${lines(rowTiles, (r) => lines(columnTiles, (c) =>
    `        subgroupMatrixStore<row_major>(&gemm_matrix_stage, ${r * M}u * ${region}u + ${c * N}u,
          acc_${r}_${c}, ${region}u);`))}
        workgroupBarrier();
${shader.storeVector === undefined ? `        for (var item = lane; item < ${region * region}u; item += ${MATRIX_LANES}u) {
          let row = row_origin + item / ${region}u;
          let column = column_origin + item % ${region}u;
          if (row < gemm_rows && column < gemm_columns) {
            // Where this row landed once the origin was pulled back.
            let element = gemm_matrix_stage[(row - load_origin) * ${region}u + item % ${region}u];
            ${shader.store}
          }
        }` : `        // Four adjacent columns at a time, which is how a packed
        // half-precision output is written. The caller bounds the columns
        // itself, as it does for the hand-tiled kernel.
        for (var item = lane * 4u; item < ${region * region}u; item += ${MATRIX_LANES * 4}u) {
          let local_column = item % ${region}u;
          let row = row_origin + item / ${region}u;
          let column = column_origin + local_column;
          if (row < gemm_rows) {
            let staged = (row - load_origin) * ${region}u + local_column;
            let values = vec4<f32>(gemm_matrix_stage[staged], gemm_matrix_stage[staged + 1u],
              gemm_matrix_stage[staged + 2u], gemm_matrix_stage[staged + 3u]);
            ${shader.storeVector}
          }
        }`}
        workgroupBarrier();
      }
    }
  } else {
    // Fewer rows than one region: there is nowhere to pull back to, so the
    // whole tile is computed directly. Only the smallest shapes reach this.
${shader.storeVector === undefined ? `    for (var item = lane; item < ${GEMM_TILE_ROWS * tileColumns}u; item += ${MATRIX_LANES}u) {
      let row = tile_row_origin + item / ${tileColumns}u;
      let column = tile_column_origin + item % ${tileColumns}u;
      if (row < gemm_rows && column < gemm_columns) {
        var total = 0.0;
        for (var k = 0u; k < gemm_inner; k += 1u) {
          total += ${source.array}[${base(source)} + row * (${source.stride}) + k]
            * ${weight.array}[${base(weight)} + k * (${weight.stride}) + column];
        }
        let element = total;
        ${shader.store}
      }
    }` : `    for (var item = lane * 4u; item < ${GEMM_TILE_ROWS * tileColumns}u; item += ${MATRIX_LANES * 4}u) {
      let row = tile_row_origin + item / ${tileColumns}u;
      let column = tile_column_origin + item % ${tileColumns}u;
      if (row < gemm_rows) {
        var values = vec4<f32>(0.0);
        for (var lane_column = 0u; lane_column < 4u; lane_column += 1u) {
          if (column + lane_column >= gemm_columns) { continue; }
          var total = 0.0;
          for (var k = 0u; k < gemm_inner; k += 1u) {
            total += ${source.array}[${base(source)} + row * (${source.stride}) + k]
              * ${weight.array}[${base(weight)} + k * (${weight.stride}) + column + lane_column];
          }
          values[lane_column] = total;
        }
        ${shader.storeVector}
      }
    }`}
  }
}`;
}

export function createTiledGemmShader(
  shader: TiledGemmShader, variant: GemmVariant = gemmVariant(),
): string {
  if (usesMatrixUnits(shader, variant)) {
    const unit = variant.matrix ?? MATRIX_SHAPE_F32_8;
    return unit.componentType === "f16"
      ? createMatrixGemmShaderF16(shader, unit, (variant.matrixDepth ?? 1) * unit.K)
      : createMatrixGemmShader(shader, variant);
  }
  const tileColumns = shader.tileColumns ?? GEMM_TILE_COLUMNS;
  // A caller that cannot reach the matrix units computes the same thing with
  // the hand-tiled kernel, in whatever precision the device settled on for
  // the callers that were never eligible in the first place.
  const precision = variant.precision === "matrix"
    ? variant.fallback ?? "f32" : variant.precision;
  const half = precision !== "f32";
  const scalar = half ? "f16" : "f32";
  // Only the pure arrangement keeps the running sum in half precision; the
  // other two reduce in f32 and differ in what they do inside one tile.
  const halfAccumulator = precision === "f16";
  const chunked = precision === "f16-chunked";
  const stageBytes = shader.epilogue === undefined ? 0 : (shader.stageElements ?? 2048) * 4;
  const operandBytes = (inner: number): number =>
    GEMM_TILE_ROWS * inner * (half ? 2 : 4) + inner * tileColumns * (half ? 2 : 4);
  // A depth that does not fit is stepped back rather than rejected: the
  // caller asked for a projection, not for a particular k tile, and the
  // shallower tile computes the same thing.
  const tileInner = operandBytes(variant.inner) + stageBytes <= GEMM_WORKGROUP_BYTES
    ? variant.inner : GEMM_TILE_INNER;
  if (operandBytes(tileInner) + stageBytes > GEMM_WORKGROUP_BYTES) {
    throw new RangeError(`GEMM tile ${GEMM_TILE_ROWS}x${tileColumns}k${tileInner} with `
      + `${stageBytes} staging bytes exceeds ${GEMM_WORKGROUP_BYTES} bytes of workgroup storage`);
  }
  const columnThreads = tileColumns / 4;
  const rowsPerThread = GEMM_TILE_ROWS / (GEMM_THREADS / columnThreads);
  const vectorsPerThread = rowsPerThread / 4;
  if (!Number.isInteger(columnThreads) || !Number.isInteger(rowsPerThread) || rowsPerThread % 4 !== 0) {
    throw new RangeError(`unsupported GEMM tile ${GEMM_TILE_ROWS}x${tileColumns}`);
  }
  const lines = (count: number, body: (index: number) => string): string =>
    Array.from({ length: count }, (_, index) => body(index)).join("\n");
  const items = (count: number, body: (index: number) => string): string =>
    Array.from({ length: count }, (_, index) => body(index)).join(", ");
  // Half precision accumulates under a private name and rebinds `acc{n}` to
  // the converted value once the k loop is done, so every epilogue and store
  // fragment a caller wrote against `vec4<f32>` keeps compiling unchanged.
  const register = halfAccumulator ? "gemm_acc" : "acc";
  const accumulatorScalar = halfAccumulator ? "f16" : "f32";
  // `enable` must precede every declaration, and a preamble may already carry
  // its own copy for an activation stored as f16.
  const enable = half && !shader.preamble.includes("enable f16;") ? "enable f16;\n" : "";
  return `${enable}${shader.preamble}

// Adjacent source elements are populated by different invocations. Keep them
// as scalar workgroup objects: assigning separate lanes of one vec4 concurrently
// is a data race and Metal may lower each lane assignment to a clobbering
// read-modify-write of the whole vector.
var<workgroup> gemm_source: array<${scalar}, ${GEMM_TILE_ROWS * tileInner}>;
var<workgroup> gemm_weight: array<vec4<${scalar}>, ${(tileInner * tileColumns) / 4}>;
${shader.epilogue === undefined ? "" : `var<workgroup> gemm_stage: array<f32, ${shader.stageElements ?? 2048}>;`}

@compute @workgroup_size(${GEMM_THREADS}, 1, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let gemm_rows = ${shader.rows};
  let gemm_inner = ${shader.inner};
  let gemm_columns = ${shader.columns};
  let thread = local.x;
  let column_thread = thread % ${columnThreads}u;
  let row_thread = thread / ${columnThreads}u;
${tileOrigins(tileColumns)}
  let row_origin = tile_row_origin + row_thread * ${rowsPerThread}u;
  let column_origin = tile_column_origin;
  let tile_column = column_origin + column_thread * 4u;
${lines(rowsPerThread, (row) => `  var ${register}${row} = vec4<${accumulatorScalar}>(0.0);`)}

  // A tile wholly inside the operands needs no test at all, and only the last
  // tile of a row or a column is not. The condition is uniform across the
  // workgroup, so the whole of it takes one branch; the contraction is asked
  // per step, since only its last step is short.
  let whole_tile = tile_row_origin + ${GEMM_TILE_ROWS}u <= gemm_rows
    && column_origin + ${tileColumns}u <= gemm_columns;
  for (var k0 = 0u; k0 < gemm_inner; k0 += ${tileInner}u) {
    let whole_step = whole_tile && k0 + ${tileInner}u <= gemm_inner;
    for (var item = thread; item < ${GEMM_TILE_ROWS * tileInner}u; item += ${GEMM_THREADS}u) {
      let load_row = item / ${tileInner}u;
      let k = k0 + (item % ${tileInner}u);
      let row = group.y * ${GEMM_TILE_ROWS}u + load_row;
      var element = 0.0;
      if (whole_step) { element = ${shader.sourceElement}; }
      else if (row < gemm_rows && k < gemm_inner) { element = ${shader.sourceElement}; }
      let slot = (item % ${tileInner}u) * ${GEMM_TILE_ROWS}u + load_row;
      gemm_source[slot] = ${half ? "f16(element)" : "element"};
    }
    for (var item = thread; item < ${(tileInner * tileColumns) / 4}u; item += ${GEMM_THREADS}u) {
      let k = k0 + item / ${columnThreads}u;
      let load_column = column_origin + (item % ${columnThreads}u) * 4u;
      var loaded = vec4<f32>(0.0);
      if (whole_step) {
${lines(4, (lane) => `        {
          let column = load_column + ${lane}u;
          loaded[${lane}u] = ${shader.weightElement};
        }`)}
      } else if (k < gemm_inner) {
${lines(4, (lane) => `        {
          let column = load_column + ${lane}u;
          if (column < gemm_columns) { loaded[${lane}u] = ${shader.weightElement}; }
        }`)}
      }
      gemm_weight[item] = ${half ? "vec4<f16>(loaded)" : "loaded"};
    }
    workgroupBarrier();
${chunked ? `${lines(rowsPerThread, (row) => `    var chunk${row} = vec4<f16>(0.0);`)}\n` : ""}    for (var step = 0u; step < ${tileInner}u; step += 1u) {
      let w = gemm_weight[step * ${columnThreads}u + column_thread];
      let a_base = step * ${GEMM_TILE_ROWS}u + row_thread * ${rowsPerThread}u;
${lines(vectorsPerThread, (vector) => `      let a${vector} = vec4<${scalar}>(${items(4,
    (lane) => `gemm_source[a_base + ${vector * 4 + lane}u]`)});`)}
${lines(rowsPerThread, (row) => {
    const product = `a${Math.floor(row / 4)}[${row % 4}u] * w`;
    if (chunked) return `      chunk${row} += ${product};`;
    if (precision === "f16-mixed") return `      acc${row} += vec4<f32>(${product});`;
    return `      ${register}${row} += ${product};`;
  })}
    }${chunked ? `\n${lines(rowsPerThread, (row) => `    acc${row} += vec4<f32>(chunk${row});`)}` : ""}
    workgroupBarrier();
  }${halfAccumulator ? `\n${lines(rowsPerThread, (row) => `  let acc${row} = vec4<f32>(gemm_acc${row});`)}` : ""}

${shader.epilogue !== undefined ? shader.epilogue : lines(rowsPerThread, (index) => `
  {
    let row = row_origin + ${index}u;
    if (row < gemm_rows) {
${shader.storeVector !== undefined ? `      let column = tile_column;
      let values = acc${index};
      ${shader.storeVector}` : lines(4, (lane) => `      {
        let column = tile_column + ${lane}u;
        if (column < gemm_columns) {
          let element = acc${index}[${lane}u];
          ${shader.store}
        }
      }`)}
    }
  }`)}
}`;
}

/**
 * The unit shape an attention kernel should use, or nothing.
 *
 * Attention needs a square tile: the same M carries the queries through both
 * multiplies, and N must divide both the staged key tile and the head width.
 * An f32 configuration would be exact and there is none on the parts that
 * offer these units at all, so an f16 one with an f32 accumulator is what this
 * looks for — the accumulator is what keeps the long reduction safe.
 */
export function attentionMatrixConfig(
  device: GPUDevice,
  configs: readonly { componentType: string; resultComponentType: string;
    M: number; N: number; K: number }[],
  queryTile: number,
): MatrixUnitShape | undefined {
  if (!device.features.has("shader-f16" as GPUFeatureName)) return undefined;
  const usable = configs.filter((config) =>
    config.resultComponentType === "f32" && config.componentType === "f16"
    && config.M === queryTile && config.N === config.M && config.K === config.M);
  const chosen = usable[0];
  return chosen === undefined ? undefined : {
    componentType: "f16", M: chosen.M, N: chosen.N, K: chosen.K,
  };
}
