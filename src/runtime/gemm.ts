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

/** The matrix shape Apple offers, the region one subgroup owns, and its width. */
const MATRIX_TILE = 8;
const MATRIX_REGION = 32;
const MATRIX_LANES = 32;

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
   * store one element at a time, and `inner` does not apply to it — the units
   * fix the contraction step at 8.
   */
  readonly precision: "f32" | "f16" | "f16-mixed" | "f16-chunked" | "matrix";
  readonly inner: 8 | 16;
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
}

export interface GemmOperandArray {
  /** Name of an `array<f32>` binding in the preamble. */
  readonly array: string;
  /** Expression for the element index the operand starts at. Defaults to zero. */
  readonly base?: string;
  /** Expression for the distance in elements between consecutive rows. */
  readonly stride: string;
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
  return [Math.ceil(columns / tileColumns), Math.ceil(rows / GEMM_TILE_ROWS)];
}

/** Whether this shader and this variant can use the hardware matrix units. */
export function usesMatrixUnits(shader: TiledGemmShader, variant: GemmVariant): boolean {
  return variant.precision === "matrix"
    && shader.sourceArray !== undefined && shader.weightArray !== undefined
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
function createMatrixGemmShader(shader: TiledGemmShader, variant: GemmVariant): string {
  const size = MATRIX_TILE;
  const region = MATRIX_REGION;
  const tiles = region / size;
  const tileColumns = shader.tileColumns ?? GEMM_TILE_COLUMNS;
  const rowBlocks = GEMM_TILE_ROWS / region;
  const columnBlocks = tileColumns / region;
  if (!Number.isInteger(rowBlocks) || !Number.isInteger(columnBlocks)) {
    throw new RangeError(`matrix GEMM needs a tile in multiples of ${region}`);
  }
  const source = shader.sourceArray!;
  const weight = shader.weightArray!;
  const lines = (count: number, body: (index: number) => string): string =>
    Array.from({ length: count }, (_, index) => body(index)).join("\n");
  const base = (operand: GemmOperandArray): string => operand.base ?? "0u";
  return `enable chromium_experimental_subgroup_matrix;
${shader.preamble}

var<workgroup> gemm_matrix_stage: array<f32, ${region * region}>;

@compute @workgroup_size(${MATRIX_LANES}, 1, 1)
fn main(
  @builtin(local_invocation_id) local: vec3<u32>,
  @builtin(workgroup_id) group: vec3<u32>,
) {
  let gemm_rows = ${shader.rows};
  let gemm_inner = ${shader.inner};
  let gemm_columns = ${shader.columns};
  let lane = local.x;
  let tile_row_origin = group.y * ${GEMM_TILE_ROWS}u;
  let tile_column_origin = group.x * ${tileColumns}u;
  // Uniform: every invocation of the workgroup takes the same branch.
  if (gemm_rows >= ${region}u) {
    for (var row_block = 0u; row_block < ${rowBlocks}u; row_block += 1u) {
      for (var column_block = 0u; column_block < ${columnBlocks}u; column_block += 1u) {
        let row_origin = tile_row_origin + row_block * ${region}u;
        let column_origin = tile_column_origin + column_block * ${region}u;
        // Pulled back so the last tile still lies inside the source.
        let load_origin = min(row_origin, gemm_rows - ${region}u);
${lines(tiles, (r) => lines(tiles, (c) =>
    `        var acc_${r}_${c} = subgroup_matrix_result<f32, ${size}, ${size}>();`))}
        for (var k0 = 0u; k0 < gemm_inner; k0 += ${size}u) {
${lines(tiles, (r) => `          let left_${r} = subgroupMatrixLoad<subgroup_matrix_left<f32, ${size}, ${size}>>(
            &${source.array}, ${base(source)} + (load_origin + ${r * size}u) * (${source.stride}) + k0,
            false, ${source.stride});`)}
${lines(tiles, (c) => `          let right_${c} = subgroupMatrixLoad<subgroup_matrix_right<f32, ${size}, ${size}>>(
            &${weight.array}, ${base(weight)} + k0 * (${weight.stride}) + column_origin + ${c * size}u,
            false, ${weight.stride});`)}
${lines(tiles, (r) => lines(tiles, (c) =>
    `          acc_${r}_${c} = subgroupMatrixMultiplyAccumulate(left_${r}, right_${c}, acc_${r}_${c});`))}
        }
        workgroupBarrier();
${lines(tiles, (r) => lines(tiles, (c) =>
    `        subgroupMatrixStore(&gemm_matrix_stage, ${r * size}u * ${region}u + ${c * size}u,
          acc_${r}_${c}, false, ${region}u);`))}
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
  if (usesMatrixUnits(shader, variant)) return createMatrixGemmShader(shader, variant);
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
  let tile_row_origin = group.y * ${GEMM_TILE_ROWS}u;
  let row_origin = tile_row_origin + row_thread * ${rowsPerThread}u;
  let column_origin = group.x * ${tileColumns}u;
  let tile_column = column_origin + column_thread * 4u;
${lines(rowsPerThread, (row) => `  var ${register}${row} = vec4<${accumulatorScalar}>(0.0);`)}

  for (var k0 = 0u; k0 < gemm_inner; k0 += ${tileInner}u) {
    for (var item = thread; item < ${GEMM_TILE_ROWS * tileInner}u; item += ${GEMM_THREADS}u) {
      let load_row = item / ${tileInner}u;
      let k = k0 + (item % ${tileInner}u);
      let row = group.y * ${GEMM_TILE_ROWS}u + load_row;
      var element = 0.0;
      if (row < gemm_rows && k < gemm_inner) { element = ${shader.sourceElement}; }
      let slot = (item % ${tileInner}u) * ${GEMM_TILE_ROWS}u + load_row;
      gemm_source[slot] = ${half ? "f16(element)" : "element"};
    }
    for (var item = thread; item < ${(tileInner * tileColumns) / 4}u; item += ${GEMM_THREADS}u) {
      let k = k0 + item / ${columnThreads}u;
      let load_column = column_origin + (item % ${columnThreads}u) * 4u;
      var loaded = vec4<f32>(0.0);
      if (k < gemm_inner) {
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
