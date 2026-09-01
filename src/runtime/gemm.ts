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

export function createTiledGemmShader(shader: TiledGemmShader): string {
  const tileColumns = shader.tileColumns ?? GEMM_TILE_COLUMNS;
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
  return `${shader.preamble}

// Adjacent source elements are populated by different invocations. Keep them
// as scalar workgroup objects: assigning separate lanes of one vec4 concurrently
// is a data race and Metal may lower each lane assignment to a clobbering
// read-modify-write of the whole vector.
var<workgroup> gemm_source: array<f32, ${GEMM_TILE_ROWS * GEMM_TILE_INNER}>;
var<workgroup> gemm_weight: array<vec4<f32>, ${(GEMM_TILE_INNER * tileColumns) / 4}>;
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
${lines(rowsPerThread, (row) => `  var acc${row} = vec4<f32>(0.0);`)}

  for (var k0 = 0u; k0 < gemm_inner; k0 += ${GEMM_TILE_INNER}u) {
    for (var item = thread; item < ${GEMM_TILE_ROWS * GEMM_TILE_INNER}u; item += ${GEMM_THREADS}u) {
      let load_row = item / ${GEMM_TILE_INNER}u;
      let k = k0 + (item % ${GEMM_TILE_INNER}u);
      let row = group.y * ${GEMM_TILE_ROWS}u + load_row;
      var element = 0.0;
      if (row < gemm_rows && k < gemm_inner) { element = ${shader.sourceElement}; }
      let slot = (item % ${GEMM_TILE_INNER}u) * ${GEMM_TILE_ROWS}u + load_row;
      gemm_source[slot] = element;
    }
    for (var item = thread; item < ${(GEMM_TILE_INNER * tileColumns) / 4}u; item += ${GEMM_THREADS}u) {
      let k = k0 + item / ${columnThreads}u;
      let load_column = column_origin + (item % ${columnThreads}u) * 4u;
      var loaded = vec4<f32>(0.0);
      if (k < gemm_inner) {
${lines(4, (lane) => `        {
          let column = load_column + ${lane}u;
          if (column < gemm_columns) { loaded[${lane}u] = ${shader.weightElement}; }
        }`)}
      }
      gemm_weight[item] = loaded;
    }
    workgroupBarrier();
    for (var step = 0u; step < ${GEMM_TILE_INNER}u; step += 1u) {
      let w = gemm_weight[step * ${columnThreads}u + column_thread];
      let a_base = step * ${GEMM_TILE_ROWS}u + row_thread * ${rowsPerThread}u;
${lines(vectorsPerThread, (vector) => `      let a${vector} = vec4<f32>(${items(4,
    (lane) => `gemm_source[a_base + ${vector * 4 + lane}u]`)});`)}
${lines(rowsPerThread, (row) => `      acc${row} += a${Math.floor(row / 4)}[${row % 4}u] * w;`)}
    }
    workgroupBarrier();
  }

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
