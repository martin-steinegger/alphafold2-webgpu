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
  readonly precision: "f32" | "f16";
  readonly inner: 8 | 16;
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

export function createTiledGemmShader(
  shader: TiledGemmShader, variant: GemmVariant = gemmVariant(),
): string {
  const tileColumns = shader.tileColumns ?? GEMM_TILE_COLUMNS;
  const half = variant.precision === "f16";
  const scalar = half ? "f16" : "f32";
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
  const register = half ? "gemm_acc" : "acc";
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
${lines(rowsPerThread, (row) => `  var ${register}${row} = vec4<${scalar}>(0.0);`)}

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
    for (var step = 0u; step < ${tileInner}u; step += 1u) {
      let w = gemm_weight[step * ${columnThreads}u + column_thread];
      let a_base = step * ${GEMM_TILE_ROWS}u + row_thread * ${rowsPerThread}u;
${lines(vectorsPerThread, (vector) => `      let a${vector} = vec4<${scalar}>(${items(4,
    (lane) => `gemm_source[a_base + ${vector * 4 + lane}u]`)});`)}
${lines(rowsPerThread, (row) => `      ${register}${row} += a${Math.floor(row / 4)}[${row % 4}u] * w;`)}
    }
    workgroupBarrier();
  }${half ? `\n${lines(rowsPerThread, (row) => `  let acc${row} = vec4<f32>(gemm_acc${row});`)}` : ""}

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
