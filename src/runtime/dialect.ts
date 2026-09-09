/**
 * What this implementation's shader language accepts, where implementations
 * disagree.
 *
 * WGSL is one language but not every implementation is at the same point in
 * it, and the differences are in directives and in the spelling of the matrix
 * types rather than in the code around them. Dawn requires "enable subgroups;"
 * before a subgroup builtin and rejects the builtin without it; naga
 * implements the builtins and not the directive, so the same line is a parse
 * error there. Both reach the hardware matrix units, under different names.
 *
 * Settled by compiling a probe rather than by naming implementations, so one
 * that adds a directive tomorrow is picked up without a release here.
 *
 * Kept per device rather than in module state. A process can hold more than
 * one: test/support/gpu-instance.ts caches an instance per flag set, and
 * test:gpu:portable runs against llvmpipe and against a card whose matrix
 * support differs, so a single installed spelling would be whichever was
 * settled last and nothing would report it. The module-scope projection
 * variant in gemm.ts predates that and is not the shape to copy here.
 */

/** How one implementation spells the cooperative-matrix types and builtins. */
export interface MatrixSpelling {
  /**
   * Every directive a matrix kernel needs, "enable f16;" aside. More than the
   * matrix extension itself, because the surrounding subgroup directives are
   * just as implementation-specific.
   */
  readonly prelude: string;
  /** Type of a left operand: columns deep over rows. */
  left(type: string, columns: number, rows: number): string;
  right(type: string, columns: number, rows: number): string;
  result(type: string, columns: number, rows: number): string;
  /** A zeroed accumulator. */
  zero(type: string, columns: number, rows: number): string;
  /**
   * Reads one operand tile. "major" is how the source is laid out; "col"
   * transposes on the way in, which costs nothing on either implementation.
   */
  load(operand: string, array: string, offset: string, stride: string,
    major?: "row" | "col"): string;
  /** Writes an accumulator out, row-major. */
  store(array: string, offset: string, value: string, stride: string): string;
  /** acc + left * right. */
  multiplyAccumulate(left: string, right: string, accumulator: string): string;
  /**
   * The entry point's subgroup-width attribute, or empty where there is none.
   * These kernels lay a tile across exactly one subgroup, so a caller without
   * the attribute must refuse the kernel unless the adapter already reports a
   * fixed width.
   */
  subgroupSize(lanes: number): string;
}

export interface Dialect {
  /** "enable subgroups;", or empty where the builtins need no directive. */
  readonly subgroupEnable: string;
  /** Undefined on a device reporting no matrix units. */
  readonly matrix: MatrixSpelling | undefined;
}

export const DAWN_MATRIX: MatrixSpelling = {
  prelude: "enable chromium_experimental_subgroup_matrix;\n"
    + "enable subgroups;\nenable subgroup_size_control;\n",
  subgroupSize: (lanes) => ` @subgroup_size(${lanes})`,
  left: (type, columns, rows) => `subgroup_matrix_left<${type}, ${columns}, ${rows}>`,
  right: (type, columns, rows) => `subgroup_matrix_right<${type}, ${columns}, ${rows}>`,
  result: (type, columns, rows) => `subgroup_matrix_result<${type}, ${columns}, ${rows}>`,
  zero: (type, columns, rows) => `subgroup_matrix_result<${type}, ${columns}, ${rows}>()`,
  load: (operand, array, offset, stride, major = "row") =>
    `subgroupMatrixLoad<${operand}, ${major}_major>(&${array}, ${offset}, ${stride})`,
  store: (array, offset, value, stride) =>
    `subgroupMatrixStore<row_major>(&${array}, ${offset}, ${value}, ${stride})`,
  multiplyAccumulate: (left, right, accumulator) =>
    `subgroupMatrixMultiplyAccumulate(${left}, ${right}, ${accumulator})`,
};

/**
 * wgpu's spelling, which differs beyond the names in three ways.
 *
 * The pointer is to the element rather than to the array with an offset beside
 * it, so an offset becomes an index.
 *
 * The majorness is in the builtin's name, and the name is the opposite of what
 * it reads as: coopLoadT is the ROW-major load and plain coopLoad is
 * column-major, which naga lowers to RowMajorKHR and ColumnMajorKHR. A
 * reversed mapping compiles, validates and computes a transposed product, so
 * it is not left to the reader of the name.
 *
 * naga predeclares square generators only. Every shape here is square, but a
 * device reporting one of the oblong configurations wgpu enumerates would
 * reach a type name naga does not know, so the shape is refused here rather
 * than at compile time inside a kernel.
 */
export const WGPU_MATRIX: MatrixSpelling = {
  prelude: "enable wgpu_cooperative_matrix;\n",
  subgroupSize: () => "",
  left: (type, columns, rows) => coopMat(type, columns, rows, "A"),
  right: (type, columns, rows) => coopMat(type, columns, rows, "B"),
  result: (type, columns, rows) => coopMat(type, columns, rows, "C"),
  zero: (type, columns, rows) => `${coopMat(type, columns, rows, "C")}()`,
  load: (operand, array, offset, stride, major = "row") =>
    `${major === "row" ? "coopLoadT" : "coopLoad"}<${operand}>((&${array}[${offset}]), ${stride})`,
  store: (array, offset, value, stride) =>
    `coopStoreT(${value}, (&${array}[${offset}]), ${stride})`,
  multiplyAccumulate: (left, right, accumulator) =>
    `coopMultiplyAdd(${left}, ${right}, ${accumulator})`,
};

function coopMat(type: string, columns: number, rows: number, role: string): string {
  if (columns !== rows) {
    throw new RangeError(`wgpu has no coop_mat${columns}x${rows}: the shape must be square`);
  }
  return `coop_mat${columns}x${rows}<${type}, ${role}>`;
}

const DIRECTIVE_FREE: Dialect = { subgroupEnable: "", matrix: undefined };
const settled = new WeakMap<GPUDevice, Dialect>();

/**
 * Compiles both probes and remembers the answer for this device.
 *
 * Call before any kernel source is generated. A device without subgroups never
 * reaches a kernel that would ask, and one without matrix units gets an
 * undefined matrix half.
 */
export async function calibrateDialect(device: GPUDevice): Promise<Dialect> {
  const subgroupEnable = device.features.has("subgroups")
    ? await firstAccepted(device, ["enable subgroups;\n", ""]) : "";
  const matrix = await settleMatrix(device);
  const answer: Dialect = { subgroupEnable, matrix };
  settled.set(device, answer);
  if (matrix !== undefined) lastMatrix = matrix;
  return answer;
}

/** What calibration settled, or the directive-free form. */
export function dialect(device: GPUDevice | undefined): Dialect {
  return (device === undefined ? undefined : settled.get(device)) ?? DIRECTIVE_FREE;
}

/**
 * The matrix half.
 *
 * The throw is a backstop rather than a device condition: the selection gates
 * refuse a matrix kernel on a device without the units, so reaching it means a
 * caller bypassed one.
 */
export function matrixSpelling(device: GPUDevice): MatrixSpelling {
  const held = dialect(device).matrix;
  if (held === undefined) {
    throw new RangeError("this device reports no subgroup matrix configurations");
  }
  return held;
}

/**
 * The spelling for a caller that has no device in reach.
 *
 * Exactly as global as gemmVariant() in gemm.ts, which decides whether a
 * matrix kernel is built at all, and no more correct: a second device with
 * different matrix support is already misserved by that variant. The per-device
 * store above is the truth, and a builder that can reach a device should take
 * it from there. This exists so the dozen shader builders between a device and
 * a projection need not each grow a parameter, and it should go when the
 * variant becomes per-device.
 */
export function lastCalibratedMatrix(): MatrixSpelling | undefined { return lastMatrix; }

let lastMatrix: MatrixSpelling | undefined;

/** Forces both halves, for a test that needs one deliberately. */
export function presetDialect(device: GPUDevice, forced: Dialect): void {
  settled.set(device, forced);
  if (forced.matrix !== undefined) lastMatrix = forced.matrix;
}

async function settleMatrix(device: GPUDevice): Promise<MatrixSpelling | undefined> {
  // The directive alone separates them, and it needs no shape: a device that
  // cannot accept a spelling's prelude cannot run its kernels either. Probing
  // a whole matrix kernel would need a shape the units implement, which is on
  // the adapter rather than the device and is not always square.
  for (const spelling of [DAWN_MATRIX, WGPU_MATRIX]) {
    if (await compiles(device, `${spelling.prelude}@compute @workgroup_size(1)\nfn main() {}\n`)) {
      return spelling;
    }
  }
  return undefined;
}

async function firstAccepted(device: GPUDevice, candidates: readonly string[]): Promise<string> {
  for (const directive of candidates) {
    // A builtin as well as the directive, since an implementation that ignores
    // an unknown directive would otherwise accept every candidate.
    const source = `${directive}@compute @workgroup_size(64)\n`
      + "fn main(@builtin(subgroup_size) width: u32) { _ = width; }\n";
    if (await compiles(device, source)) return directive;
  }
  return "";
}

async function compiles(device: GPUDevice, code: string): Promise<boolean> {
  device.pushErrorScope("validation");
  device.createShaderModule({ code });
  return await device.popErrorScope() === null;
}
