/**
 * Dawn instance flags the native entry points ask for.
 *
 * Dawn takes its toggles when the instance is made, so a caller that reaches
 * the GPU through webgpu's create chooses them and a browser does not get
 * to. Left to its defaults Dawn does not expose shader-f16 on Nvidia and
 * does not admit an experimental extension, so the half-precision kernels and
 * the matrix units are both out of reach — which is most of this model's
 * speed. The entry points therefore ask for them rather than each remembering
 * a string.
 */

/** Toggles every native entry point wants. */
const REQUIRED = [
  // Nvidia's Vulkan driver reports f16 support that Dawn does not expose
  // without this, and without shader-f16 there is no half-precision
  // projection and no f16 matrix configuration to select. It is Nvidia-only in
  // effect as well as in name: an instance with no toggles at all reports
  // shader-f16 false on a Blackwell and true on an RDNA 3.5 part.
  "vulkan_enable_f16_on_nvidia",
  // chromium-experimental-subgroup-matrix is an experimental feature, and
  // Dawn hides experimental features behind this. On every vendor: a bare
  // instance reports no matrix configurations on either of the two above.
  "allow_unsafe_apis",
] as const;

/**
 * The bounds clamp WebGPU puts on every dynamically indexed array access.
 *
 * Tint clamps every workgroup, private and function access unconditionally —
 * there is no way to relax it per address space — and wraps every
 * subgroupMatrixLoad and subgroupMatrixStore on workgroup memory in
 * saturating arithmetic besides. These kernels are little but dynamic indexing
 * into workgroup arrays, and dropping the clamp is worth 11% of a recycle:
 * 20.48 s against 18.23 s on a 1,650-residue dimer, with every confidence
 * identical.
 *
 * It is off by default because it is a promise, not a setting. Dropping the
 * clamp makes any access outside an array undefined rather than clamped, so it
 * is only sound while no kernel relies on the clamp. None of them do — a
 * guarded load sits under an if rather than inside a select, which
 * evaluates both of its arms, and the pair bias carries a row of slack for the
 * vector read at its end — but that is a property of the shaders that has to
 * be kept true, not one the platform enforces.
 */
const UNCLAMPED = "disable_robustness";

/**
 * Scratch budget scales worth trying, widest first.
 *
 * The dial saturates for a short chain: an 825-residue monomer reads 4213 ms a
 * recycle at sixteen, 5398 at thirty-two and 5461 at sixty-four, for twice the
 * resident memory. Sixteen is where it stops paying there.
 */
export const SCRATCH_BUDGET_SCALES: readonly number[] = [16, 8, 4, 2, 1];

/** Above this many residues a wider budget starts paying again. See below. */
export const WIDE_SCRATCH_RESIDUES = 1000;

/** The ladder with the wide rungs on it, for a chain long enough to want them. */
export const WIDE_SCRATCH_BUDGET_SCALES: readonly number[] = [64, 32, 16, 8, 4, 2, 1];

/**
 * The scales worth trying at this length, widest first.
 *
 * Sixteen is not the ceiling it was once measured to be. The triangle
 * contraction is a GEMM whose rows are the block the budget sizes, and at a
 * long chain sixteen leaves that block at sixty-four rows: a 3,300-residue
 * tetramer then makes fifty-two passes over the operand it contracts against
 * and runs the kernel at 41.5 TFLOP/s, against 93 at sixty-four. Short chains
 * do not have that problem, because the same budget already buys them a wide
 * block.
 *
 * Measured a recycle, one idle card, this tree:
 *
 *   scale        825        1,650       3,300
 *       16   **4213**       12500       74788
 *       32      5398        13544          --
 *       64      5461     **11897**   **62474**
 *
 * The optimum is not a smooth function of the block rows -- 1,650 is fastest
 * at 576 rows and slowest at 256 -- so this is a threshold fitted to three
 * lengths rather than a model, and it is written down as such. What is not
 * fitted is the direction: past a thousand residues the widest rung has won
 * every time it was tried, and below it has lost every time.
 *
 * The estimate still decides: fitScratchBudgetScale costs each rung against
 * the host's memory and takes the widest that fits, so a card too small for
 * sixty-four simply gets the next one down.
 */
export function scratchBudgetScalesFor(residues: number): readonly number[] {
  return residues > WIDE_SCRATCH_RESIDUES ? WIDE_SCRATCH_BUDGET_SCALES : SCRATCH_BUDGET_SCALES;
}

/**
 * The widest scratch budget whose estimated peak fits budgetBytes.
 *
 * WebGPU does not report how much memory a device has — adapter.info carries
 * no heaps, and maxBufferSize is the API's theoretical maximum rather than
 * the card's — so the budget comes from the caller, who knows. What does not
 * need guessing is the scale: the planner already estimates a prediction's
 * peak, so each scale is costed and the widest affordable one is taken.
 */
export function fitScratchBudgetScale(
  estimatePeakBytes: (scale: number) => number,
  budgetBytes: number,
  scales: readonly number[] = SCRATCH_BUDGET_SCALES,
): number {
  for (const scale of scales) {
    if (estimatePeakBytes(scale) <= budgetBytes) return scale;
  }
  return scales[scales.length - 1] ?? 1;
}

export interface DawnInstanceOptions {
  /** Drop the bounds clamp. See UNCLAMPED; measure before trusting it. */
  readonly unclamped?: boolean;
  /** Report timestamps unrounded. Dawn otherwise quantizes them to ~65.5 us. */
  readonly exactTimestamps?: boolean;
}

/**
 * Warns when a card was asked for and nothing pointed the loader at it.
 *
 * CUDA_VISIBLE_DEVICES does nothing on its own: selectGpu turns it into the
 * DRI_PRIME tag the Vulkan loader reads, and a harness that forgets to call it
 * ignores the variable in silence. One that did cost a whole investigation,
 * measuring a 1,650-residue dimer at 37.2 s a recycle against a real 12.0
 * because every run landed on a card someone else was using. The check is here
 * rather than in a harness because this is the one call every native entry
 * point already has to make.
 */
function warnUnselectedGpu(): void {
  const asked = process.env?.CUDA_VISIBLE_DEVICES;
  if (asked === undefined || asked === "" || process.env?.DRI_PRIME !== undefined) return;
  console.error(`CUDA_VISIBLE_DEVICES=${asked} but DRI_PRIME is unset, so the loader will pick the`
    + " card, not you. Call selectGpu() from tools/native-device.js before creating the instance.");
}

/** Flags for create, which takes them as name=value strings. */
export function dawnInstanceFlags(options: DawnInstanceOptions = {}): string[] {
  warnUnselectedGpu();
  const enabled = [...REQUIRED, ...(options.unclamped === true ? [UNCLAMPED] : [])];
  return [
    `enable-dawn-features=${enabled.join(",")}`,
    // Quantized timestamps round a query to about 65.5 microseconds, which is
    // a whole dispatch of a small kernel. Only a profile needs them exact.
    ...(options.exactTimestamps === true ? ["disable-dawn-features=timestamp_quantization"] : []),
  ];
}
