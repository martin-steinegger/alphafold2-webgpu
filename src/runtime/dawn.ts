/**
 * Dawn instance flags the native entry points ask for.
 *
 * Dawn takes its toggles when the instance is made, so a caller that reaches
 * the GPU through `webgpu`'s `create` chooses them and a browser does not get
 * to. Left to its defaults Dawn does not expose `shader-f16` on Nvidia and
 * does not admit an experimental extension, so the half-precision kernels and
 * the matrix units are both out of reach — which is most of this model's
 * speed. The entry points therefore ask for them rather than each remembering
 * a string.
 */

/** Toggles every native entry point wants. */
const REQUIRED = [
  // Nvidia's Vulkan driver reports f16 support that Dawn does not expose
  // without this, and without `shader-f16` there is no half-precision
  // projection and no f16 matrix configuration to select. It is Nvidia-only in
  // effect as well as in name: an instance with no toggles at all reports
  // `shader-f16` false on a Blackwell and true on an RDNA 3.5 part.
  "vulkan_enable_f16_on_nvidia",
  // `chromium-experimental-subgroup-matrix` is an experimental feature, and
  // Dawn hides experimental features behind this. On every vendor: a bare
  // instance reports no matrix configurations on either of the two above.
  "allow_unsafe_apis",
] as const;

/**
 * The bounds clamp WebGPU puts on every dynamically indexed array access.
 *
 * Tint clamps every workgroup, private and function access unconditionally —
 * there is no way to relax it per address space — and wraps every
 * `subgroupMatrixLoad` and `subgroupMatrixStore` on workgroup memory in
 * saturating arithmetic besides. These kernels are little but dynamic indexing
 * into workgroup arrays, and dropping the clamp is worth 11% of a recycle:
 * 20.48 s against 18.23 s on a 1,650-residue dimer, with every confidence
 * identical.
 *
 * It is off by default because it is a promise, not a setting. Dropping the
 * clamp makes any access outside an array undefined rather than clamped, so it
 * is only sound while no kernel relies on the clamp. None of them do — a
 * guarded load sits under an `if` rather than inside a `select`, which
 * evaluates both of its arms, and the pair bias carries a row of slack for the
 * vector read at its end — but that is a property of the shaders that has to
 * be kept true, not one the platform enforces.
 */
const UNCLAMPED = "disable_robustness";

/**
 * Scratch budget scales worth trying, widest first.
 *
 * The dial saturates: a 1,650-residue dimer reads 27.3 s a recycle at one,
 * 20.7 s at sixteen and 20.4 s at thirty-two, for another 950 MiB. Sixteen is
 * where it stops paying, so nothing above it is offered.
 */
export const SCRATCH_BUDGET_SCALES: readonly number[] = [16, 8, 4, 2, 1];

/**
 * The widest scratch budget whose estimated peak fits `budgetBytes`.
 *
 * WebGPU does not report how much memory a device has — `adapter.info` carries
 * no heaps, and `maxBufferSize` is the API's theoretical maximum rather than
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
  /** Drop the bounds clamp. See `UNCLAMPED`; measure before trusting it. */
  readonly unclamped?: boolean;
  /** Report timestamps unrounded. Dawn otherwise quantizes them to ~65.5 us. */
  readonly exactTimestamps?: boolean;
}

/** Flags for `create`, which takes them as `name=value` strings. */
export function dawnInstanceFlags(options: DawnInstanceOptions = {}): string[] {
  const enabled = [...REQUIRED, ...(options.unclamped === true ? [UNCLAMPED] : [])];
  return [
    `enable-dawn-features=${enabled.join(",")}`,
    // Quantized timestamps round a query to about 65.5 microseconds, which is
    // a whole dispatch of a small kernel. Only a profile needs them exact.
    ...(options.exactTimestamps === true ? ["disable-dawn-features=timestamp_quantization"] : []),
  ];
}
