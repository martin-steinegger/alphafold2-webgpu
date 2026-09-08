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
  // without this. Without it there is no `shader-f16`, so no half-precision
  // projection and no matrix units.
  "vulkan_enable_f16_on_nvidia",
  // `chromium-experimental-subgroup-matrix` is an experimental feature, and
  // Dawn hides experimental features behind this.
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
 * is only sound while no kernel relies on the clamp. None of them do — the
 * loads that used to sit inside a `select`, which evaluates both of its arms,
 * are under an `if` now, and the pair bias carries a row of slack for the
 * vector read at its end — but that is a property of the shaders that has to
 * be kept true, not one the platform enforces.
 */
const UNCLAMPED = "disable_robustness";

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
