/**
 * How large the bounded scratch windows are allowed to be on this device.
 *
 * Every operation that cannot hold its whole intermediate at once — the
 * outer-product contraction, the attention batch window, the transition
 * chunk, the triangle block, the PAE logits — bounds itself against a byte
 * budget and dispatches once per window. The budgets are written for the
 * device the browser hands out: a few megabytes each, so a phone and a laptop
 * both finish rather than failing to allocate.
 *
 * On a server accelerator those same budgets are the cost. At 1,650 residues
 * the outer-product intermediate is 6.45 MiB per residue, so a 16 MiB budget
 * blocks the contraction two residues at a time and issues 825 dispatches for
 * work the device could do in twenty-one. Measured on one Multimer recycle,
 * raising every budget together took a block from over 2,048 dispatches to
 * 307 and the recycle from 53.3 to 39.0 seconds, with the prediction
 * unchanged to every reported digit.
 *
 * This is deliberately not chosen automatically. WebGPU reports no total
 * device memory, only per-binding and per-buffer limits, so an adapter that
 * permits a 2 GiB binding may still be an 8 GiB laptop part that a scaled
 * scratch would push into an out-of-memory failure. A caller that knows it
 * owns a large accelerator says so; every other caller keeps the budgets the
 * browser was tuned for.
 *
 * The scale is process-wide and installed before a prediction starts, exactly
 * as setGemmVariant is, because the same budgets decide both what
 * monomerDeviceRequirements estimates and what the run then allocates.
 * Setting it between the two would have the estimate describe a different run
 * from the one that executes.
 */

/** Largest scale worth offering: past this the windows stop dividing the work. */
export const MAX_SCRATCH_BUDGET_SCALE = 64;

let scale = 1;

/** The scale in force, one unless a caller installed another. */
export function scratchBudgetScale(): number {
  return scale;
}

/**
 * Installs the scale.
 *
 * Rejects anything but a finite value of at least one, so a bad environment
 * variable cannot silently shrink the windows below what the browser budgets
 * assume and change the arithmetic of an operation that bounds itself.
 */
export function setScratchBudgetScale(next: number): void {
  if (!Number.isFinite(next) || next < 1 || next > MAX_SCRATCH_BUDGET_SCALE) {
    throw new RangeError(
      `scratch budget scale must be between 1 and ${MAX_SCRATCH_BUDGET_SCALE}, not ${next}`);
  }
  scale = next;
}

/**
 * One budget, scaled. Windows stay whole bytes and never shrink.
 *
 * maxScale caps how much of the process scale this budget takes, because not
 * every window wants the same one. A 3,300-residue tetramer measured on an
 * empty card: taking the scale from 16 to 64 moved the triangle contraction
 * from 41.5 to 93.2 TFLOP/s, and triangle attention the other way, from 74.6
 * to 59.3. The contraction is a GEMM whose rows are the block, so a bigger
 * block is more rows a tile and fewer passes over the operand it contracts
 * against; attention already had every query it needed and a wider window only
 * costs it. A window that measures worse as it grows says so here.
 */
export function scratchBudget(baseBytes: number, maxScale = MAX_SCRATCH_BUDGET_SCALE): number {
  if (!Number.isSafeInteger(baseBytes) || baseBytes <= 0) {
    throw new RangeError("a scratch budget must be a positive safe integer of bytes");
  }
  if (!Number.isFinite(maxScale) || maxScale < 1) {
    throw new RangeError("a scratch budget scale ceiling must be at least one");
  }
  return Math.floor(baseBytes * Math.min(scale, maxScale));
}
