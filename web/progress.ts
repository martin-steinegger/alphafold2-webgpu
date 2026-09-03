/**
 * Turning blocks completed into time remaining.
 *
 * The trunk is the whole cost of a long prediction, and it is two stacks of
 * blocks whose costs differ by an order of magnitude, so the estimate keeps
 * them apart and only answers once it has timed a block of each.
 */
export interface TrunkPosition {
  /** Which stack is running: the extra-MSA stack or the main Evoformer. */
  readonly phase: "extra-msa" | "evoformer" | "structure";
  /** Blocks of that stack already finished. */
  readonly completed: number;
  /** Recycles already finished, which is the index of the one running. */
  readonly recycle: number;
}

export interface TrunkShape {
  readonly extraBlocks: number;
  readonly mainBlocks: number;
  /** Recycles after the one running. */
  readonly recycles: number;
}

export interface TrunkRates {
  readonly extraSeconds?: number | undefined;
  readonly mainSeconds?: number | undefined;
}

/**
 * Seconds of trunk left, or undefined until both stacks have been timed.
 *
 * The structure module and the confidence heads are not counted: at the
 * lengths where this matters they are a few percent of a recycle.
 */
export function remainingTrunkSeconds(
  position: TrunkPosition, shape: TrunkShape, rates: TrunkRates,
): number | undefined {
  const { extraSeconds, mainSeconds } = rates;
  if (extraSeconds === undefined || mainSeconds === undefined) return undefined;
  const recycle = shape.extraBlocks * extraSeconds + shape.mainBlocks * mainSeconds;
  const thisRecycle = position.phase === "extra-msa"
    ? (shape.extraBlocks - position.completed) * extraSeconds + shape.mainBlocks * mainSeconds
    : Math.max(0, shape.mainBlocks - position.completed) * mainSeconds;
  const later = Math.max(0, shape.recycles - position.recycle) * recycle;
  return thisRecycle + later;
}

/** The same, as the phrase the stage line carries. */
export function remainingPhrase(seconds: number | undefined): string {
  if (seconds === undefined || seconds < 30) return "";
  const minutes = Math.round(seconds / 60);
  return minutes < 1 ? " · under a minute left" : ` · about ${minutes} min left`;
}
