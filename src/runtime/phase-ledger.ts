/**
 * A closed ledger of where a run's wall clock went.
 *
 * Closed is the point. Every millisecond between the ledger's origin and its
 * report belongs to exactly one row, including the ones no phase claimed:
 * entering a phase switches which bucket time accrues to, and anything running
 * outside a phase accrues to `unaccounted`. So the rows always sum to the
 * total and work nobody thought to name shows up as a gap rather than
 * disappearing into a neighbour.
 *
 * That is not hypothetical. Featurisation built the next recycle's features
 * from inside the recycle loop, so its 3.3 s a recycle was billed to the model
 * and read as GPU time for a year.
 *
 * Time is exclusive: a phase entered inside another suspends the outer one, so
 * calibration does not also count as device setup. A row's first entry is kept
 * apart from the rest, which is what separates a cold start's compiles from
 * the steady state without a second mechanism.
 *
 * Off by default and free when off: `phase` calls its body directly when no
 * ledger is installed.
 */

const UNACCOUNTED = "unaccounted";

export interface PhaseRow {
  readonly name: string;
  readonly count: number;
  /** Exclusive wall time over every entry. */
  readonly milliseconds: number;
  /** The first entry alone, where a cold start's compiles land. */
  readonly firstMilliseconds: number;
}

export interface PhaseReport {
  readonly rows: readonly PhaseRow[];
  readonly totalMilliseconds: number;
}

interface Bucket { milliseconds: number; count: number; first: number | undefined }

export class PhaseLedger {
  readonly #buckets = new Map<string, Bucket>();
  readonly #stack: string[] = [];
  readonly #entered: number[] = [];
  /** Whether each open level was opened by `mark`, so `mark` can close it. */
  readonly #marked: boolean[] = [];
  readonly #origin: number;
  #switched: number;
  #before: string;

  /**
   * `origin` back-dates the ledger, and `before` names the time between it and
   * the first phase. On node `performance.now()` counts from process start, so
   * `new PhaseLedger(0, "module load")` bills the imports honestly. Once a
   * phase has opened, later gaps go back to `unaccounted`.
   */
  constructor(origin: number = performance.now(), before: string = UNACCOUNTED) {
    this.#origin = origin;
    this.#switched = origin;
    this.#before = before;
  }

  #bucket(name: string): Bucket {
    let bucket = this.#buckets.get(name);
    if (bucket === undefined) {
      bucket = { milliseconds: 0, count: 0, first: undefined };
      this.#buckets.set(name, bucket);
    }
    return bucket;
  }

  /** Charges everything since the last switch to whatever is running now. */
  #charge(): number {
    const now = performance.now();
    this.#bucket(this.#stack[this.#stack.length - 1] ?? this.#before)
      .milliseconds += now - this.#switched;
    this.#switched = now;
    return now;
  }

  enter(name: string): void {
    this.#charge();
    this.#before = UNACCOUNTED;
    const bucket = this.#bucket(name);
    bucket.count += 1;
    this.#stack.push(name);
    this.#marked.push(false);
    // The bucket's running exclusive total, so the first entry's own cost is
    // the delta at `leave` and stays comparable with the column beside it.
    this.#entered.push(bucket.milliseconds);
  }

  /**
   * Switches to `name` at the current level, ending whatever mark preceded it.
   *
   * A recycle is one long imperative block, and wrapping each of its stages in
   * a closure would mean restructuring the bindings that cross them. A
   * sequence of marks costs one line a boundary instead.
   */
  mark(name: string): void {
    this.endMark();
    this.enter(name);
    this.#marked[this.#marked.length - 1] = true;
  }

  /** Ends the current mark, leaving the enclosing phase running. */
  endMark(): void {
    if (this.#marked[this.#marked.length - 1] === true) this.leave();
  }

  leave(): void {
    this.#charge();
    const name = this.#stack.pop();
    const entered = this.#entered.pop();
    this.#marked.pop();
    if (name === undefined || entered === undefined) return;
    const bucket = this.#bucket(name);
    if (bucket.first === undefined) bucket.first = bucket.milliseconds - entered;
  }

  report(): PhaseReport {
    // One clock reading for both the buckets and the total, or the rows would
    // fall short of it by however long building the report takes.
    const now = this.#charge();
    const rows = [...this.#buckets].map(([name, bucket]) => ({
      name, count: bucket.count, milliseconds: bucket.milliseconds,
      firstMilliseconds: bucket.first ?? bucket.milliseconds,
    })).filter((row) => row.count > 0 || row.milliseconds > 0)
      .sort((left, right) => right.milliseconds - left.milliseconds);
    return { rows, totalMilliseconds: now - this.#origin };
  }
}

let installed: PhaseLedger | undefined;

/** Installs a ledger; `undefined` turns the instrumentation back off. */
export function setPhaseLedger(ledger: PhaseLedger | undefined): void {
  installed = ledger;
}

export function phaseLedger(): PhaseLedger | undefined { return installed; }

/** Starts a named stage in a sequence; see `PhaseLedger.mark`. */
export function markPhase(name: string): void { installed?.mark(name); }

/** Ends the current stage started by `markPhase`. */
export function endPhase(): void { installed?.endMark(); }

/** Runs `body` as a named phase, or just runs it when no ledger is installed. */
export async function timed<T>(name: string, body: () => Promise<T>): Promise<T> {
  const ledger = installed;
  if (ledger === undefined) return body();
  ledger.enter(name);
  try { return await body(); } finally { ledger.leave(); }
}

/** The synchronous form, for a section that does not await. */
export function timedSync<T>(name: string, body: () => T): T {
  const ledger = installed;
  if (ledger === undefined) return body();
  ledger.enter(name);
  try { return body(); } finally { ledger.leave(); }
}

/** A report as aligned text, with the rows summing to the total by construction. */
export function formatPhaseReport(report: PhaseReport): string {
  const width = Math.max(12, ...report.rows.map((row) => row.name.length));
  const seconds = (ms: number): string => `${(ms / 1000).toFixed(2)} s`;
  const lines = [
    `${"phase".padEnd(width)}  ${"total".padStart(9)}  ${"share".padStart(6)}`
    + `  ${"n".padStart(4)}  ${"first".padStart(9)}  ${"rest/n".padStart(9)}`,
    "-".repeat(width + 46),
  ];
  for (const row of report.rows) {
    const rest = row.count > 1
      ? seconds((row.milliseconds - row.firstMilliseconds) / (row.count - 1)) : "-";
    lines.push(`${row.name.padEnd(width)}  ${seconds(row.milliseconds).padStart(9)}`
      + `  ${`${(100 * row.milliseconds / report.totalMilliseconds).toFixed(1)}%`.padStart(6)}`
      + `  ${String(row.count).padStart(4)}  ${seconds(row.firstMilliseconds).padStart(9)}`
      + `  ${rest.padStart(9)}`);
  }
  lines.push("-".repeat(width + 46));
  lines.push(`${"total".padEnd(width)}  ${seconds(report.totalMilliseconds).padStart(9)}`);
  return lines.join("\n");
}
