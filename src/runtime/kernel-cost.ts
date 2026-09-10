/**
 * What each Evoformer kernel has to compute, so a millisecond becomes a rate.
 *
 * A profile says a kernel took 70 ms. That is not comparable to anything:
 * not to the same kernel at another length, not to the same kernel after the
 * block was resized, and not to ColabFold's Pallas kernels, which run a
 * different decomposition of the same mathematics. Arithmetic a second is
 * comparable, and it is what says whether a kernel is near the hardware or
 * near nothing.
 *
 * Every count here is the whole block's work, summed over dispatches, because
 * that is how a profile groups them: a blocked kernel with six dispatches
 * covers the same output as one with eleven, and only the total is stable.
 * The counts are exact multiply-adds times two, taken from each shader's own
 * rows, inner and columns, not estimated.
 *
 * Kernels that move data rather than contract it return undefined. A
 * normalization or a statistics pass is at the bandwidth roof by design and
 * quoting a FLOP rate for one says only that it has few flops.
 */

/** The dimensions one Evoformer stack runs at. */
export interface EvoformerCostShape {
  /** Residues. */
  readonly length: number;
  /** Alignment rows this stack carries: clustered for the main, extra for the other. */
  readonly sequences: number;
  /** MSA channels: 256 in the main stack, 64 in the extra one. */
  readonly cM: number;
  readonly cZ: number;
  readonly cOuter: number;
  readonly triangleHidden: number;
  /** MSA attention, row and column alike. */
  readonly msaHeads: number;
  readonly msaHeadDim: number;
  /** Triangle attention over the pair. */
  readonly pairHeads: number;
  readonly pairHeadDim: number;
  /** Global column attention, which only the extra stack runs. */
  readonly globalHeads: number;
  readonly globalHeadDim: number;
  /** Whether the stack runs global column attention rather than column attention. */
  readonly globalColumnAttention: boolean;
}

/**
 * The shapes of AlphaFold's released monomer model.
 *
 * The channel counts and head widths are fixed by the parameters, so a caller
 * that knows the length and the two alignment depths knows the rest.
 */
export function monomerCostShapes(
  length: number, msaSequences: number, extraSequences: number,
): { readonly mainEvoformer: EvoformerCostShape; readonly extraMsa: EvoformerCostShape } {
  const pair = { cZ: 128, cOuter: 32, triangleHidden: 128, pairHeads: 4, pairHeadDim: 32 } as const;
  return {
    mainEvoformer: {
      ...pair, length, sequences: msaSequences, cM: 256,
      msaHeads: 8, msaHeadDim: 32, globalHeads: 8, globalHeadDim: 8,
      globalColumnAttention: false,
    },
    extraMsa: {
      ...pair, length, sequences: extraSequences, cM: 64,
      msaHeads: 8, msaHeadDim: 8, globalHeads: 8, globalHeadDim: 8,
      globalColumnAttention: true,
    },
  };
}

/** Multiply-adds are two operations, and every count below is in those. */
const gemm = (rows: number, inner: number, columns: number): number => 2 * rows * inner * columns;

/**
 * Flash attention over a batch of independent query rows.
 *
 * Two products of the same shape, the scores and the weighted values, so the
 * softmax between them is not counted: it is a few operations an element
 * against the contraction's head width and ablating it measures free.
 */
const flash = (batch: number, heads: number, queries: number, keys: number, headDim: number): number =>
  2 * gemm(batch * heads * queries, headDim, keys);

/** The four triangle attention kernels, which the two directions share. */
function triangleAttention(
  direction: "starting" | "ending",
): Record<string, (shape: EvoformerCostShape) => number> {
  const at = `triangle-attention-${direction}`;
  return {
    [`${at}.project`]: (s) => gemm(s.length * s.length, s.cZ, 4 * s.pairHeads * s.pairHeadDim),
    [`${at}.flash`]: (s) => flash(s.length, s.pairHeads, s.length, s.length, s.pairHeadDim),
    [`${at}.output`]: (s) => gemm(s.length * s.length, s.pairHeads * s.pairHeadDim, s.cZ),
    [`${at}.pair-bias`]: (s) => gemm(s.length * s.length, s.cZ, s.pairHeads),
  };
}

/**
 * What one block of each kernel contracts, by the label a profile groups on.
 *
 * The key is the label with its stack prefix and dispatch index removed, so
 * one entry covers a kernel in either stack. Absent means data movement, and
 * the report says so rather than dividing by a count it invented.
 */
const KERNEL_FLOPS: Readonly<Record<string, (shape: EvoformerCostShape) => number>> = {
  // Outer product mean. The projections are per alignment row; the
  // contraction and the output projection are per residue pair.
  "opm.project": (s) => 2 * gemm(s.sequences * s.length, s.cM, s.cOuter),
  "opm.contract": (s) => gemm(s.length * s.cOuter, s.sequences, s.length * s.cOuter),
  "opm.project-output": (s) => gemm(s.length * s.length, s.cOuter * s.cOuter, s.cZ),

  // MSA row attention, biased by the pair. Its batch is the alignment.
  "msa-row-attention.project": (s) =>
    gemm(s.sequences * s.length, s.cM, 4 * s.msaHeads * s.msaHeadDim),
  "msa-row-attention.flash": (s) =>
    flash(s.sequences, s.msaHeads, s.length, s.length, s.msaHeadDim),
  "msa-row-attention.output": (s) =>
    gemm(s.sequences * s.length, s.msaHeads * s.msaHeadDim, s.cM),
  "msa-row-attention.pair-bias": (s) => gemm(s.length * s.length, s.cZ, s.msaHeads),

  // MSA column attention, which the main stack runs: the transposed view, so
  // its batch is the residues and it attends over the alignment.
  "msa-column-attention.project": (s) =>
    gemm(s.sequences * s.length, s.cM, 4 * s.msaHeads * s.msaHeadDim),
  "msa-column-attention.flash": (s) =>
    flash(s.length, s.msaHeads, s.sequences, s.sequences, s.msaHeadDim),
  "msa-column-attention.output": (s) =>
    gemm(s.sequences * s.length, s.msaHeads * s.msaHeadDim, s.cM),

  // Global column attention, which the extra stack runs instead. One key and
  // one value a residue serve every head, so the projection is head-free.
  "msa-column-global-attention.kv": (s) =>
    2 * gemm(s.sequences * s.length, s.cM, s.globalHeadDim),
  "msa-column-global-attention.query": (s) =>
    gemm(s.length, s.cM, s.globalHeads * s.globalHeadDim),
  "msa-column-global-attention.flash": (s) =>
    flash(s.length, s.globalHeads, 1, s.sequences, s.globalHeadDim),
  // The gate is a contraction of the same shape as the projection it gates,
  // fused into the projection's source. Both halves count.
  "msa-column-global-attention.output": (s) =>
    2 * gemm(s.sequences * s.length, s.globalHeads * s.globalHeadDim, s.cM),

  // Transitions, four times the channels wide in the middle.
  "msa-transition.first": (s) => gemm(s.sequences * s.length, s.cM, 4 * s.cM),
  "msa-transition.second": (s) => gemm(s.sequences * s.length, 4 * s.cM, s.cM),
  "pair-transition.first": (s) => gemm(s.length * s.length, s.cZ, 4 * s.cZ),
  "pair-transition.second": (s) => gemm(s.length * s.length, 4 * s.cZ, s.cZ),

  // Triangle attention. Both directions contract the same shapes, so the
  // entries below are the same four functions listed twice.
  ...triangleAttention("starting"),
  ...triangleAttention("ending"),

  // Triangle multiplication. The contraction is the only cubic kernel in the
  // model, and it runs once a hidden channel.
  "triangle.project-gate": (s) => gemm(s.length * s.length, s.cZ, s.cZ),
  "triangle.project-block": (s) => gemm(s.length * s.length, s.cZ, 2 * s.triangleHidden),
  "triangle.project-whole": (s) => gemm(s.length * s.length, s.cZ, 2 * s.triangleHidden),
  "triangle.contract": (s) => gemm(s.length * s.length, s.length, s.triangleHidden),
  "triangle.project-output": (s) => gemm(s.length * s.length, s.triangleHidden, s.cZ),
};

/**
 * The key a label groups under.
 *
 * A profile label carries the stack it ran in and the dispatch that wrote it,
 * and a triangle label carries its direction. None of the three changes what
 * the kernel contracts.
 */
export function kernelCostKey(label: string): string {
  return label
    .replace(/[.-]?\d+$/, "")
    .replace(/^extra\./, "")
    .replace(/^triangle\.(incoming|outgoing)\./, "triangle.");
}

/** The block's arithmetic for one label, or undefined where it moves data instead. */
export function kernelFlops(label: string, shape: EvoformerCostShape): number | undefined {
  const key = kernelCostKey(label);
  if (key === "msa-column-attention.flash" && shape.globalColumnAttention) return undefined;
  return KERNEL_FLOPS[key]?.(shape);
}

/**
 * The head width a kernel contracts over, or undefined where it has no heads.
 *
 * An attention kernel's rate is not one number: the same kernel reaches a
 * different fraction of the hardware at head 8 than at head 32, and here head
 * 8 is padded to the matrix unit's 16, so it does twice the counted work. A
 * reference rate measured at one width says nothing about the other.
 */
export function kernelHeadDim(label: string, shape: EvoformerCostShape): number | undefined {
  const key = kernelCostKey(label);
  if (key.startsWith("msa-column-global-attention.")) return shape.globalHeadDim;
  if (key.startsWith("msa-row-attention.") || key.startsWith("msa-column-attention.")) {
    return shape.msaHeadDim;
  }
  if (key.startsWith("triangle-attention-")) return shape.pairHeadDim;
  return undefined;
}

/** A Pallas figure to hold ours against, and the shape it was taken at. */
export interface KernelBaseline {
  /** The labels of ours that together do this kernel's work. */
  readonly kernels: readonly string[];
  readonly name: string;
  /** Arithmetic a second, where the reference was measured as a rate. */
  readonly tflops?: number;
  /** Milliseconds one Evoformer block spends here, where that is what was measured. */
  readonly millisecondsPerBlock?: number;
  readonly length: number;
  /**
   * The head width the reference ran at, where it has one. A kernel of ours at
   * another width matches the row by name and is reported, but is given no
   * ratio: the two rates are not of the same thing.
   *
   * This gates the rate only. A rate carries between the two stacks wherever
   * the width agrees, because it is a property of the kernel and the shape.
   */
  readonly headDim?: number;
  /**
   * The stack the per-block milliseconds were measured on. This gates that
   * figure only, and it must be set for the figure to be used at all: a main
   * Evoformer block and an extra-MSA block hold different channel counts and
   * different depths, so one block's milliseconds are not the other's.
   */
  readonly stack?: "mainEvoformer" | "extraMsa";
  readonly source: string;
}

export interface KernelCostRow {
  readonly label: string;
  readonly count: number;
  readonly milliseconds: number;
  /** The share of the block this kernel is. */
  readonly share: number;
  readonly flops: number | undefined;
  readonly tflops: number | undefined;
  /** The width this kernel's heads contract over, where it has heads. */
  readonly headDim: number | undefined;
  readonly baseline: KernelBaseline | undefined;
  /** Baseline rate over ours, so above one is a gap to close. */
  readonly ratio: number | undefined;
  /** Why a matched baseline gave no ratio. Empty where it did, or where none matched. */
  readonly mismatch: string;
}

export interface KernelCostEntry {
  readonly label: string;
  readonly nanoseconds: number;
}

/**
 * One row a kernel, sorted by what it costs.
 *
 * A baseline matches a row when it names that row's label. Where the baseline
 * covers several of our kernels it is attached to each of them and its rate is
 * compared against their total, because splitting one reference kernel's time
 * between three of ours would be an invention.
 */
export function kernelCostRows(
  entries: readonly KernelCostEntry[],
  shape: EvoformerCostShape,
  baselines: readonly KernelBaseline[] = [],
): readonly KernelCostRow[] {
  const grouped = new Map<string, { nanoseconds: number; count: number }>();
  for (const entry of entries) {
    const key = entry.label.replace(/[.-]?\d+$/, "");
    const held = grouped.get(key) ?? { nanoseconds: 0, count: 0 };
    grouped.set(key, { nanoseconds: held.nanoseconds + entry.nanoseconds, count: held.count + 1 });
  }
  const totalNanoseconds = entries.reduce((sum, entry) => sum + entry.nanoseconds, 0);
  const rows: KernelCostRow[] = [];
  for (const [label, value] of grouped) {
    const milliseconds = value.nanoseconds / 1e6;
    const flops = kernelFlops(label, shape);
    const tflops = flops === undefined || value.nanoseconds === 0
      ? undefined : flops / value.nanoseconds / 1e3;
    const baseline = baselines.find((entry) => entry.kernels.includes(kernelCostKey(label)));
    const headDim = kernelHeadDim(label, shape);
    const comparable = baseline?.headDim === undefined || baseline.headDim === headDim;
    rows.push({
      label, count: value.count, milliseconds,
      share: totalNanoseconds === 0 ? 0 : value.nanoseconds / totalNanoseconds,
      flops, tflops, headDim, baseline,
      ratio: !comparable || baseline?.tflops === undefined || tflops === undefined
        ? undefined : baseline.tflops / tflops,
      mismatch: comparable || baseline === undefined ? ""
        : `head ${headDim} against the baseline's ${baseline.headDim}`,
    });
  }
  return rows.sort((left, right) => right.milliseconds - left.milliseconds);
}

/**
 * What a baseline that covers several kernels is worth against their sum.
 *
 * The reference measured one kernel where we run three, so only the group
 * compares. Milliseconds are per block on both sides, which is why the shape
 * the baseline was taken at has to be reported beside it.
 */
export interface KernelGroupComparison {
  readonly baseline: KernelBaseline;
  readonly labels: readonly string[];
  readonly milliseconds: number;
  readonly tflops: number | undefined;
  /** Theirs over ours: above one where they are ahead. */
  readonly rateRatio: number | undefined;
  readonly millisecondRatio: number | undefined;
}

export function kernelGroupComparisons(
  rows: readonly KernelCostRow[], baselines: readonly KernelBaseline[],
  stack?: "mainEvoformer" | "extraMsa",
): readonly KernelGroupComparison[] {
  return baselines.map((baseline) => {
    const members = rows.filter((row) => baseline.kernels.includes(kernelCostKey(row.label))
      && (baseline.headDim === undefined || baseline.headDim === row.headDim));
    const milliseconds = members.reduce((sum, row) => sum + row.milliseconds, 0);
    const flops = members.reduce(
      (sum, row) => sum === undefined || row.flops === undefined ? undefined : sum + row.flops,
      0 as number | undefined);
    const tflops = flops === undefined || milliseconds === 0
      ? undefined : flops / milliseconds / 1e9;
    return {
      baseline, labels: members.map((row) => row.label), milliseconds, tflops,
      rateRatio: baseline.tflops === undefined || tflops === undefined
        ? undefined : baseline.tflops / tflops,
      millisecondRatio: baseline.millisecondsPerBlock === undefined || milliseconds === 0
        || baseline.stack === undefined || baseline.stack !== stack
        ? undefined : milliseconds / baseline.millisecondsPerBlock,
    };
  }).filter((comparison) => comparison.labels.length > 0);
}

const rate = (value: number | undefined): string =>
  value === undefined ? "     -" : value.toFixed(1).padStart(6);

export function formatKernelCostReport(
  rows: readonly KernelCostRow[], groups: readonly KernelGroupComparison[], shape: EvoformerCostShape,
): string {
  const lines = [
    `  ${"ms".padStart(8)} ${"share".padStart(6)} ${"n".padStart(3)}`
    + ` ${"GFLOP".padStart(9)} ${"TFLOP/s".padStart(7)} ${"vs Pallas".padStart(9)}  kernel`,
  ];
  for (const row of rows) {
    const flops = row.flops === undefined ? "        -" : (row.flops / 1e9).toFixed(1).padStart(9);
    const against = row.ratio !== undefined ? `${row.ratio.toFixed(2)}x`.padStart(9)
      : row.mismatch === "" ? "        -" : "  no match";
    lines.push(`  ${row.milliseconds.toFixed(3).padStart(8)} ${(row.share * 100).toFixed(1).padStart(5)}%`
      + ` ${String(row.count).padStart(3)} ${flops} ${rate(row.tflops)} ${against}  ${row.label}`
      + (row.mismatch === "" ? "" : `  (${row.mismatch})`));
  }
  if (groups.length > 0) {
    lines.push("", `  against ColabFold's Pallas kernels (this stack is ${shape.length} residues,`
      + ` ${shape.sequences} rows):`);
    for (const group of groups) {
      // Both sides in one unit, and only the unit the reference was measured
      // in: a rate against a rate, or milliseconds against milliseconds.
      const [ours, theirs, verdict] = group.rateRatio !== undefined
        ? [`${group.tflops!.toFixed(1)} TFLOP/s`, `${group.baseline.tflops!.toFixed(1)} TFLOP/s`,
          `${group.rateRatio.toFixed(2)}x`]
        : group.millisecondRatio !== undefined
          ? [`${group.milliseconds.toFixed(2)} ms`,
            `${group.baseline.millisecondsPerBlock!.toFixed(2)} ms`,
            `${group.millisecondRatio.toFixed(2)}x`]
          : [`${group.milliseconds.toFixed(2)} ms`, "-", "not comparable here"];
      lines.push(`    ${group.baseline.name.padEnd(28)} ours ${ours.padStart(14)}`
        + `  theirs ${theirs.padStart(14)}  ${verdict}`);
      lines.push(`      ${group.labels.join(", ")}`);
      lines.push(`      baseline at ${group.baseline.length} residues; ${group.baseline.source}`);
    }
  }
  return lines.join("\n");
}
