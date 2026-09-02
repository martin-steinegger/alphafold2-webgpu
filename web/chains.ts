/**
 * Chain identity across the result views.
 *
 * A complex is only readable when the same chain carries the same letter and
 * the same colour in the structure, in every plot and in the PDB it downloads,
 * so the letters here are the ones `predictionToPdb` writes and the colours are
 * the ones the viewer, the legend and the plot labels all draw from.
 *
 * ColabFold marks chains with black boundary lines and letters from A onward;
 * this keeps that convention and swaps its neon PyMOL palette for one that
 * stays legible on a white plot and colour-blind safe up to eight chains.
 */

export const CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

const CHAIN_COLORS = [
  "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#e69f00", "#56b4e9", "#8c564b", "#7570b3",
  "#1b7837", "#c2185b", "#00838f", "#5d4037",
] as const;

export const chainLetter = (index: number): string => CHAIN_LETTERS[index] ?? `#${index + 1}`;

export const chainColor = (index: number): string => CHAIN_COLORS[index % CHAIN_COLORS.length]!;

/** One chain's place in the concatenated sequence, with the marks every view uses for it. */
export interface ChainSpan {
  readonly index: number;
  readonly letter: string;
  readonly color: string;
  /** First residue, counting from zero. */
  readonly start: number;
  /** One past the last residue. */
  readonly end: number;
  readonly length: number;
}

export function chainSpans(chainLengths: readonly number[]): ChainSpan[] {
  const spans: ChainSpan[] = [];
  let start = 0;
  chainLengths.forEach((length, index) => {
    spans.push({ index, letter: chainLetter(index), color: chainColor(index), start, end: start + length, length });
    start += length;
  });
  return spans;
}

/** Residue positions where one chain ends and the next begins; empty for a monomer. */
export function chainBoundaries(chainLengths: readonly number[]): number[] {
  const boundaries: number[] = [];
  let position = 0;
  for (const length of chainLengths.slice(0, -1)) { position += length; boundaries.push(position); }
  return boundaries;
}

/** Mean pLDDT within each chain, which is what tells a weak chain from a weak interface. */
export function chainMeanPlddt(plddt: Float32Array, chainLengths: readonly number[]): number[] {
  return chainSpans(chainLengths).map((span) => {
    let sum = 0;
    for (let residue = span.start; residue < span.end; residue += 1) sum += plddt[residue] ?? 0;
    return span.length === 0 ? 0 : sum / span.length;
  });
}

/**
 * Mean predicted aligned error for every ordered pair of chains.
 *
 * The diagonal says how well each chain is folded on its own and the
 * off-diagonal cells say how confidently the model placed one chain relative
 * to another, which is the number to read before trusting an interface.
 */
export function chainPairError(
  predictedAlignedError: Float32Array, chainLengths: readonly number[],
): number[][] {
  const spans = chainSpans(chainLengths);
  const length = spans.length === 0 ? 0 : spans[spans.length - 1]!.end;
  return spans.map((row) => spans.map((column) => {
    let sum = 0; let count = 0;
    for (let residue = row.start; residue < row.end; residue += 1) {
      for (let partner = column.start; partner < column.end; partner += 1) {
        sum += predictedAlignedError[residue * length + partner] ?? 0; count += 1;
      }
    }
    return count === 0 ? 0 : sum / count;
  }));
}
