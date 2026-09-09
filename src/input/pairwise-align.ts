import { RESTYPES, RESTYPE_INDEX } from "./residue-atoms.js";

/**
 * A query aligned to one template sequence.
 *
 * AlphaFold reaches this mapping the long way round: it searches a database for
 * the template, reads the mapping out of the hit, and calls kalign when the hit
 * disagrees with the structure's own sequence. A template someone hands us by
 * name needs none of that. What is left is one pairwise alignment, which is
 * what this is: Gotoh's affine-gap algorithm over BLOSUM62, with both pairs of
 * terminal gaps free so a domain aligns inside a longer query, and a construct
 * with tags on either end aligns to the query it covers, without either paying
 * for the overhang.
 */
export interface PairwiseAlignment {
  /**
   * Template residue index for each query position, or -1 where the query has
   * no aligned template residue. This is AlphaFold's mapping, densified.
   */
  readonly queryToTemplate: Int32Array;
  /** Query positions with a template residue behind them. */
  readonly alignedResidues: number;
  /** Aligned pairs whose residues are the same letter. */
  readonly identicalResidues: number;
  /** Aligned positions as a fraction of the query's length. */
  readonly coverage: number;
  /** Identical positions as a fraction of the aligned ones; 0 when none align. */
  readonly identity: number;
  readonly score: number;
}

// BLOSUM62 in AlphaFold's residue order with an unknown-residue row, which is
// the substitution matrix every pairwise protein aligner reaches for first.
const SUBSTITUTION_ORDER = `${RESTYPES}X`;
const SUBSTITUTION_SCORES = Int8Array.from([
   4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0,  0,
  -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,
  -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, -1,
  -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, -1,
   0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -2,
  -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, -1,
  -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, -1,
   0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1,
  -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, -1,
  -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -1,
  -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -1,
  -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, -1,
  -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -1,
  -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -1,
  -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2,
   1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,
   0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0,  0,
  -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -2,
  -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -1,
   0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -1,
   0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1,
]);

const SUBSTITUTION_SIZE = SUBSTITUTION_ORDER.length;
const UNKNOWN = SUBSTITUTION_SIZE - 1;

/** BLAST's own affine penalties for this matrix. */
const GAP_OPEN = 11;
const GAP_EXTEND = 1;

const NEGATIVE = -1e9;

function residueCodes(sequence: string): Uint8Array {
  return Uint8Array.from(sequence, (residue) => RESTYPE_INDEX.get(residue) ?? UNKNOWN);
}

/** Traceback states, one per matrix Gotoh keeps. */
const FROM_DIAGONAL = 0;
const FROM_QUERY_GAP = 1;
const FROM_TEMPLATE_GAP = 2;

/**
 * Aligns a query to a template sequence.
 *
 * Both sequences are one-letter codes. Terminal gaps are free at both ends of
 * both sequences, so the alignment finds where the two overlap rather than
 * forcing the shorter through the longer.
 */
export function alignSequences(query: string, template: string): PairwiseAlignment {
  const queryCodes = residueCodes(query.toUpperCase());
  const templateCodes = residueCodes(template.toUpperCase());
  const rows = queryCodes.length;
  const columns = templateCodes.length;
  if (rows === 0 || columns === 0) {
    return {
      queryToTemplate: new Int32Array(rows).fill(-1), alignedResidues: 0,
      identicalResidues: 0, coverage: 0, identity: 0, score: 0,
    };
  }

  // Gotoh keeps three running scores per cell. Only the previous row of each is
  // needed to fill the next, but the traceback needs every cell's decision, so
  // the scores stay on two rows and the decisions fill a rows-by-columns byte
  // array: 2.25 MB for a pair of 1500-residue sequences.
  const width = columns + 1;
  let bestPrevious = new Float64Array(width);
  let best = new Float64Array(width);
  let queryGapPrevious = new Float64Array(width);
  let queryGap = new Float64Array(width);
  let templateGapPrevious = new Float64Array(width);
  let templateGap = new Float64Array(width);
  const bestFrom = new Uint8Array(rows * columns);
  // The alignment may end anywhere along the last row or the last column, so
  // the last column's score is kept as it is computed.
  const lastColumnScores = new Float64Array(rows + 1);
  const queryGapExtends = new Uint8Array(rows * columns);
  const templateGapExtends = new Uint8Array(rows * columns);

  // Free terminal gaps: opening the alignment anywhere along either sequence
  // costs nothing, so the first row and column start at zero.
  for (let column = 0; column <= columns; column += 1) {
    bestPrevious[column] = 0;
    queryGapPrevious[column] = NEGATIVE;
    templateGapPrevious[column] = NEGATIVE;
  }

  for (let row = 1; row <= rows; row += 1) {
    best[0] = 0;
    queryGap[0] = NEGATIVE;
    templateGap[0] = NEGATIVE;
    const queryCode = queryCodes[row - 1]!;
    const scoreRow = queryCode * SUBSTITUTION_SIZE;
    const decisions = (row - 1) * columns;
    for (let column = 1; column <= columns; column += 1) {
      // A gap in the template: the query consumes a residue, the template does
      // not, so the score comes down the column.
      const openTemplateGap = bestPrevious[column]! - GAP_OPEN - GAP_EXTEND;
      const extendTemplateGap = templateGapPrevious[column]! - GAP_EXTEND;
      const templateGapScore = Math.max(openTemplateGap, extendTemplateGap);
      templateGap[column] = templateGapScore;
      templateGapExtends[decisions + column - 1] = extendTemplateGap > openTemplateGap ? 1 : 0;

      const openQueryGap = best[column - 1]! - GAP_OPEN - GAP_EXTEND;
      const extendQueryGap = queryGap[column - 1]! - GAP_EXTEND;
      const queryGapScore = Math.max(openQueryGap, extendQueryGap);
      queryGap[column] = queryGapScore;
      queryGapExtends[decisions + column - 1] = extendQueryGap > openQueryGap ? 1 : 0;

      const substitution = SUBSTITUTION_SCORES[scoreRow + templateCodes[column - 1]!]!;
      const diagonal = bestPrevious[column - 1]! + substitution;
      let bestScore = diagonal;
      let from = FROM_DIAGONAL;
      if (queryGapScore > bestScore) { bestScore = queryGapScore; from = FROM_QUERY_GAP; }
      if (templateGapScore > bestScore) { bestScore = templateGapScore; from = FROM_TEMPLATE_GAP; }
      best[column] = bestScore;
      bestFrom[decisions + column - 1] = from;
    }
    lastColumnScores[row] = best[columns]!;
    [bestPrevious, best] = [best, bestPrevious];
    [queryGapPrevious, queryGap] = [queryGap, queryGapPrevious];
    [templateGapPrevious, templateGap] = [templateGap, templateGapPrevious];
  }

  // Free terminal gaps again: the alignment may end anywhere along the last row
  // or the last column, whichever scored best.
  let endRow = rows;
  let endColumn = columns;
  let score = bestPrevious[columns]!;
  for (let column = 0; column <= columns; column += 1) {
    if (bestPrevious[column]! > score) { score = bestPrevious[column]!; endRow = rows; endColumn = column; }
  }
  for (let row = 0; row <= rows; row += 1) {
    if (lastColumnScores[row]! > score) { score = lastColumnScores[row]!; endRow = row; endColumn = columns; }
  }

  const queryToTemplate = new Int32Array(rows).fill(-1);
  let aligned = 0;
  let identical = 0;
  let row = endRow;
  let column = endColumn;
  let state = FROM_DIAGONAL;
  while (row > 0 && column > 0) {
    const cell = (row - 1) * columns + column - 1;
    if (state === FROM_DIAGONAL) state = bestFrom[cell]!;
    if (state === FROM_DIAGONAL) {
      queryToTemplate[row - 1] = column - 1;
      aligned += 1;
      if (queryCodes[row - 1] === templateCodes[column - 1] && queryCodes[row - 1] !== UNKNOWN) identical += 1;
      row -= 1; column -= 1;
    } else if (state === FROM_QUERY_GAP) {
      if (queryGapExtends[cell] === 0) state = FROM_DIAGONAL;
      column -= 1;
    } else {
      if (templateGapExtends[cell] === 0) state = FROM_DIAGONAL;
      row -= 1;
    }
  }

  return {
    queryToTemplate, alignedResidues: aligned, identicalResidues: identical,
    coverage: aligned / rows, identity: aligned === 0 ? 0 : identical / aligned, score,
  };
}
