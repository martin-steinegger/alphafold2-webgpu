import { parseA3m } from "../src/input/a3m.js";
import { chainBoundaries, chainSpans, type ChainSpan } from "./chains.js";

export interface MsaCoverageData {
  readonly sequences: readonly string[];
  readonly identities: Float32Array;
  /** Rows by ascending identity to the query. */
  readonly order: Uint32Array;
  /** Rows in the order the plot draws them, top first. */
  readonly rows: Uint32Array;
  /**
   * Indices into `rows` where a block of alignments covering a different set
   * of chains begins. For a complex this separates the paired rows from each
   * chain's own hits, which is the structure ColabFold's coverage plot shows.
   */
  readonly blockStarts: readonly number[];
  readonly coverage: Uint32Array;
  readonly depth: number;
  readonly length: number;
}

/**
 * Row scores and grouping, following ColabFold's `plot_msa_v2`.
 *
 * A row is scored against the chains it actually covers: the mean identity to
 * the query over each covered chain's own columns, averaged over those chains.
 * A hit that matches one chain of a complex closely is therefore drawn as a
 * close hit, not as a distant one diluted by the chains it is gapped in, and a
 * monomer reduces to plain identity over the whole query.
 */
interface RowScore {
  readonly identity: number;
  /** Which chains the row covers, chain A in the highest bit. */
  readonly key: number;
}

function scoreRow(sequence: string, query: string, spans: readonly ChainSpan[]): RowScore {
  let covered = 0; let total = 0; let key = 0;
  for (const span of spans) {
    let matches = 0; let residues = 0;
    for (let position = span.start; position < span.end; position += 1) {
      if (sequence[position] === query[position]) matches += 1;
      if (sequence[position] !== "-") residues += 1;
    }
    total += matches / span.length;
    if (residues > 0) { covered += 1; key |= 1 << (spans.length - 1 - span.index); }
  }
  return { identity: covered === 0 ? 0 : total / covered, key };
}

export function analyzeMsa(a3m: string, chainLengths?: readonly number[]): MsaCoverageData {
  const alignment = parseA3m(a3m); const { sequences, depth, length } = alignment;
  const chains = chainLengths !== undefined && chainLengths.length > 1 ? chainLengths : undefined;
  const spans = chainSpans(chains ?? [length]);
  const identities = new Float32Array(depth); const coverage = new Uint32Array(length);
  const keys = new Uint32Array(depth);
  for (let row = 0; row < depth; row += 1) {
    const sequence = sequences[row]!;
    for (let position = 0; position < length; position += 1) {
      if (sequence[position] !== "-") coverage[position] = coverage[position]! + 1;
    }
    const score = scoreRow(sequence, sequences[0]!, spans);
    identities[row] = score.identity; keys[row] = score.key;
  }
  const sorted = Array.from({ length: depth }, (_, index) => index)
    .sort((left, right) => identities[left]! - identities[right]! || left - right);

  // A complex draws its rows in blocks, one per set of covered chains, each
  // block sorted by identity. ColabFold orders the blocks by that set read as a
  // binary number, largest first, which puts the rows covering every chain at
  // the top and then each chain's own hits in chain order. A monomer has one
  // block, so this is just the sorted rows.
  const blocks = new Map<number, number[]>();
  // Most similar first, and the query ahead of anything that ties with it.
  const byIdentity = Array.from({ length: depth }, (_, index) => index)
    .sort((left, right) => identities[right]! - identities[left]! || left - right);
  for (const row of byIdentity) {
    const key = chains === undefined ? 0 : keys[row]!;
    const block = blocks.get(key) ?? [];
    block.push(row);
    blocks.set(key, block);
  }
  const ordered = [...blocks.entries()].sort(([left], [right]) => right - left);
  const rows: number[] = []; const blockStarts: number[] = [];
  for (const [, block] of ordered) {
    if (rows.length > 0) blockStarts.push(rows.length);
    rows.push(...block);
  }
  return {
    sequences, identities, order: Uint32Array.from(sorted), rows: Uint32Array.from(rows), blockStarts,
    coverage, depth, length,
  };
}

/**
 * Matplotlib's `rainbow_r`, which is the colormap ColabFold's coverage plot
 * uses: red at no identity, through green and blue, to violet at the query
 * itself. Matplotlib defines `rainbow` by these three functions and samples
 * them into a 256-entry table, so the quantization is reproduced too and the
 * colours match to a level.
 */
export function identityColor(value: number): readonly [number, number, number] {
  const entry = Math.min(255, Math.floor(Math.max(0, Math.min(1, value)) * 256));
  const x = 1 - entry / 255;
  const byte = (channel: number): number => Math.round(Math.max(0, Math.min(1, channel)) * 255);
  return [byte(Math.abs(2 * x - .5)), byte(Math.sin(x * Math.PI)), byte(Math.cos(x * Math.PI / 2))];
}

export function drawMsaCoverage(
  canvas: HTMLCanvasElement, a3m: string, chainLengths?: readonly number[],
): MsaCoverageData {
  const data = analyzeMsa(a3m, chainLengths); const context = canvas.getContext("2d");
  if (context === null) return data;
  const boundaries = chainLengths === undefined || chainLengths.length < 2 ? [] : chainBoundaries(chainLengths);
  const { width, height } = canvas; const left = 56; const right = 82; const bottom = 38;
  // Complexes get a strip above the plot for the chain letters.
  const top = boundaries.length === 0 ? 18 : 32;
  const plotWidth = width - left - right; const plotHeight = height - top - bottom;
  context.clearRect(0, 0, width, height); context.fillStyle = "#fff"; context.fillRect(0, 0, width, height);

  const sampledRows = Math.min(data.depth, 2_048);
  const image = new ImageData(data.length, sampledRows);
  image.data.fill(255);
  for (let outputRow = 0; outputRow < sampledRows; outputRow += 1) {
    const rank = sampledRows === 1 ? 0
      : Math.round(outputRow / (sampledRows - 1) * (data.depth - 1));
    const source = data.rows[rank]!; const identity = data.identities[source]!; const color = identityColor(identity);
    for (let position = 0; position < data.length; position += 1) {
      if (data.sequences[source]![position] === "-") continue;
      const pixel = (outputRow * data.length + position) * 4;
      image.data[pixel] = color[0]; image.data[pixel + 1] = color[1]; image.data[pixel + 2] = color[2];
    }
  }
  const temporary = document.createElement("canvas"); temporary.width = data.length; temporary.height = sampledRows;
  temporary.getContext("2d")?.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false; context.drawImage(temporary, left, top, plotWidth, plotHeight);

  context.strokeStyle = "#111"; context.lineWidth = 1.5; context.beginPath();
  for (let position = 0; position < data.length; position += 1) {
    const x = left + (position + .5) / data.length * plotWidth;
    const y = top + (1 - data.coverage[position]! / data.depth) * plotHeight;
    if (position === 0) context.moveTo(x, y); else context.lineTo(x, y);
  }
  context.stroke();

  // Chain boundaries down the plot and block boundaries across it, both in
  // black, the way ColabFold marks a complex's coverage.
  if (boundaries.length > 0) {
    context.strokeStyle = "#111"; context.lineWidth = 1.5; context.beginPath();
    for (const position of boundaries) {
      const x = left + position / data.length * plotWidth;
      context.moveTo(x, top); context.lineTo(x, top + plotHeight);
    }
    for (const start of data.blockStarts) {
      const y = top + start / data.depth * plotHeight;
      context.moveTo(left, y); context.lineTo(left + plotWidth, y);
    }
    context.stroke();
    context.font = "600 12px Roboto Mono"; context.textAlign = "center";
    for (const span of chainSpans(chainLengths!)) {
      context.fillStyle = span.color;
      context.fillText(span.letter, left + (span.start + span.end) / 2 / data.length * plotWidth, top - 8);
    }
  }

  context.strokeStyle = "#777"; context.lineWidth = 1; context.strokeRect(left, top, plotWidth, plotHeight);
  context.fillStyle = "#666"; context.font = "11px Roboto Mono"; context.textAlign = "center";
  context.fillText("Positions", left + plotWidth / 2, height - 8);
  context.save(); context.translate(13, top + plotHeight / 2); context.rotate(-Math.PI / 2);
  context.fillText("Sequences", 0, 0); context.restore();
  context.textAlign = "right"; context.fillText(String(data.depth), left - 7, top + 5); context.fillText("0", left - 7, top + plotHeight);
  context.textAlign = "center"; context.fillText("1", left, height - 22); context.fillText(String(data.length), left + plotWidth, height - 22);

  const colorX = width - 52; const colorWidth = 13;
  for (let pixel = 0; pixel < plotHeight; pixel += 1) {
    const identity = 1 - pixel / Math.max(1, plotHeight - 1); const color = identityColor(identity);
    context.fillStyle = `rgb(${color.join(",")})`; context.fillRect(colorX, top + pixel, colorWidth, 1);
  }
  context.strokeStyle = "#777"; context.strokeRect(colorX, top, colorWidth, plotHeight);
  context.fillStyle = "#666"; context.textAlign = "left";
  context.fillText("100%", colorX + 18, top + 5); context.fillText("50%", colorX + 18, top + plotHeight / 2 + 4);
  context.fillText("0%", colorX + 18, top + plotHeight);
  return data;
}
