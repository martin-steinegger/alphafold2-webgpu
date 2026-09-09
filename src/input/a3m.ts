export interface A3mAlignment {
  readonly query: string;
  readonly descriptions: readonly string[];
  readonly sequences: readonly string[];
  readonly deletionMatrix: readonly (readonly number[])[];
  readonly depth: number;
  readonly length: number;
}

const ALIGNED_RESIDUES = /^[ACDEFGHIKLMNPQRSTVWYX-]+$/;
/**
 * The same alphabet as a lookup by character code.
 *
 * The residue loop runs once per character of the whole alignment, 9.1 million
 * times on an 8.77 MB file. Doing it with strings meant a one-character string
 * from the iterator, another from `toUpperCase`, and a regex match, for every
 * one of them.
 */
const ALIGNED_CODES = new Uint8Array(128);
for (const residue of "ACDEFGHIKLMNPQRSTVWYX-") ALIGNED_CODES[residue.charCodeAt(0)] = 1;

/** Builds a string from character codes without exceeding the argument limit. */
function stringFromCodes(codes: Uint16Array, count: number): string {
  const CHUNK = 4096;
  if (count <= CHUNK) return String.fromCharCode(...codes.subarray(0, count));
  let text = "";
  for (let start = 0; start < count; start += CHUNK) {
    text += String.fromCharCode(...codes.subarray(start, Math.min(start + CHUNK, count)));
  }
  return text;
}

export function parseA3m(text: string): A3mAlignment {
  const descriptions: string[] = [];
  const rawSequences: string[] = [];
  let current = -1;
  for (const sourceLine of text.split(/\r?\n/)) {
    const line = sourceLine.trim();
    if (line === "" || line.startsWith("#")) continue;
    if (line.startsWith(">")) {
      const description = line.slice(1).trim();
      if (description === "") throw new Error("A3M contains an empty FASTA header");
      descriptions.push(description);
      rawSequences.push("");
      current += 1;
      continue;
    }
    if (current < 0) throw new Error("A3M sequence data appears before the first FASTA header");
    if (/\s/.test(line)) throw new Error(`A3M sequence ${descriptions[current]} contains whitespace`);
    rawSequences[current] += line;
  }
  if (rawSequences.length === 0) throw new Error("A3M contains no sequences");

  const sequences: string[] = [];
  const deletionMatrix: number[][] = [];
  for (let row = 0; row < rawSequences.length; row += 1) {
    const raw = rawSequences[row]!;
    if (raw === "") throw new Error(`A3M sequence ${descriptions[row]} is empty`);
    let insertionCount = 0;
    const kept = new Uint16Array(raw.length);
    const deletions: number[] = new Array(raw.length);
    let count = 0;
    for (let index = 0; index < raw.length; index += 1) {
      const code = raw.charCodeAt(index);
      // Lower case is an insertion relative to the query and is not aligned.
      if (code >= 97 && code <= 122) { insertionCount += 1; continue; }
      if (code >= 128 || ALIGNED_CODES[code] === 0) {
        throw new Error(`A3M sequence ${descriptions[row]} contains invalid residue ${JSON.stringify(raw[index])}`);
      }
      kept[count] = code;
      deletions[count] = insertionCount;
      count += 1;
      insertionCount = 0;
    }
    deletions.length = count;
    sequences.push(stringFromCodes(kept, count));
    deletionMatrix.push(deletions);
  }

  const length = sequences[0]!.length;
  if (length === 0 || sequences[0]!.includes("-")) {
    throw new Error("the first A3M sequence must be a non-empty, ungapped query");
  }
  for (let row = 0; row < sequences.length; row += 1) {
    if (sequences[row]!.length !== length) {
      throw new Error(
        `A3M row ${descriptions[row]} has aligned length ${sequences[row]!.length}; expected ${length}`,
      );
    }
  }
  return {
    query: sequences[0]!,
    descriptions,
    sequences,
    deletionMatrix,
    depth: sequences.length,
    length,
  };
}

