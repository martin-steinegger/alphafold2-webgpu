import { assembleComplexA3m } from "./mmseqs2-api.js";

export interface ColabFoldComplexA3m {
  readonly chains: readonly string[];
  readonly uniqueSequences: readonly string[];
  readonly cardinalities: readonly number[];
  readonly a3m: string;
  readonly mask: Float32Array;
  readonly depth: number;
}

interface A3mRecord { readonly header: string; readonly sequence: string; }

function records(lines: readonly string[]): readonly A3mRecord[] {
  const output: { header: string; sequence: string }[] = [];
  for (const source of lines) {
    const line = source.trim();
    if (line === "") continue;
    if (line.startsWith(">")) {
      if (line.length === 1) throw new Error("ColabFold A3M contains an empty header");
      output.push({ header: line, sequence: "" });
    } else {
      const current = output.at(-1);
      if (current === undefined) throw new Error("ColabFold A3M sequence data precedes its first header");
      if (/\s/.test(line)) throw new Error("ColabFold A3M sequence rows cannot contain whitespace");
      current.sequence += line;
    }
  }
  if (output.length === 0 || output.some((record) => record.sequence === "")) {
    throw new Error("ColabFold A3M contains no complete sequence records");
  }
  return output;
}

function splitAligned(sequence: string, lengths: readonly number[]): {
  readonly segments: readonly string[]; readonly hasResidues: readonly boolean[];
} {
  const segments = new Array<string>(lengths.length).fill("");
  const hasResidues = new Array<boolean>(lengths.length).fill(false);
  let segment = 0; let aligned = 0; let start = 0;
  for (let index = 0; index < sequence.length && segment < lengths.length; index += 1) {
    const residue = sequence[index]!;
    const insertion = residue >= "a" && residue <= "z";
    if (insertion) continue;
    if (residue !== "-" && (residue < "A" || residue > "Z")) {
      throw new Error(`ColabFold A3M contains invalid residue ${JSON.stringify(residue)}`);
    }
    if (residue !== "-") hasResidues[segment] = true;
    aligned += 1;
    if (aligned === lengths[segment]) {
      segments[segment] = sequence.slice(start, index + 1);
      segment += 1; aligned = 0; start = index + 1;
    }
  }
  if (segment !== lengths.length || aligned !== 0 || !/^[a-z]*$/.test(sequence.slice(start))) {
    throw new Error("ColabFold A3M row does not match the declared chain lengths");
  }
  return { segments, hasResidues };
}

/** Parse ColabFold's serialized `#lengths<TAB>cardinalities` complex-A3M format. */
export function parseColabFoldComplexA3m(text: string): ColabFoldComplexA3m | undefined {
  const lines = text.replace(/\0/g, "").replace(/\r/g, "").split("\n");
  const first = lines.findIndex((line) => line.trim() !== "");
  if (first < 0 || !lines[first]!.trim().startsWith("#")) return undefined;
  const match = /^#([1-9][0-9]*(?:,[1-9][0-9]*)*)\t([1-9][0-9]*(?:,[1-9][0-9]*)*)$/.exec(lines[first]!.trim());
  if (match === null) throw new Error("invalid ColabFold complex-A3M header");
  const lengths = match[1]!.split(",").map(Number);
  const cardinalities = match[2]!.split(",").map(Number);
  if (lengths.length !== cardinalities.length || lengths.length === 0) {
    throw new Error("ColabFold complex-A3M lengths and cardinalities disagree");
  }
  const inputRecords = records(lines.slice(first + 1));
  const query = splitAligned(inputRecords[0]!.sequence, lengths).segments;
  const uniqueSequences = query.map((segment) => segment.toUpperCase());
  if (uniqueSequences.some((sequence, index) => sequence.length !== lengths[index]
    || sequence.includes("-") || !/^[A-Z]+$/.test(sequence))) {
    throw new Error("ColabFold complex-A3M query must contain one ungapped sequence per declared chain");
  }
  const paired = Array.from({ length: lengths.length }, () => [] as string[]);
  const unpaired = Array.from({ length: lengths.length }, () => [] as string[]);
  const seen = new Set<string>();
  const homomer = lengths.length === 1 && cardinalities[0]! > 1;
  for (const record of inputRecords) {
    const key = `${record.header}\n${record.sequence}`;
    if (seen.has(key)) continue;
    seen.add(key);
    const split = splitAligned(record.sequence, lengths);
    const isPaired = lengths.length > 1 && !homomer
      && split.hasResidues.filter(Boolean).length > 1;
    if (isPaired) {
      const labels = record.header.slice(1).split("\t");
      for (let chain = 0; chain < lengths.length; chain += 1) {
        paired[chain]!.push(`>${labels[chain] ?? labels.at(-1) ?? "paired"}\n${split.segments[chain]}\n`);
      }
    } else {
      for (let chain = 0; chain < lengths.length; chain += 1) if (split.hasResidues[chain]) {
        unpaired[chain]!.push(`${record.header}\n${split.segments[chain]}\n`);
      }
    }
  }
  if (unpaired.some((chunks) => chunks.length === 0)
    || (!homomer && lengths.length > 1 && paired.some((chunks) => chunks.length === 0))) {
    throw new Error("ColabFold complex-A3M is missing a paired or unpaired query row");
  }
  const chains = uniqueSequences.flatMap((sequence, index) =>
    Array.from({ length: cardinalities[index]! }, () => sequence));
  if (chains.length < 2) throw new Error("ColabFold complex-A3M does not describe a multimer");
  const assembled = assembleComplexA3m(
    chains, uniqueSequences, unpaired.map((chunks) => chunks.join("")),
    homomer ? undefined : paired.map((chunks) => chunks.join("")),
  );
  return { chains, uniqueSequences, cardinalities, ...assembled };
}
