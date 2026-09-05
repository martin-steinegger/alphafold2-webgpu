import {
  ATOM_INDEX, ATOM_TYPE_COUNT, isModifiedResidue, residueLetter,
} from "./residue-atoms.js";

/**
 * One polymer chain read out of an uploaded structure.
 *
 * The sequence and the coordinates come from the same records, so they cannot
 * disagree. This is the reason AlphaFold needs kalign and we do not: it takes
 * the template's sequence from the mmCIF's `_entity_poly_seq` and its
 * coordinates from the model, and when a structure's SEQRES does not line up
 * with the residues it actually resolved, the two have to be realigned before
 * anything can be indexed. Reading only the residues that have atoms leaves
 * nothing to reconcile. Residues a structure did not resolve are simply absent,
 * which the query-to-template alignment absorbs as a gap, and the geometry
 * stays true because every distance is measured between coordinates that exist.
 */
export interface StructureChain {
  readonly id: string;
  /** One letter per resolved residue, in the order the file lists them. */
  readonly sequence: string;
  /** Author residue numbers, for reporting which part of a structure was used. */
  readonly residueNumbers: Int32Array;
  /** Coordinates on AlphaFold's 37-atom axis, `[residues, 37, 3]`. */
  readonly atomPositions: Float32Array;
  /** 1 where an atom was present, `[residues, 37]`. */
  readonly atomMask: Float32Array;
}

export interface ParsedStructure {
  readonly chains: readonly StructureChain[];
  readonly format: "pdb" | "mmcif";
}

interface ResidueRecord {
  readonly key: string;
  readonly chain: string;
  readonly number: number;
  readonly letter: string;
  readonly positions: Float32Array;
  readonly mask: Float32Array;
}

class ChainBuilder {
  readonly residues: ResidueRecord[] = [];
  readonly byKey = new Map<string, ResidueRecord>();

  add(chain: string, key: string, number: number, letter: string): ResidueRecord {
    const existing = this.byKey.get(key);
    if (existing !== undefined) return existing;
    const record: ResidueRecord = {
      key, chain, number, letter,
      positions: new Float32Array(ATOM_TYPE_COUNT * 3), mask: new Float32Array(ATOM_TYPE_COUNT),
    };
    this.byKey.set(key, record);
    this.residues.push(record);
    return record;
  }
}

function place(
  record: ResidueRecord, atomName: string, x: number, y: number, z: number,
): void {
  const atom = ATOM_INDEX.get(atomName);
  // Hydrogens, deuteriums and anything a modified residue carries beyond the
  // standard set have no place on the 37-atom axis and are dropped.
  if (atom === undefined || record.mask[atom] === 1) return;
  record.mask[atom] = 1;
  record.positions[atom * 3] = x;
  record.positions[atom * 3 + 1] = y;
  record.positions[atom * 3 + 2] = z;
}

function assemble(builder: ChainBuilder): StructureChain[] {
  const byChain = new Map<string, ResidueRecord[]>();
  for (const residue of builder.residues) {
    const list = byChain.get(residue.chain);
    if (list === undefined) byChain.set(residue.chain, [residue]); else list.push(residue);
  }
  const chains: StructureChain[] = [];
  for (const [id, residues] of byChain) {
    if (residues.length === 0) continue;
    const positions = new Float32Array(residues.length * ATOM_TYPE_COUNT * 3);
    const mask = new Float32Array(residues.length * ATOM_TYPE_COUNT);
    const numbers = new Int32Array(residues.length);
    residues.forEach((residue, index) => {
      positions.set(residue.positions, index * ATOM_TYPE_COUNT * 3);
      mask.set(residue.mask, index * ATOM_TYPE_COUNT);
      numbers[index] = residue.number;
    });
    chains.push({
      id, sequence: residues.map((residue) => residue.letter).join(""),
      residueNumbers: numbers, atomPositions: positions, atomMask: mask,
    });
  }
  return chains;
}

/** Whether a record's residue name is one this reads, given where it came from. */
function acceptedResidue(name: string, heteroAtom: boolean): string | undefined {
  const letter = residueLetter(name);
  if (letter === undefined) return undefined;
  // A HETATM is only a residue when it is a modified form of one; otherwise it
  // is a ligand, an ion or a water and does not belong in a polymer chain.
  if (heteroAtom && !isModifiedResidue(name)) return undefined;
  return letter;
}

function parsePdb(text: string): StructureChain[] {
  const builder = new ChainBuilder();
  for (const line of text.split(/\r?\n/)) {
    const record = line.slice(0, 6);
    // Only the first model: an NMR ensemble's later models are the same chain
    // again, and appending them would make one residue appear many times.
    if (record === "ENDMDL") break;
    const heteroAtom = record === "HETATM";
    if (record !== "ATOM  " && !heteroAtom) continue;
    if (line.length < 54) continue;
    const alternate = line[16] ?? " ";
    if (alternate !== " " && alternate !== "A") continue;
    const letter = acceptedResidue(line.slice(17, 20), heteroAtom);
    if (letter === undefined) continue;
    const x = Number(line.slice(30, 38));
    const y = Number(line.slice(38, 46));
    const z = Number(line.slice(46, 54));
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    const chain = (line[21] ?? " ").trim() || "A";
    const number = Number.parseInt(line.slice(22, 26), 10);
    // The key carries the insertion code, so 52A and 52B stay separate residues.
    const residue = builder.add(chain, `${chain}|${line.slice(22, 27)}`, number, letter);
    place(residue, line.slice(12, 16).trim(), x, y, z);
  }
  return assemble(builder);
}

/** Splits an mmCIF loop row, honouring the single and double quoting it allows. */
function cifFields(line: string): string[] {
  const fields: string[] = [];
  let index = 0;
  while (index < line.length) {
    const character = line[index]!;
    if (character === " " || character === "\t") { index += 1; continue; }
    if (character === "'" || character === '"') {
      const end = line.indexOf(character, index + 1);
      if (end === -1) { fields.push(line.slice(index + 1)); break; }
      fields.push(line.slice(index + 1, end));
      index = end + 1;
      continue;
    }
    let end = index;
    while (end < line.length && line[end] !== " " && line[end] !== "\t") end += 1;
    fields.push(line.slice(index, end));
    index = end;
  }
  return fields;
}

function parseMmcif(text: string): StructureChain[] {
  const lines = text.split(/\r?\n/);
  const builder = new ChainBuilder();
  let index = 0;
  while (index < lines.length) {
    if (lines[index]!.trim() !== "loop_") { index += 1; continue; }
    index += 1;
    const columns = new Map<string, number>();
    while (index < lines.length && lines[index]!.trimStart().startsWith("_")) {
      const tag = lines[index]!.trim();
      if (tag.startsWith("_atom_site.")) columns.set(tag.slice("_atom_site.".length), columns.size);
      else { columns.clear(); break; }
      index += 1;
    }
    if (columns.size === 0) continue;
    const column = (...names: readonly string[]): number => {
      for (const name of names) { const found = columns.get(name); if (found !== undefined) return found; }
      return -1;
    };
    const group = column("group_PDB");
    const atomName = column("label_atom_id", "auth_atom_id");
    const residueName = column("label_comp_id", "auth_comp_id");
    const chainName = column("auth_asym_id", "label_asym_id");
    const sequenceNumber = column("auth_seq_id", "label_seq_id");
    const insertion = column("pdbx_PDB_ins_code");
    const alternate = column("label_alt_id", "auth_alt_id");
    const model = column("pdbx_PDB_model_num");
    const xColumn = column("Cartn_x"); const yColumn = column("Cartn_y"); const zColumn = column("Cartn_z");
    if (atomName < 0 || residueName < 0 || xColumn < 0) { continue; }
    let firstModel: string | undefined;
    while (index < lines.length) {
      const line = lines[index]!;
      const trimmed = line.trim();
      if (trimmed === "" || trimmed === "#" || trimmed === "loop_" || trimmed.startsWith("_")) break;
      index += 1;
      const fields = cifFields(line);
      if (fields.length < columns.size) continue;
      if (model >= 0) {
        firstModel ??= fields[model];
        if (fields[model] !== firstModel) continue;
      }
      const alternateValue = alternate < 0 ? "." : fields[alternate]!;
      if (alternateValue !== "." && alternateValue !== "?" && alternateValue !== "A") continue;
      const heteroAtom = group >= 0 && fields[group] === "HETATM";
      const letter = acceptedResidue(fields[residueName]!, heteroAtom);
      if (letter === undefined) continue;
      const x = Number(fields[xColumn]); const y = Number(fields[yColumn]); const z = Number(fields[zColumn]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      const chain = chainName < 0 ? "A" : fields[chainName]!;
      const rawNumber = sequenceNumber < 0 ? "0" : fields[sequenceNumber]!;
      const insertionCode = insertion < 0 || fields[insertion] === "?" ? "" : fields[insertion]!;
      const residue = builder.add(
        chain, `${chain}|${rawNumber}|${insertionCode}`, Number.parseInt(rawNumber, 10) || 0, letter,
      );
      place(residue, fields[atomName]!.replace(/^["']|["']$/g, ""), x, y, z);
    }
  }
  return assemble(builder);
}

/**
 * Reads a template structure.
 *
 * The format is taken from the text rather than from a file extension, since a
 * `.pdb` holding mmCIF and the reverse are both things people upload.
 */
export function parseStructure(text: string): ParsedStructure {
  const mmcif = /^\s*(data_|#|loop_|_atom_site\.)/m.test(text) && /_atom_site\./.test(text);
  const chains = mmcif ? parseMmcif(text) : parsePdb(text);
  if (chains.length === 0) {
    throw new Error(
      mmcif
        ? "This mmCIF has no protein chain: no _atom_site rows named a standard amino acid."
        : "This PDB has no protein chain: no ATOM records named a standard amino acid.",
    );
  }
  return { chains, format: mmcif ? "mmcif" : "pdb" };
}
