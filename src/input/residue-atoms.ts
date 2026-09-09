/**
 * The residue and atom naming AlphaFold's template features are written in.
 *
 * These are residue_constants.atom_types, restype_3to1 and restypes from
 * the official implementation, transcribed rather than derived: the order of
 * ATOM_TYPES is the atom37 axis every template tensor is indexed by, so a
 * single transposition here would misplace coordinates silently. The test
 * beside this file checks the ones the template pair features actually read.
 */

export const ATOM_TYPES = [
  "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1",
  "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE",
  "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3",
  "NZ", "OXT",
] as const;

export const ATOM_TYPE_COUNT = ATOM_TYPES.length;

export const ATOM_INDEX: ReadonlyMap<string, number> = new Map(
  ATOM_TYPES.map((atom, index) => [atom, index]),
);

/** Backbone atoms the template frame mask and the pseudo-beta both depend on. */
export const ATOM_N = 0;
export const ATOM_CA = 1;
export const ATOM_C = 2;
export const ATOM_CB = 3;

/** AlphaFold's residue order; index 20 is unknown and 21 is a gap. */
export const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
export const RESTYPE_UNKNOWN = 20;
export const RESTYPE_GAP = 21;

export const RESTYPE_INDEX: ReadonlyMap<string, number> = new Map(
  [...RESTYPES].map((residue, index) => [residue, index]),
);

const THREE_TO_ONE_ENTRIES = [
  ["ALA", "A"], ["ARG", "R"], ["ASN", "N"], ["ASP", "D"], ["CYS", "C"],
  ["GLN", "Q"], ["GLU", "E"], ["GLY", "G"], ["HIS", "H"], ["ILE", "I"],
  ["LEU", "L"], ["LYS", "K"], ["MET", "M"], ["PHE", "F"], ["PRO", "P"],
  ["SER", "S"], ["THR", "T"], ["TRP", "W"], ["TYR", "Y"], ["VAL", "V"],
] as const;

export const RESIDUE_THREE_TO_ONE: ReadonlyMap<string, string> = new Map(THREE_TO_ONE_ENTRIES);

/**
 * Modified residues a structure may carry in place of a standard one.
 *
 * The mmCIF path in AlphaFold resolves these through the file's own
 * _pdbx_struct_mod_residue records, which a PDB file does not have. These are
 * the substitutions common enough to matter in practice: selenomethionine above
 * all, which is in a large share of crystal structures and would otherwise cost
 * the template every methionine it has.
 */
const MODIFIED_RESIDUES: ReadonlyMap<string, string> = new Map([
  ["MSE", "MET"], ["SEC", "CYS"], ["PYL", "LYS"], ["HYP", "PRO"], ["SEP", "SER"],
  ["TPO", "THR"], ["PTR", "TYR"], ["CSO", "CYS"], ["CME", "CYS"], ["MLY", "LYS"],
  ["M3L", "LYS"], ["KCX", "LYS"], ["LLP", "LYS"], ["CAS", "CYS"], ["OCS", "CYS"],
  ["CSD", "CYS"], ["MHO", "MET"], ["FME", "MET"], ["AYA", "ALA"], ["DAL", "ALA"],
]);

/** The one-letter code for a residue name, or undefined when it is not a residue. */
export function residueLetter(name: string): string | undefined {
  const upper = name.trim().toUpperCase();
  const standard = RESIDUE_THREE_TO_ONE.get(upper);
  if (standard !== undefined) return standard;
  const substitute = MODIFIED_RESIDUES.get(upper);
  return substitute === undefined ? undefined : RESIDUE_THREE_TO_ONE.get(substitute);
}

/** Whether a residue name is a modified form standing in for a standard residue. */
export function isModifiedResidue(name: string): boolean {
  return MODIFIED_RESIDUES.has(name.trim().toUpperCase());
}
