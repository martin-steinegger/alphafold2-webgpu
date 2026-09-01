import type { ConfidenceResult } from "../src/heads/confidence.js";
import type { StructureModuleResult } from "../src/structure/module.js";

const RESIDUE_NAMES: Readonly<Record<string, string>> = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS",
  I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP",
  Y: "TYR", V: "VAL", X: "UNK",
};

// AlphaFold's atom37 order from residue_constants.py.
const ATOM_NAMES = [
  "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD", "CD1", "CD2",
  "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3", "NE", "NE1", "NE2", "OE1",
  "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2", "CZ3", "NZ", "OXT",
] as const;

function field(value: number, width: number, decimals: number): string {
  return value.toFixed(decimals).padStart(width);
}

const CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/** Serializes an AlphaFold atom37 result as PDB with pLDDT in the B-factor field. */
export function predictionToPdb(
  sequence: string,
  structure: StructureModuleResult,
  plddt: Float32Array,
  chainLengths: readonly number[] = [sequence.length],
): string {
  if (structure.atom37.length !== sequence.length * 37 * 3 || structure.atom37Mask.length !== sequence.length * 37) {
    throw new RangeError("atom37 output does not match the sequence length");
  }
  if (plddt.length !== sequence.length) throw new RangeError("pLDDT output does not match the sequence length");
  if (chainLengths.length === 0 || chainLengths.length > CHAIN_IDS.length
    || chainLengths.some((length) => !Number.isSafeInteger(length) || length <= 0)
    || chainLengths.reduce((sum, length) => sum + length, 0) !== sequence.length) {
    throw new RangeError("chain lengths must be positive integers that cover the sequence");
  }
  const lines = ["REMARK   1 ALPHAFOLD2 WEBGPU PREDICTION"];
  let serial = 1; let chain = 0; let chainStart = 0; let chainEnd = chainLengths[0]!;
  for (let residue = 0; residue < sequence.length; residue += 1) {
    if (residue === chainEnd) {
      lines.push("TER"); chain += 1; chainStart = chainEnd; chainEnd += chainLengths[chain]!;
    }
    const residueName = RESIDUE_NAMES[sequence[residue]!] ?? "UNK";
    for (let atom = 0; atom < ATOM_NAMES.length; atom += 1) {
      if (structure.atom37Mask[residue * 37 + atom]! < 0.5) continue;
      const offset = (residue * 37 + atom) * 3;
      const atomName = ATOM_NAMES[atom]!;
      const element = atomName[0]!;
      lines.push(
        `ATOM  ${String(serial).padStart(5)} ${atomName.padStart(4)} ${residueName} ${CHAIN_IDS[chain]}`
        + `${String(residue - chainStart + 1).padStart(4)}    `
        + `${field(structure.atom37[offset]!, 8, 3)}${field(structure.atom37[offset + 1]!, 8, 3)}`
        + `${field(structure.atom37[offset + 2]!, 8, 3)}  1.00${field(plddt[residue]!, 6, 2)}`
        + `          ${element.padStart(2)}`,
      );
      serial += 1;
    }
  }
  lines.push("TER", "END");
  return `${lines.join("\n")}\n`;
}

export function confidenceJson(
  sequence: string,
  confidence: Omit<ConfidenceResult, "lddtLogits" | "paeLogits">,
): string {
  const multimer = confidence as typeof confidence & { readonly iptm?: number; readonly rankingConfidence?: number };
  return JSON.stringify({
    sequence,
    plddt: Array.from(confidence.plddt),
    mean_plddt: confidence.meanPlddt,
    ptm: confidence.ptm,
    ...(multimer.iptm === undefined ? {} : { iptm: multimer.iptm }),
    ...(multimer.rankingConfidence === undefined ? {} : { ranking_confidence: multimer.rankingConfidence }),
    predicted_aligned_error: Array.from(confidence.predictedAlignedError),
    max_predicted_aligned_error: confidence.maxPredictedAlignedError,
  }, null, 2);
}

export function safeJobName(value: string): string {
  return value.trim().replace(/[^a-zA-Z0-9_.-]+/g, "_").replace(/^[_\.]+|[_\.]+$/g, "").slice(0, 80) || "prediction";
}
