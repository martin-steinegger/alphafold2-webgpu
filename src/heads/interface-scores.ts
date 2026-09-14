/**
 * ipSAE and pDockQ2: how confident each interface is, not the whole complex.
 *
 * pTM and ipTM average over every pair that crosses a chain boundary, so a
 * small interface on a large chain scores low however good it is. ipSAE drops
 * pairs above a predicted-error cutoff, sizes the TM-score d0 by how many
 * survive, and keeps the best anchor residue. pDockQ2 is a logistic fit of the
 * mean pLDDT and the mean error over the residue contacts.
 *
 * From ColabFold's colabfold/alphafold/ipsae.py, itself a reimplementation of
 * github.com/DunbrackLab/IPSAE. Its pDockQ, which pDockQ2 supersedes and which
 * ColabFold does not report, is left out.
 */

import { ATOM_CA, ATOM_CB, ATOM_TYPE_COUNT } from "../input/residue-atoms.js";

/** Predicted aligned error, in Angstrom, above which a pair does not count. */
const PAE_CUTOFF = 15;

/** CB separation, in Angstrom, at or below which two residues are in contact. */
const CONTACT_DISTANCE = 8;

/** The chain letters a pair is named by, as in the PDB this prediction writes. */
const CHAIN_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

export interface InterfaceScoreInput {
  /** Predicted aligned error in Angstrom, [length, length], anchor major. */
  readonly predictedAlignedError: Float32Array;
  readonly plddt: Float32Array;
  /** One positive integer chain identifier a residue, as the multimer builds it. */
  readonly asymId: Float32Array;
  /** Final atom37 coordinates, [length, 37, 3]. */
  readonly atom37: Float32Array;
  readonly atom37Mask: Float32Array;
}

/**
 * Both scores, for each ordered chain pair, named "A-B".
 *
 * Ordered because both read the error from one chain to the other, which is
 * not symmetric. See interfaceMaxima for the one number an interface.
 */
export interface InterfaceScores {
  readonly ipsae: Readonly<Record<string, number>>;
  readonly pdockq2: Readonly<Record<string, number>>;
}

/**
 * The TM-score kernel 1 / (1 + (error / d0)^2), taking 1 / d0^2.
 *
 * It runs once a residue pair, so it takes the reciprocal the caller already
 * has rather than dividing by d0 here.
 */
const tmTerm = (error: number, inverseSquareD0: number): number => 1 / (1 + error * error * inverseSquareD0);

/** 1 / d0^2 for an alignment of that many residues, with TM-score's floor on d0. */
function inverseSquareD0(residues: number): number {
  const d0 = Math.max(1, 1.24 * Math.cbrt(Math.max(26, residues) - 15) - 1.8);
  return 1 / (d0 * d0);
}

/**
 * CB of every residue, or CA where there is no CB, and which residues are placed.
 *
 * Glycine has no CB, and the structure module leaves the atom unplaced rather
 * than inventing one, so its CA stands in. A residue whose CA itself is
 * unplaced has no position at all and takes part in no contact.
 */
function betaCarbons(atom37: Float32Array, atom37Mask: Float32Array, length: number): {
  readonly coordinates: Float64Array; readonly placed: Uint8Array;
} {
  const coordinates = new Float64Array(length * 3);
  const placed = new Uint8Array(length);
  for (let residue = 0; residue < length; residue += 1) {
    const mask = residue * ATOM_TYPE_COUNT;
    if (atom37Mask[mask + ATOM_CA]! <= 0.5) continue;
    const atom = atom37Mask[mask + ATOM_CB]! > 0.5 ? ATOM_CB : ATOM_CA;
    const source = (residue * ATOM_TYPE_COUNT + atom) * 3;
    coordinates[residue * 3] = atom37[source]!;
    coordinates[residue * 3 + 1] = atom37[source + 1]!;
    coordinates[residue * 3 + 2] = atom37[source + 2]!;
    placed[residue] = 1;
  }
  return { coordinates, placed };
}

/**
 * Where each chain starts, and where the last one ends.
 *
 * The features concatenate the chains, so a chain's residues are contiguous.
 * Every loop below walks these ranges rather than asking each residue which
 * chain it is in, which is what makes the quadratic passes cheap.
 */
function chainBounds(asymId: Float32Array): Int32Array {
  const bounds = [0];
  for (let residue = 1; residue < asymId.length; residue += 1) {
    if (asymId[residue] !== asymId[residue - 1]) bounds.push(residue);
  }
  bounds.push(asymId.length);
  if (new Set(asymId).size !== bounds.length - 1) {
    throw new RangeError("each chain must hold one run of residues");
  }
  return Int32Array.from(bounds);
}

/**
 * ipSAE of every ordered chain pair.
 *
 * One anchor residue and one partner chain at a time, because d0 is sized by
 * that anchor's own count of partners under the cutoff: the range is read once
 * to count them and once more to sum the TM terms that count gives.
 */
function ipsaeByPair(
  pae: Float32Array, bounds: Int32Array, chains: number, length: number, cutoff: number,
): Float64Array {
  const best = new Float64Array(chains * chains);
  for (let anchorChain = 0; anchorChain < chains; anchorChain += 1) {
    for (let anchor = bounds[anchorChain]!; anchor < bounds[anchorChain + 1]!; anchor += 1) {
      const row = anchor * length;
      for (let chain = 0; chain < chains; chain += 1) {
        if (chain === anchorChain) continue;
        const start = bounds[chain]!, end = bounds[chain + 1]!;
        let count = 0;
        for (let partner = start; partner < end; partner += 1) {
          if (pae[row + partner]! < cutoff) count += 1;
        }
        if (count === 0) continue;
        const scale = inverseSquareD0(count);
        let sum = 0;
        for (let partner = start; partner < end; partner += 1) {
          const error = pae[row + partner]!;
          if (error < cutoff) sum += tmTerm(error, scale);
        }
        const index = anchorChain * chains + chain;
        best[index] = Math.max(best[index]!, sum / count);
      }
    }
  }
  return best;
}

/**
 * Gives ipSAE and pDockQ2 for every chain pair, or undefined for one chain.
 *
 * The scores match ColabFold's, except that the chains are named from their own
 * order: ColabFold indexes the PDB chain letters with the one-based asym_id, so
 * its first chain is called B.
 */
export function interfaceScores(input: InterfaceScoreInput): InterfaceScores | undefined {
  const { predictedAlignedError: pae, plddt, asymId, atom37, atom37Mask } = input;
  const length = asymId.length;
  if (plddt.length !== length || pae.length !== length * length) {
    throw new RangeError("the confidence outputs do not match the chain identifiers");
  }
  if (atom37.length !== length * ATOM_TYPE_COUNT * 3 || atom37Mask.length !== length * ATOM_TYPE_COUNT) {
    throw new RangeError("the atom37 output does not match the chain identifiers");
  }
  const bounds = chainBounds(asymId);
  const chains = bounds.length - 1;
  if (chains < 2) return undefined;
  if (chains > CHAIN_LABELS.length) throw new RangeError(`${chains} chains have no letters to name them`);

  const best = ipsaeByPair(pae, bounds, chains, length, PAE_CUTOFF);

  // Contacts, which are symmetric, and the error across them, which is not.
  // Each unordered pair of chains is walked once, the earlier chain outermost.
  const { coordinates, placed } = betaCarbons(atom37, atom37Mask, length);
  const contacts = new Float64Array(chains * chains);
  const contactTm = new Float64Array(chains * chains);
  // Whether a residue is in contact with any residue of a chain.
  const touches = new Uint8Array(length * chains);
  const limit = CONTACT_DISTANCE * CONTACT_DISTANCE;
  for (let chain = 0; chain < chains; chain += 1) {
    for (let residue = bounds[chain]!; residue < bounds[chain + 1]!; residue += 1) {
      if (placed[residue] === 0) continue;
      const x = coordinates[residue * 3]!, y = coordinates[residue * 3 + 1]!, z = coordinates[residue * 3 + 2]!;
      for (let partnerChain = chain + 1; partnerChain < chains; partnerChain += 1) {
        for (let partner = bounds[partnerChain]!; partner < bounds[partnerChain + 1]!; partner += 1) {
          if (placed[partner] === 0) continue;
          const dx = x - coordinates[partner * 3]!;
          const dy = y - coordinates[partner * 3 + 1]!;
          const dz = z - coordinates[partner * 3 + 2]!;
          if (dx * dx + dy * dy + dz * dz > limit) continue;
          contacts[chain * chains + partnerChain]! += 1;
          contacts[partnerChain * chains + chain]! += 1;
          // pDockQ2 fixes d0 at 10 A, so the kernel takes 1 / 100.
          contactTm[chain * chains + partnerChain]! += tmTerm(pae[residue * length + partner]!, 0.01);
          contactTm[partnerChain * chains + chain]! += tmTerm(pae[partner * length + residue]!, 0.01);
          touches[residue * chains + partnerChain] = 1;
          touches[partner * chains + chain] = 1;
        }
      }
    }
  }

  const ipsae: Record<string, number> = {};
  const pdockq2: Record<string, number> = {};
  for (let first = 0; first < chains; first += 1) {
    for (let second = 0; second < chains; second += 1) {
      if (first === second) continue;
      const name = `${CHAIN_LABELS[first]}-${CHAIN_LABELS[second]}`;
      ipsae[name] = best[first * chains + second]!;
      const count = contacts[first * chains + second]!;
      if (count === 0) { pdockq2[name] = 0; continue; }
      // pDockQ2 reads the pLDDT of every residue in the interface, which is the
      // residues of either chain that touch the other one.
      let plddtSum = 0; let interfaceResidues = 0;
      for (const [chain, other] of [[first, second], [second, first]] as const) {
        for (let residue = bounds[chain]!; residue < bounds[chain + 1]!; residue += 1) {
          if (touches[residue * chains + other] === 0) continue;
          plddtSum += plddt[residue]!; interfaceResidues += 1;
        }
      }
      const meanError = contactTm[first * chains + second]! / count;
      pdockq2[name] = 1.31 / (1 + Math.exp(-0.075 * (plddtSum / interfaceResidues * meanError - 84.733))) + 0.005;
    }
  }
  return { ipsae, pdockq2 };
}

/**
 * One number an interface, from the two directions ipSAE and pDockQ2 give it.
 *
 * Both are asymmetric because they read the error from one chain to the other,
 * and the convention is to report the better direction. ColabFold prints these
 * maxima beside pLDDT and pTM, in the same order.
 */
export function interfaceMaxima(scores: Readonly<Record<string, number>>): Record<string, number> {
  const pairs: Record<string, number> = {};
  for (const [name, value] of Object.entries(scores)) {
    const [first, second] = name.split("-") as [string, string];
    const key = first < second ? name : `${second}-${first}`;
    pairs[key] = Math.max(pairs[key] ?? 0, value);
  }
  return pairs;
}
