import { alignSequences, type PairwiseAlignment } from "./pairwise-align.js";
import {
  ATOM_C, ATOM_CA, ATOM_CB, ATOM_N, ATOM_TYPE_COUNT, RESTYPES, RESTYPE_GAP,
} from "./residue-atoms.js";
import type { StructureChain } from "./structure.js";

/**
 * AlphaFold's per-template features, indexed by query position.
 *
 * Everything here is what templates.py produces for one hit, reached without
 * a hit: the query is aligned to the chain, and each query position takes the
 * template residue aligned to it or a gap. A gap contributes nothing — its
 * aatype is 21, its coordinates are zero and every mask over it is zero — so
 * the model sees the covered part of the query and nothing invented for the
 * rest.
 */
export interface TemplateFeatures {
  readonly length: number;
  /** Template residue per query position; 21 where nothing aligned. */
  readonly aatype: Int32Array;
  /** [length, 37, 3], zero where nothing aligned. */
  readonly atomPositions: Float32Array;
  /** [length, 37]. */
  readonly atomMask: Float32Array;
  /** CB, or CA for glycine, [length, 3]. */
  readonly pseudoBeta: Float32Array;
  readonly pseudoBetaMask: Float32Array;
  /** 1 where N, CA and C are all present, which is what a backbone frame needs. */
  readonly backboneMask: Float32Array;
  readonly alignment: PairwiseAlignment;
}

/** Alanine is the smallest residue at five atoms; fewer than that is not a structure. */
const MINIMUM_ATOMS = 5;

const GLYCINE = RESTYPES.indexOf("G");

export interface TemplateFeatureOptions {
  /** Reject a template covering less of the query than this. */
  readonly minimumCoverage?: number;
}

/**
 * Builds the features for one template chain against one query sequence.
 *
 * query is the ungapped query, one letter per residue. For a complex this is
 * one chain's sequence: templates are per chain, and the caller places the
 * result into the concatenated query's window.
 */
export function templateFeatures(
  query: string, chain: StructureChain, options: TemplateFeatureOptions = {},
): TemplateFeatures {
  const length = query.length;
  const alignment = alignSequences(query, chain.sequence);
  const minimum = options.minimumCoverage ?? 0;
  if (alignment.coverage < minimum) {
    throw new Error(
      `Template chain ${chain.id} covers ${(alignment.coverage * 100).toFixed(0)}% of the query, `
      + `below the ${(minimum * 100).toFixed(0)}% this run asks for.`,
    );
  }

  const aatype = new Int32Array(length).fill(RESTYPE_GAP);
  const atomPositions = new Float32Array(length * ATOM_TYPE_COUNT * 3);
  const atomMask = new Float32Array(length * ATOM_TYPE_COUNT);
  const pseudoBeta = new Float32Array(length * 3);
  const pseudoBetaMask = new Float32Array(length);
  const backboneMask = new Float32Array(length);

  let atoms = 0;
  for (let position = 0; position < length; position += 1) {
    const source = alignment.queryToTemplate[position]!;
    if (source < 0) continue;
    const letter = chain.sequence[source]!;
    const code = RESTYPES.indexOf(letter);
    aatype[position] = code < 0 ? RESTYPES.length : code;
    atomPositions.set(
      chain.atomPositions.subarray(source * ATOM_TYPE_COUNT * 3, (source + 1) * ATOM_TYPE_COUNT * 3),
      position * ATOM_TYPE_COUNT * 3,
    );
    const mask = chain.atomMask.subarray(source * ATOM_TYPE_COUNT, (source + 1) * ATOM_TYPE_COUNT);
    atomMask.set(mask, position * ATOM_TYPE_COUNT);
    for (const value of mask) atoms += value;

    // Glycine has no CB, so it stands in with its CA, exactly as AlphaFold's
    // pseudo_beta_fn does.
    const atom = code === GLYCINE ? ATOM_CA : ATOM_CB;
    pseudoBetaMask[position] = mask[atom]!;
    for (let axis = 0; axis < 3; axis += 1) {
      pseudoBeta[position * 3 + axis] = chain.atomPositions[(source * ATOM_TYPE_COUNT + atom) * 3 + axis]!;
    }
    backboneMask[position] = mask[ATOM_N]! * mask[ATOM_CA]! * mask[ATOM_C]!;
  }

  if (atoms < MINIMUM_ATOMS) {
    throw new Error(`Template chain ${chain.id} contributes no atoms to this query.`);
  }
  return {
    length, aatype, atomPositions, atomMask, pseudoBeta, pseudoBetaMask, backboneMask, alignment,
  };
}
