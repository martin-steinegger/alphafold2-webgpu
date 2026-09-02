import { CLUSTERED_MSA_CHANNELS } from "./msa-features.js";
import type { QueryOnlyFeatureTables } from "./query-only-features.js";
import {
  iterateA3mFeatures, recycleFeatureSource, type A3mFeatureOptions, type RecycleFeatureSource,
} from "./a3m-features.js";
import { parseA3m } from "./a3m.js";
import type { MonomerRecycleFeatures } from "../model/monomer.js";

const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
const RESTYPE_INDEX = new Map([...RESTYPES].map((residue, index) => [residue, index]));
export const MULTIMER_MAX_RELATIVE_INDEX = 32;
export const MULTIMER_MAX_RELATIVE_CHAIN = 2;
export const MULTIMER_RELATIVE_CHANNELS = 73;

export interface MultimerSequenceFeatures {
  readonly sequence: string;
  readonly chains: readonly string[];
  readonly chainLengths: readonly number[];
  readonly aatype: Float32Array;
  readonly targetFeatures: Float32Array;
  readonly residueIndex: Float32Array;
  readonly asymId: Float32Array;
  readonly entityId: Float32Array;
  readonly symId: Float32Array;
  readonly seqMask: Float32Array;
  readonly atom37ToAtom14: Float32Array;
  readonly atom37Mask: Float32Array;
}

export interface MultimerFeatureOptions {
  /** Number of recycling passes after the initial pass. Multimer-v3 defaults to 20. */
  readonly recycles?: number;
  readonly randomSeed?: number;
}

export interface MultimerRecycleFeatures extends MonomerRecycleFeatures {
  readonly chainRelative: {
    readonly asymId: Float32Array; readonly entityId: Float32Array; readonly symId: Float32Array;
  };
}

function randomGenerator(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function validatedChains(value: string | readonly string[]): readonly string[] {
  const chains = (typeof value === "string" ? value.split(":") : [...value])
    .map((chain) => chain.replace(/\s+/g, "").toUpperCase());
  if (chains.length < 2) throw new RangeError("AlphaFold-Multimer requires at least two chains");
  for (const chain of chains) {
    if (chain.length === 0 || [...chain].some((residue) => residue !== "X" && !RESTYPE_INDEX.has(residue))) {
      throw new Error("each multimer chain must contain only the 20 standard amino acids or X");
    }
  }
  const length = chains.reduce((sum, chain) => sum + chain.length, 0);
  if (!Number.isSafeInteger(length) || length <= 0) throw new RangeError("multimer length is invalid");
  return chains;
}

/** Reproduce AlphaFold-Multimer's asym/entity/sym identifiers for ordered chains. */
export function multimerChainIdentifiers(chainsValue: string | readonly string[]): {
  readonly chains: readonly string[];
  readonly asymId: Float32Array;
  readonly entityId: Float32Array;
  readonly symId: Float32Array;
  readonly residueIndex: Float32Array;
} {
  const chains = validatedChains(chainsValue);
  const length = chains.reduce((sum, chain) => sum + chain.length, 0);
  const asymId = new Float32Array(length);
  const entityId = new Float32Array(length);
  const symId = new Float32Array(length);
  const residueIndex = new Float32Array(length);
  const entities = new Map<string, number>();
  const copies = new Map<number, number>();
  let offset = 0;
  for (let chainIndex = 0; chainIndex < chains.length; chainIndex += 1) {
    const chain = chains[chainIndex]!;
    let entity = entities.get(chain);
    if (entity === undefined) { entity = entities.size + 1; entities.set(chain, entity); }
    const symmetry = (copies.get(entity) ?? 0) + 1;
    copies.set(entity, symmetry);
    for (let residue = 0; residue < chain.length; residue += 1) {
      const index = offset + residue;
      asymId[index] = chainIndex + 1;
      entityId[index] = entity;
      symId[index] = symmetry;
      residueIndex[index] = residue;
    }
    offset += chain.length;
  }
  return { chains, asymId, entityId, symId, residueIndex };
}

/** Build the exact 66+1+6 chain-relative feature channels used by multimer-v3. */
export function multimerRelativeFeatures(
  residueIndex: Float32Array,
  asymId: Float32Array,
  entityId: Float32Array,
  symId: Float32Array,
): Float32Array {
  const length = residueIndex.length;
  if (length === 0 || asymId.length !== length || entityId.length !== length || symId.length !== length) {
    throw new RangeError("multimer identifiers must be non-empty arrays of equal length");
  }
  for (const values of [residueIndex, asymId, entityId, symId]) {
    if (values.some((value) => !Number.isSafeInteger(value) || value < 0)) {
      throw new RangeError("multimer identifiers must contain non-negative integers");
    }
  }
  const output = new Float32Array(length * length * MULTIMER_RELATIVE_CHANNELS);
  for (let i = 0; i < length; i += 1) for (let j = 0; j < length; j += 1) {
    const base = (i * length + j) * MULTIMER_RELATIVE_CHANNELS;
    const sameChain = asymId[i] === asymId[j];
    const relative = sameChain
      ? Math.max(0, Math.min(2 * MULTIMER_MAX_RELATIVE_INDEX,
        residueIndex[i]! - residueIndex[j]! + MULTIMER_MAX_RELATIVE_INDEX))
      : 2 * MULTIMER_MAX_RELATIVE_INDEX + 1;
    output[base + relative] = 1;
    const sameEntity = entityId[i] === entityId[j];
    output[base + 66] = sameEntity ? 1 : 0;
    const relativeChain = sameEntity
      ? Math.max(0, Math.min(2 * MULTIMER_MAX_RELATIVE_CHAIN,
        symId[i]! - symId[j]! + MULTIMER_MAX_RELATIVE_CHAIN))
      : 2 * MULTIMER_MAX_RELATIVE_CHAIN + 1;
    output[base + 67 + relativeChain] = 1;
  }
  return output;
}

/** Construct sequence-level features required at the multimer-v3 public boundary. */
export function makeMultimerSequenceFeatures(
  chainsValue: string | readonly string[],
  tables: QueryOnlyFeatureTables,
): MultimerSequenceFeatures {
  if (tables.atom37ToAtom14.length !== 21 * 37 || tables.atom37Mask.length !== 21 * 37) {
    throw new RangeError("residue feature tables must have shape [21, 37]");
  }
  const identifiers = multimerChainIdentifiers(chainsValue);
  const sequence = identifiers.chains.join("");
  const length = sequence.length;
  const aatype = Float32Array.from(sequence, (residue) => RESTYPE_INDEX.get(residue) ?? 20);
  const targetFeatures = new Float32Array(length * 21);
  const atom37ToAtom14 = new Float32Array(length * 37);
  const atom37Mask = new Float32Array(length * 37);
  for (let residue = 0; residue < length; residue += 1) {
    const aa = aatype[residue]!;
    targetFeatures[residue * 21 + aa] = 1;
    atom37ToAtom14.set(tables.atom37ToAtom14.subarray(aa * 37, (aa + 1) * 37), residue * 37);
    atom37Mask.set(tables.atom37Mask.subarray(aa * 37, (aa + 1) * 37), residue * 37);
  }
  return {
    sequence,
    chains: identifiers.chains,
    chainLengths: identifiers.chains.map((chain) => chain.length),
    aatype,
    targetFeatures,
    residueIndex: identifiers.residueIndex,
    asymId: identifiers.asymId,
    entityId: identifiers.entityId,
    symId: identifiers.symId,
    seqMask: new Float32Array(length).fill(1),
    atom37ToAtom14,
    atom37Mask,
  };
}

/**
 * Build a deterministic, no-MSA Multimer-v3 input. Chains remain distinct
 * through asym/entity/sym identifiers even though their query rows are joined.
 */
export function iterateMultimerQueryOnlyFeatures(
  chainsValue: string | readonly string[],
  tables: QueryOnlyFeatureTables,
  options: MultimerFeatureOptions = {},
): RecycleFeatureSource<MultimerRecycleFeatures> {
  const sequence = makeMultimerSequenceFeatures(chainsValue, tables);
  const length = sequence.aatype.length;
  const recycles = options.recycles ?? 20;
  if (!Number.isSafeInteger(recycles) || recycles < 0) {
    throw new RangeError("recycles must be a non-negative safe integer");
  }
  return recycleFeatureSource(recycles + 1, function* features() {
    for (let recycle = 0; recycle <= recycles; recycle += 1) {
    const random = randomGenerator(((options.randomSeed ?? 0) ^ Math.imul(recycle + 1, 0x9e3779b9)) >>> 0);
    const msaCodes = sequence.aatype.slice();
    for (let residue = 0; residue < length; residue += 1) {
      if (random() >= 0.15) continue;
      const draw = random();
      if (draw < 0.7) msaCodes[residue] = 22;
      else if (draw >= 0.9) msaCodes[residue] = Math.floor(random() * 20);
    }
    const msaFeatures = new Float32Array(length * CLUSTERED_MSA_CHANNELS);
    for (let residue = 0; residue < length; residue += 1) {
      const code = msaCodes[residue]!;
      msaFeatures[residue * CLUSTERED_MSA_CHANNELS] = code;
      msaFeatures[residue * CLUSTERED_MSA_CHANNELS + 3 + code] = 1 / (1 + 1e-6);
    }
    yield {
      targetFeatures: sequence.targetFeatures,
      msaFeatures,
      msaMask: new Float32Array(length).fill(1),
      extraMsa: new Float32Array(length),
      extraHasDeletion: new Float32Array(length),
      extraDeletionValue: new Float32Array(length),
      extraMsaMask: new Float32Array(length),
      residueIndex: sequence.residueIndex,
      aatype: sequence.aatype,
      seqMask: sequence.seqMask,
      atom37ToAtom14: sequence.atom37ToAtom14,
      atom37Mask: sequence.atom37Mask,
      msaSequences: 1,
      extraSequences: 1,
      targetChannels: 21,
      msaFeatureChannels: CLUSTERED_MSA_CHANNELS,
      chainRelative: {
        asymId: sequence.asymId, entityId: sequence.entityId, symId: sequence.symId,
      },
    };
    }
  });
}

/** Eager compatibility wrapper for callers that need random access. */
export function makeMultimerQueryOnlyFeatures(
  chainsValue: string | readonly string[],
  tables: QueryOnlyFeatureTables,
  options: MultimerFeatureOptions = {},
): readonly MultimerRecycleFeatures[] {
  return [...iterateMultimerQueryOnlyFeatures(chainsValue, tables, options)];
}

/** Build Multimer-v3 tensors from ColabFold-style paired/unpaired complex MSA rows. */
export function iterateMultimerA3mFeatures(
  chainsValue: string | readonly string[],
  a3mText: string,
  alignmentMask: Float32Array,
  tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions = {},
): RecycleFeatureSource<MultimerRecycleFeatures> {
  const sequence = makeMultimerSequenceFeatures(chainsValue, tables);
  const alignment = parseA3m(a3mText);
  if (alignment.query !== sequence.sequence) {
    throw new Error("complex A3M query does not match the concatenated input chains");
  }
  if (alignmentMask.length !== alignment.depth * alignment.length) {
    throw new RangeError("complex MSA mask must have shape [depth, total residues]");
  }
  const source = iterateA3mFeatures(a3mText, tables, {
    maxExtraSequences: 2048,
    colabFoldMultimerProcess: true,
    ...options,
    alignmentMask,
  });
  return recycleFeatureSource(source.length, function* features() {
    for (const recycle of source) yield {
      ...recycle,
      targetFeatures: sequence.targetFeatures,
      residueIndex: sequence.residueIndex,
      aatype: sequence.aatype,
      seqMask: sequence.seqMask,
      atom37ToAtom14: sequence.atom37ToAtom14,
      atom37Mask: sequence.atom37Mask,
      targetChannels: 21,
      chainRelative: {
        asymId: sequence.asymId, entityId: sequence.entityId, symId: sequence.symId,
      },
    };
  });
}

/** Eager compatibility wrapper for callers that need random access. */
export function makeMultimerA3mFeatures(
  chainsValue: string | readonly string[],
  a3mText: string,
  alignmentMask: Float32Array,
  tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions = {},
): readonly MultimerRecycleFeatures[] {
  return [...iterateMultimerA3mFeatures(chainsValue, a3mText, alignmentMask, tables, options)];
}
