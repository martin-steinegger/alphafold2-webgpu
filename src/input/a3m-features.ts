import { parseA3m } from "./a3m.js";
import {
  jaxPaddingConsistentUniform, multimerMsaKeys, type JaxKey,
} from "./jax-prng.js";
import { makeQueryOnlyFeatures, type QueryOnlyFeatureTables } from "./query-only-features.js";
import type { MonomerRecycleFeatures } from "../model/monomer.js";

const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
const INDEX = new Map([...RESTYPES].map((residue, index) => [residue, index]));

export interface A3mFeatureOptions {
  readonly recycles?: number; readonly randomSeed?: number;
  readonly maxMsaSequences?: number; readonly maxExtraSequences?: number;
  /** Use AlphaFold-Multimer's JAX sampling, masking, and clustering pipeline. */
  readonly colabFoldMultimerProcess?: boolean;
  /** Per-cell mask for masked-MSA augmentation; Multimer block padding remains visible as gaps to the model. */
  readonly alignmentMask?: Float32Array;
}

/** A replayable, counted feature stream that materializes one recycle at a time. */
export interface RecycleFeatureSource<T> extends Iterable<T> { readonly length: number; }

export function recycleFeatureSource<T>(
  length: number,
  iterator: () => Iterator<T>,
): RecycleFeatureSource<T> {
  return { length, [Symbol.iterator]: iterator };
}

function generator(seed: number): () => number {
  let state = seed >>> 0;
  return () => { state = (state + 0x6d2b79f5) >>> 0; let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1); value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296; };
}

function shuffle(values: number[], random: () => number): void {
  for (let index = values.length - 1; index > 0; index -= 1) {
    const other = Math.floor(random() * (index + 1)); [values[index], values[other]] = [values[other]!, values[index]!];
  }
}

function deletionValue(value: number): number { return Math.atan(value / 3) * 2 / Math.PI; }

function gumbel(key: JaxKey, indices: readonly number[]): number {
  const epsilon = 1e-6;
  const uniform = jaxPaddingConsistentUniform(key, indices);
  return -Math.log(-Math.log(uniform + epsilon) + epsilon);
}

function makeColabFoldMultimerFeatures(
  alignment: ReturnType<typeof parseA3m>,
  encodedInput: Uint8Array,
  tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions,
): RecycleFeatureSource<MonomerRecycleFeatures> {
  const length = alignment.length;
  const inputDepth = alignment.depth;
  const recycles = options.recycles ?? 3;
  const maxMsa = options.maxMsaSequences ?? 508;
  const maxExtra = options.maxExtraSequences ?? 2048;
  for (const [name, value] of [["recycles", recycles], ["maximum MSA sequences", maxMsa],
    ["maximum extra MSA sequences", maxExtra]] as const) {
    if (!Number.isSafeInteger(value) || value < (name === "recycles" ? 0 : 1)) {
      throw new RangeError(`${name} must be a ${name === "recycles" ? "non-negative" : "positive"} safe integer`);
    }
  }
  const depth = Math.max(inputDepth, maxMsa + 4);
  const encoded = new Uint8Array(depth * length);
  encoded.set(encodedInput);
  const deletionMatrix = new Float32Array(depth * length);
  for (let row = 0; row < inputDepth; row += 1) {
    deletionMatrix.set(alignment.deletionMatrix[row]!, row * length);
  }
  const rowMask = new Uint8Array(depth);
  rowMask.fill(1, 0, inputDepth);

  // Multimer computes the categorical masking profile over the complete raw MSA.
  const msaProfile = new Float32Array(length * 22);
  for (let row = 0; row < inputDepth; row += 1) for (let residue = 0; residue < length; residue += 1) {
    const slot = residue * 22 + encoded[row * length + residue]!;
    msaProfile[slot] = msaProfile[slot]! + 1;
  }
  for (let index = 0; index < msaProfile.length; index += 1) {
    msaProfile[index] = msaProfile[index]! / (inputDepth + 1e-10);
  }

  const base = makeQueryOnlyFeatures(alignment.query, tables, { recycles: 0, maskedMsaCodes: [
    Float32Array.from(encoded.subarray(0, length)),
  ] })[0]!;
  return recycleFeatureSource(recycles + 1, function* features() {
    let rootKey: JaxKey = [0, (options.randomSeed ?? 0) >>> 0];
    for (let recycle = 0; recycle <= recycles; recycle += 1) {
    const keys = multimerMsaKeys(rootKey); rootKey = keys.nextRoot;
    const order = Array.from({ length: depth }, (_, row) => row);
    order.sort((left, right) => {
      const leftBias = left === 0 ? 1 : 0; const rightBias = right === 0 ? 1 : 0;
      if (leftBias !== rightBias) return rightBias - leftBias;
      if (rowMask[left] !== rowMask[right]) return rowMask[right]! - rowMask[left]!;
      const leftNoise = jaxPaddingConsistentUniform(keys.sample, [left]);
      const rightNoise = jaxPaddingConsistentUniform(keys.sample, [right]);
      return rightNoise - leftNoise;
    });
    const centers = order.slice(0, Math.min(maxMsa, depth));
    const extras = order.slice(centers.length);
    const centerCodes = new Uint8Array(centers.length * length);
    for (let center = 0; center < centers.length; center += 1) {
      centerCodes.set(encoded.subarray(centers[center]! * length, (centers[center]! + 1) * length), center * length);
    }

    // JAX draws each element independently using nested fold_in keys, so skipped
    // unmasked positions do not alter any other random value.
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      if (rowMask[centers[center]!] === 0
        || jaxPaddingConsistentUniform(keys.maskPosition, [center, residue]) >= 0.15) continue;
      const original = centerCodes[center * length + residue]!;
      let bestCode = 0; let bestScore = Number.NEGATIVE_INFINITY;
      for (let code = 0; code < 23; code += 1) {
        const uniformProbability = code < 20 ? 0.005 : 0;
        const profileProbability = code < 22 ? 0.1 * msaProfile[residue * 22 + code]! : 0;
        const sameProbability = code === original ? 0.1 : 0;
        const maskProbability = code === 22 ? 0.7 : 0;
        const score = Math.log(uniformProbability + profileProbability + sameProbability + maskProbability + 1e-6)
          + gumbel(keys.maskGumbel, [center, residue, code]);
        if (score > bestScore) { bestScore = score; bestCode = code; }
      }
      centerCodes[center * length + residue] = bestCode;
    }

    const profile = new Float32Array(centers.length * length * 23);
    const deletionSums = new Float32Array(centers.length * length);
    const counts = new Float32Array(centers.length * length).fill(1);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue;
      if (rowMask[centers[center]!] !== 0) profile[slot * 23 + centerCodes[slot]!] = 1;
      deletionSums[slot] = deletionMatrix[centers[center]! * length + residue]!;
    }
    for (const extraRow of extras) {
      if (rowMask[extraRow] === 0) continue;
      let bestAgreement = -1; const nearest: number[] = [];
      for (let center = 0; center < centers.length; center += 1) {
        if (rowMask[centers[center]!] === 0) continue;
        let agreement = 0;
        for (let residue = 0; residue < length; residue += 1) {
          const code = centerCodes[center * length + residue]!;
          if (code <= 20 && code === encoded[extraRow * length + residue]!) agreement += 1;
        }
        if (agreement > bestAgreement) { bestAgreement = agreement; nearest.length = 0; nearest.push(center); }
        else if (agreement === bestAgreement) nearest.push(center);
      }
      if (nearest.length === 0) continue;
      const assignment = length / nearest.length;
      for (const center of nearest) for (let residue = 0; residue < length; residue += 1) {
        const slot = center * length + residue;
        counts[slot] = counts[slot]! + assignment;
        const profileSlot = slot * 23 + encoded[extraRow * length + residue]!;
        profile[profileSlot] = profile[profileSlot]! + assignment;
        deletionSums[slot] = deletionSums[slot]!
          + assignment * deletionMatrix[extraRow * length + residue]!;
      }
    }

    const msaFeatures = new Float32Array(centers.length * length * 49);
    const msaMask = new Float32Array(centers.length * length);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue; const output = slot * 49;
      msaMask[slot] = rowMask[centers[center]!]!;
      msaFeatures[output + centerCodes[slot]!] = rowMask[centers[center]!]!;
      const deletion = deletionMatrix[centers[center]! * length + residue]!;
      msaFeatures[output + 23] = Math.min(deletion, 1) * msaMask[slot]!;
      msaFeatures[output + 24] = deletionValue(deletion) * msaMask[slot]!;
      for (let code = 0; code < 23; code += 1) {
        msaFeatures[output + 25 + code] = profile[slot * 23 + code]! / counts[slot]!;
      }
      msaFeatures[output + 48] = deletionValue(deletionSums[slot]! / counts[slot]!);
    }

    const selectedExtras = extras.slice(0, maxExtra);
    const extraSequences = selectedExtras.length;
    const extraMsa = new Float32Array(extraSequences * length);
    const extraHasDeletion = new Float32Array(extraSequences * length);
    const extraDeletionValue = new Float32Array(extraSequences * length);
    const extraMsaMask = new Float32Array(extraSequences * length);
    for (let extra = 0; extra < extraSequences; extra += 1) for (let residue = 0; residue < length; residue += 1) {
      const row = selectedExtras[extra]!; const slot = extra * length + residue;
      const deletion = deletionMatrix[row * length + residue]!;
      extraMsa[slot] = encoded[row * length + residue]!;
      extraHasDeletion[slot] = Math.min(deletion, 1);
      extraDeletionValue[slot] = deletionValue(deletion);
      extraMsaMask[slot] = rowMask[row]!;
    }
    yield {
      targetFeatures: base.targetFeatures, msaFeatures, msaMask,
      extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask,
      residueIndex: base.residueIndex, aatype: base.aatype, seqMask: base.seqMask,
      atom37ToAtom14: base.atom37ToAtom14, atom37Mask: base.atom37Mask,
      msaSequences: centers.length, extraSequences, targetChannels: 22, msaFeatureChannels: 49,
    };
    }
  });
}

/** Lazily preprocess A3M text, retaining at most one recycle's large feature tensors. */
export function iterateA3mFeatures(a3mText: string, tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions = {}): RecycleFeatureSource<MonomerRecycleFeatures> {
  const alignment = parseA3m(a3mText);
  const length = alignment.length; const depth = alignment.depth;
  const alignmentMask = options.alignmentMask ?? new Float32Array(depth * length).fill(1);
  if (alignmentMask.length !== depth * length
    || alignmentMask.some((value) => value !== 0 && value !== 1)) {
    throw new RangeError("A3M alignment mask must have shape [depth, length] and contain only zero or one");
  }
  const encoded = new Uint8Array(depth * length);
  for (let row = 0; row < depth; row += 1) for (let residue = 0; residue < length; residue += 1) {
    const symbol = alignment.sequences[row]![residue]!;
    encoded[row * length + residue] = symbol === "-" ? 21 : (INDEX.get(symbol) ?? 20);
  }
  if (options.colabFoldMultimerProcess === true) {
    return makeColabFoldMultimerFeatures(alignment, encoded, tables, options);
  }
  const base = makeQueryOnlyFeatures(alignment.query, tables, { recycles: 0, maskedMsaCodes: [
    Float32Array.from(encoded.subarray(0, length)),
  ] })[0]!;
  const recycles = options.recycles ?? 3;
  if (!Number.isSafeInteger(recycles) || recycles < 0) {
    throw new RangeError("recycles must be a non-negative safe integer");
  }
  const maxMsa = Math.min(options.maxMsaSequences ?? 508, depth);
  const maxExtra = options.maxExtraSequences ?? 1024;
  return recycleFeatureSource(recycles + 1, function* features() {
    for (let recycle = 0; recycle <= recycles; recycle += 1) {
    const random = generator(((options.randomSeed ?? 0) ^ Math.imul(recycle + 1, 0x9e3779b9)) >>> 0);
    const remainder = Array.from({ length: depth - 1 }, (_, index) => index + 1); shuffle(remainder, random);
    const centers = [0, ...remainder.slice(0, Math.max(0, maxMsa - 1))];
    const extraPool = remainder.slice(Math.max(0, maxMsa - 1)); shuffle(extraPool, random);
    const extras = extraPool.slice(0, maxExtra);
    const centerCodes = new Uint8Array(centers.length * length);
    for (let center = 0; center < centers.length; center += 1) {
      centerCodes.set(encoded.subarray(centers[center]! * length, (centers[center]! + 1) * length), center * length);
    }
    for (let index = 0; index < centerCodes.length; index += 1) {
      const center = Math.floor(index / length); const residue = index % length;
      if (alignmentMask[centers[center]! * length + residue] === 0) continue;
      if (random() >= 0.15) continue;
      const original = centerCodes[index]!; const draw = random();
      if (draw < 0.7) centerCodes[index] = 22;
      else if (draw >= 0.9) centerCodes[index] = Math.floor(random() * 20);
      else centerCodes[index] = original;
    }
    const assignments = new Uint16Array(extras.length);
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) {
      const extraRow = extras[extraIndex]!; let best = 0; let bestScore = -1;
      for (let center = 0; center < centers.length; center += 1) {
        let score = 0;
        for (let residue = 0; residue < length; residue += 1) {
          const code = centerCodes[center * length + residue]!;
          if (code <= 20 && code === encoded[extraRow * length + residue]!) score += 1;
        }
        if (score > bestScore) { bestScore = score; best = center; }
      }
      assignments[extraIndex] = best;
    }
    const profile = new Float32Array(centers.length * length * 23);
    const deletionSums = new Float32Array(centers.length * length);
    const counts = new Float32Array(centers.length * length).fill(1 + 1e-6);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue;
      profile[slot * 23 + centerCodes[slot]!] = 1;
      deletionSums[slot] = alignment.deletionMatrix[centers[center]!]![residue]!;
    }
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) {
      const row = extras[extraIndex]!; const center = assignments[extraIndex]!;
      for (let residue = 0; residue < length; residue += 1) {
        const slot = center * length + residue;
        counts[slot] = counts[slot]! + 1;
        const profileSlot = slot * 23 + encoded[row * length + residue]!;
        profile[profileSlot] = profile[profileSlot]! + 1;
        deletionSums[slot] = deletionSums[slot]! + alignment.deletionMatrix[row]![residue]!;
      }
    }
    const msaFeatures = new Float32Array(centers.length * length * 49);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue; const output = slot * 49;
      msaFeatures[output + centerCodes[slot]!] = 1;
      const deletion = alignment.deletionMatrix[centers[center]!]![residue]!;
      msaFeatures[output + 23] = Math.min(deletion, 1); msaFeatures[output + 24] = deletionValue(deletion);
      for (let code = 0; code < 23; code += 1) msaFeatures[output + 25 + code] = profile[slot * 23 + code]! / counts[slot]!;
      msaFeatures[output + 48] = deletionValue(deletionSums[slot]! / counts[slot]!);
    }
    const extraSequences = Math.max(1, extras.length);
    const extraMsa = new Float32Array(extraSequences * length);
    const extraHasDeletion = new Float32Array(extraSequences * length);
    const extraDeletionValue = new Float32Array(extraSequences * length);
    const extraMsaMask = new Float32Array(extraSequences * length);
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = extraIndex * length + residue; const row = extras[extraIndex]!;
      const deletion = alignment.deletionMatrix[row]![residue]!;
      extraMsa[slot] = encoded[row * length + residue]!; extraHasDeletion[slot] = Math.min(deletion, 1);
      extraDeletionValue[slot] = deletionValue(deletion);
      extraMsaMask[slot] = 1;
    }
    yield {
      targetFeatures: base.targetFeatures, msaFeatures,
      msaMask: new Float32Array(centers.length * length).fill(1),
      extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask,
      residueIndex: base.residueIndex, aatype: base.aatype, seqMask: base.seqMask,
      atom37ToAtom14: base.atom37ToAtom14, atom37Mask: base.atom37Mask,
      msaSequences: centers.length, extraSequences, targetChannels: 22, msaFeatureChannels: 49,
    };
    }
  });
}

/** Eager compatibility wrapper. Prefer iterateA3mFeatures for browser inference. */
export function makeA3mFeatures(a3mText: string, tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions = {}): readonly MonomerRecycleFeatures[] {
  return [...iterateA3mFeatures(a3mText, tables, options)];
}
