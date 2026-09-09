import {
  assignNearestCentres, nearestCentreSets, tieSetWords,
} from "./msa-clustering-webgpu.js";
import { CLUSTERED_MSA_CHANNELS, MSA_CODE_NONE } from "./msa-features.js";
import { endPhase, markPhase, timedSync } from "../runtime/phase-ledger.js";
import { parseA3m, type A3mAlignment } from "./a3m.js";
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

/**
 * A replayable, counted feature stream that materializes one recycle at a time.
 *
 * Asynchronous because each recycle's clustering runs on the device.
 */
export interface RecycleFeatureSource<T> extends AsyncIterable<T> { readonly length: number; }

/** Materialised recycles as a source, for a caller that already has them. */
export function recycleFeatureSourceOf<T>(items: readonly T[]): RecycleFeatureSource<T> {
  return recycleFeatureSource(items.length, async function* held() { yield* items; });
}

export function recycleFeatureSource<T>(
  length: number,
  iterator: () => AsyncIterator<T>,
): RecycleFeatureSource<T> {
  return { length, [Symbol.asyncIterator]: iterator };
}

function bitCount(word: number): number {
  let bits = word - ((word >>> 1) & 0x55555555);
  bits = (bits & 0x33333333) + ((bits >>> 2) & 0x33333333);
  return Math.imul((bits + (bits >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
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
  device: GPUDevice,
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
  return recycleFeatureSource(recycles + 1, async function* features() {
    let rootKey: JaxKey = [0, (options.randomSeed ?? 0) >>> 0];
    for (let recycle = 0; recycle <= recycles; recycle += 1) {
    const keys = multimerMsaKeys(rootKey); rootKey = keys.nextRoot;
    markPhase("featurise: sample");
    // The noise is a per-row constant, so drawing it once a row rather than
    // once a comparison is the same order for a twentieth of the draws.
    const noise = Float64Array.from({ length: depth },
      (_, row) => jaxPaddingConsistentUniform(keys.sample, [row]));
    const order = Array.from({ length: depth }, (_, row) => row);
    order.sort((left, right) => {
      const leftBias = left === 0 ? 1 : 0; const rightBias = right === 0 ? 1 : 0;
      if (leftBias !== rightBias) return rightBias - leftBias;
      if (rowMask[left] !== rowMask[right]) return rowMask[right]! - rowMask[left]!;
      return noise[right]! - noise[left]!;
    });
    const centers = order.slice(0, Math.min(maxMsa, depth));
    const extras = order.slice(centers.length);
    const centerCodes = new Uint8Array(centers.length * length);
    for (let center = 0; center < centers.length; center += 1) {
      centerCodes.set(encoded.subarray(centers[center]! * length, (centers[center]! + 1) * length), center * length);
    }

    markPhase("featurise: mask");
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

    markPhase("featurise: centre rows");
    // The profile accumulates straight into the feature array's profile
    // channels, so the run never holds a second copy of it.
    const msaFeatures = new Float32Array(centers.length * length * CLUSTERED_MSA_CHANNELS);
    const deletionSums = new Float32Array(centers.length * length);
    const counts = new Float32Array(centers.length * length).fill(1);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue;
      if (rowMask[centers[center]!] !== 0) {
        msaFeatures[slot * CLUSTERED_MSA_CHANNELS + 3 + centerCodes[slot]!] = 1;
      }
      deletionSums[slot] = deletionMatrix[centers[center]! * length + residue]!;
    }
    markPhase("featurise: cluster");
    // A masked row takes no part in the search, so the padding a block adds is
    // never gathered and never costs the kernel anything.
    const active = extras.filter((row) => rowMask[row] !== 0);
    const extraCodes = new Uint8Array(active.length * length);
    for (let index = 0; index < active.length; index += 1) {
      extraCodes.set(
        encoded.subarray(active[index]! * length, (active[index]! + 1) * length), index * length);
    }
    const words = tieSetWords(centers.length);
    const sets = await nearestCentreSets(
      device, centerCodes, centers.length, extraCodes, active.length, length,
      Uint8Array.from(centers, (row) => rowMask[row]!));
    markPhase("featurise: profile");
    for (let index = 0; index < active.length; index += 1) {
      const extraRow = active[index]!;
      // Multimer keeps every centre tied at the best agreement and splits the
      // row's weight between them, so the whole set is read, not one winner.
      let tied = 0;
      for (let word = 0; word < words; word += 1) {
        tied += bitCount(sets[index * words + word]!);
      }
      if (tied === 0) continue;
      const assignment = length / tied;
      for (let word = 0; word < words; word += 1) {
        let bits = sets[index * words + word]!;
        while (bits !== 0) {
          const center = word * 32 + 31 - Math.clz32(bits & -bits);
          bits &= bits - 1;
          for (let residue = 0; residue < length; residue += 1) {
            const slot = center * length + residue;
            counts[slot] = counts[slot]! + assignment;
            const profileSlot = slot * CLUSTERED_MSA_CHANNELS + 3 + encoded[extraRow * length + residue]!;
            msaFeatures[profileSlot] = msaFeatures[profileSlot]! + assignment;
            deletionSums[slot] = deletionSums[slot]!
              + assignment * deletionMatrix[extraRow * length + residue]!;
          }
        }
      }
    }

    markPhase("featurise: normalise");
    const msaMask = new Float32Array(centers.length * length);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue; const output = slot * CLUSTERED_MSA_CHANNELS;
      msaMask[slot] = rowMask[centers[center]!]!;
      // A masked row contributes no one-hot at all, which the code says.
      msaFeatures[output] = msaMask[slot] === 0 ? MSA_CODE_NONE : centerCodes[slot]!;
      const deletion = deletionMatrix[centers[center]! * length + residue]!;
      msaFeatures[output + 1] = Math.min(deletion, 1) * msaMask[slot]!;
      msaFeatures[output + 2] = deletionValue(deletion) * msaMask[slot]!;
      for (let code = 0; code < 23; code += 1) {
        msaFeatures[output + 3 + code] = msaFeatures[output + 3 + code]! / counts[slot]!;
      }
      msaFeatures[output + 26] = deletionValue(deletionSums[slot]! / counts[slot]!);
    }

    markPhase("featurise: extra rows");
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
    endPhase();
    yield {
      targetFeatures: base.targetFeatures, msaFeatures, msaMask,
      extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask,
      residueIndex: base.residueIndex, aatype: base.aatype, seqMask: base.seqMask,
      atom37ToAtom14: base.atom37ToAtom14, atom37Mask: base.atom37Mask,
      msaSequences: centers.length, extraSequences, targetChannels: 22,
      msaFeatureChannels: CLUSTERED_MSA_CHANNELS,
    };
    }
  });
}

/** Lazily preprocess A3M text, retaining at most one recycle's large feature tensors. */
export function iterateA3mFeatures(
  device: GPUDevice, a3mText: string | A3mAlignment, tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions = {},
): RecycleFeatureSource<MonomerRecycleFeatures> {
  // Parsing and encoding the alignment happens once, before any recycle, so it
  // is not part of the per-recycle featurisation and gets its own row.
  return timedSync("featurise: setup",
    () => buildA3mFeatureSource(device, a3mText, tables, options));
}

function buildA3mFeatureSource(
  device: GPUDevice, a3mText: string | A3mAlignment, tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions,
): RecycleFeatureSource<MonomerRecycleFeatures> {
  // A caller that already parsed the alignment to size the device passes it
  // back rather than paying for a second parse, which on an 8.77 MB alignment
  // is 0.72 s of a 17.6 s fold.
  const alignment = typeof a3mText === "string" ? parseA3m(a3mText) : a3mText;
  const length = alignment.length; const depth = alignment.depth;
  // No mask means every position is maskable. Materialising that as ones would
  // cost depth by length floats, 31 MiB for an 8,000-row alignment of 1,000
  // residues, to say what its absence already says.
  const alignmentMask = options.alignmentMask;
  if (alignmentMask !== undefined && (alignmentMask.length !== depth * length
    || alignmentMask.some((value) => value !== 0 && value !== 1))) {
    throw new RangeError("A3M alignment mask must have shape [depth, length] and contain only zero or one");
  }
  const encoded = new Uint8Array(depth * length);
  for (let row = 0; row < depth; row += 1) for (let residue = 0; residue < length; residue += 1) {
    const symbol = alignment.sequences[row]![residue]!;
    encoded[row * length + residue] = symbol === "-" ? 21 : (INDEX.get(symbol) ?? 20);
  }
  if (options.colabFoldMultimerProcess === true) {
    return makeColabFoldMultimerFeatures(device, alignment, encoded, tables, options);
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
  return recycleFeatureSource(recycles + 1, async function* features() {
    for (let recycle = 0; recycle <= recycles; recycle += 1) {
    markPhase("featurise: sample");
    const random = generator(((options.randomSeed ?? 0) ^ Math.imul(recycle + 1, 0x9e3779b9)) >>> 0);
    const remainder = Array.from({ length: depth - 1 }, (_, index) => index + 1); shuffle(remainder, random);
    const centers = [0, ...remainder.slice(0, Math.max(0, maxMsa - 1))];
    const extraPool = remainder.slice(Math.max(0, maxMsa - 1)); shuffle(extraPool, random);
    const extras = extraPool.slice(0, maxExtra);
    const centerCodes = new Uint8Array(centers.length * length);
    for (let center = 0; center < centers.length; center += 1) {
      centerCodes.set(encoded.subarray(centers[center]! * length, (centers[center]! + 1) * length), center * length);
    }
    markPhase("featurise: mask");
    for (let index = 0; index < centerCodes.length; index += 1) {
      const center = Math.floor(index / length); const residue = index % length;
      if (alignmentMask?.[centers[center]! * length + residue] === 0) continue;
      if (random() >= 0.15) continue;
      const original = centerCodes[index]!; const draw = random();
      if (draw < 0.7) centerCodes[index] = 22;
      else if (draw >= 0.9) centerCodes[index] = Math.floor(random() * 20);
      else centerCodes[index] = original;
    }
    markPhase("featurise: cluster");
    // Gathered so the kernel reads the drawn rows densely.
    const extraCodes = new Uint8Array(extras.length * length);
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) {
      extraCodes.set(
        encoded.subarray(extras[extraIndex]! * length, (extras[extraIndex]! + 1) * length),
        extraIndex * length);
    }
    const assignments = await assignNearestCentres(
      device, centerCodes, centers.length, extraCodes, extras.length, length);
    markPhase("featurise: profile");
    const msaFeatures = new Float32Array(centers.length * length * CLUSTERED_MSA_CHANNELS);
    const deletionSums = new Float32Array(centers.length * length);
    const counts = new Float32Array(centers.length * length).fill(1 + 1e-6);
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue;
      msaFeatures[slot * CLUSTERED_MSA_CHANNELS + 3 + centerCodes[slot]!] = 1;
      deletionSums[slot] = alignment.deletionMatrix[centers[center]!]![residue]!;
    }
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) {
      const row = extras[extraIndex]!; const center = assignments[extraIndex]!;
      for (let residue = 0; residue < length; residue += 1) {
        const slot = center * length + residue;
        counts[slot] = counts[slot]! + 1;
        const profileSlot = slot * CLUSTERED_MSA_CHANNELS + 3 + encoded[row * length + residue]!;
        msaFeatures[profileSlot] = msaFeatures[profileSlot]! + 1;
        deletionSums[slot] = deletionSums[slot]! + alignment.deletionMatrix[row]![residue]!;
      }
    }
    for (let center = 0; center < centers.length; center += 1) for (let residue = 0; residue < length; residue += 1) {
      const slot = center * length + residue; const output = slot * CLUSTERED_MSA_CHANNELS;
      msaFeatures[output] = centerCodes[slot]!;
      const deletion = alignment.deletionMatrix[centers[center]!]![residue]!;
      msaFeatures[output + 1] = Math.min(deletion, 1); msaFeatures[output + 2] = deletionValue(deletion);
      for (let code = 0; code < 23; code += 1) {
        msaFeatures[output + 3 + code] = msaFeatures[output + 3 + code]! / counts[slot]!;
      }
      msaFeatures[output + 26] = deletionValue(deletionSums[slot]! / counts[slot]!);
    }
    markPhase("featurise: extra rows");
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
    endPhase();
    yield {
      targetFeatures: base.targetFeatures, msaFeatures,
      msaMask: new Float32Array(centers.length * length).fill(1),
      extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask,
      residueIndex: base.residueIndex, aatype: base.aatype, seqMask: base.seqMask,
      atom37ToAtom14: base.atom37ToAtom14, atom37Mask: base.atom37Mask,
      msaSequences: centers.length, extraSequences, targetChannels: 22,
      msaFeatureChannels: CLUSTERED_MSA_CHANNELS,
    };
    }
  });
}

/** Eager compatibility wrapper. Prefer iterateA3mFeatures for browser inference. */
export async function makeA3mFeatures(
  device: GPUDevice, a3mText: string, tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions = {},
): Promise<readonly MonomerRecycleFeatures[]> {
  const all: MonomerRecycleFeatures[] = [];
  for await (const features of iterateA3mFeatures(device, a3mText, tables, options)) {
    all.push(features);
  }
  return all;
}
