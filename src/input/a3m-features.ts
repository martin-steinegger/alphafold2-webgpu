import { parseA3m } from "./a3m.js";
import { makeQueryOnlyFeatures, type QueryOnlyFeatureTables } from "./query-only-features.js";
import type { MonomerRecycleFeatures } from "../model/monomer.js";

const RESTYPES = "ARNDCQEGHILKMFPSTWYV";
const INDEX = new Map([...RESTYPES].map((residue, index) => [residue, index]));

export interface A3mFeatureOptions {
  readonly recycles?: number; readonly randomSeed?: number;
  readonly maxMsaSequences?: number; readonly maxExtraSequences?: number;
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

/** CPU feature preprocessing for A3M text. Neural inference remains entirely on WebGPU. */
export function makeA3mFeatures(a3mText: string, tables: QueryOnlyFeatureTables,
  options: A3mFeatureOptions = {}): readonly MonomerRecycleFeatures[] {
  const alignment = parseA3m(a3mText);
  const length = alignment.length; const depth = alignment.depth;
  const encoded = new Uint8Array(depth * length);
  for (let row = 0; row < depth; row += 1) for (let residue = 0; residue < length; residue += 1) {
    const symbol = alignment.sequences[row]![residue]!;
    encoded[row * length + residue] = symbol === "-" ? 21 : (INDEX.get(symbol) ?? 20);
  }
  const base = makeQueryOnlyFeatures(alignment.query, tables, { recycles: 0, maskedMsaCodes: [
    Float32Array.from(encoded.subarray(0, length)),
  ] })[0]!;
  const recycles = options.recycles ?? 3;
  const maxMsa = Math.min(options.maxMsaSequences ?? 508, depth);
  const maxExtra = options.maxExtraSequences ?? 1024;
  const results: MonomerRecycleFeatures[] = [];
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
      profile[(center * length + residue) * 23 + centerCodes[center * length + residue]!] = 1;
      deletionSums[center * length + residue] = alignment.deletionMatrix[centers[center]!]![residue]!;
    }
    for (let extraIndex = 0; extraIndex < extras.length; extraIndex += 1) {
      const row = extras[extraIndex]!; const center = assignments[extraIndex]!;
      for (let residue = 0; residue < length; residue += 1) {
        const slot = center * length + residue; counts[slot] = counts[slot]! + 1;
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
      extraDeletionValue[slot] = deletionValue(deletion); extraMsaMask[slot] = 1;
    }
    results.push({
      targetFeatures: base.targetFeatures.slice(), msaFeatures, msaMask: new Float32Array(centers.length * length).fill(1),
      extraMsa, extraHasDeletion, extraDeletionValue, extraMsaMask,
      residueIndex: base.residueIndex.slice(), aatype: base.aatype.slice(), seqMask: base.seqMask.slice(),
      atom37ToAtom14: base.atom37ToAtom14.slice(), atom37Mask: base.atom37Mask.slice(),
      msaSequences: centers.length, extraSequences, targetChannels: 22, msaFeatureChannels: 49,
    });
  }
  return results;
}
