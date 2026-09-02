/**
 * Storage of an activation tensor.
 *
 * `f32` is the exact representation every reference was produced with. `f16`
 * packs two half-precision values into each 32-bit word with `pack2x16float`,
 * which needs no device feature and halves the tensor, at the cost of rounding
 * every value written to about three significant digits. Kernels read packed
 * elements one at a time and write them a whole word at a time, so the pair
 * of channels sharing a word is always owned by one invocation.
 */
export type ActivationStorage = "f32" | "f16";

/** WGSL element type of an array holding values stored as `storage`. */
export function storageArray(storage: ActivationStorage): "f32" | "u32" {
  return storage === "f16" ? "u32" : "f32";
}

/** WGSL expression reading value number `index` of `array`, whatever its storage. */
export function storedElement(storage: ActivationStorage, array: string, index: string): string {
  return storage === "f16" ? `unpack2x16float(${array}[(${index}) >> 1u])[(${index}) & 1u]` : `${array}[${index}]`;
}

/** 32-bit words backing `elements` values stored as `storage`. */
export function storageWords(elements: number, storage: ActivationStorage): number {
  return storage === "f16" ? Math.ceil(elements / 2) : elements;
}

function halfToFloat(half: number): number {
  const sign = (half & 0x8000) !== 0 ? -1 : 1;
  const exponent = (half >> 10) & 0x1f;
  const mantissa = half & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function floatToHalf(value: number): number {
  if (Number.isNaN(value)) return 0x7e00;
  const sign = value < 0 || (value === 0 && 1 / value < 0) ? 0x8000 : 0;
  const magnitude = Math.abs(value);
  if (magnitude === Number.POSITIVE_INFINITY) return sign | 0x7c00;
  if (magnitude < 2 ** -24 / 2) return sign;
  if (magnitude < 2 ** -14) {
    // Subnormal: round to the nearest multiple of 2^-24, ties to even.
    const scaled = magnitude * 2 ** 24;
    let mantissa = Math.floor(scaled);
    const remainder = scaled - mantissa;
    if (remainder > 0.5 || (remainder === 0.5 && (mantissa & 1) === 1)) mantissa += 1;
    return sign | mantissa;
  }
  let exponent = Math.floor(Math.log2(magnitude));
  let fraction = magnitude / 2 ** exponent - 1;
  if (fraction >= 1) { exponent += 1; fraction = magnitude / 2 ** exponent - 1; }
  const scaled = fraction * 1024;
  let mantissa = Math.floor(scaled);
  const remainder = scaled - mantissa;
  if (remainder > 0.5 || (remainder === 0.5 && (mantissa & 1) === 1)) mantissa += 1;
  if (mantissa === 1024) { mantissa = 0; exponent += 1; }
  if (exponent > 15) return sign | 0x7c00;
  return sign | ((exponent + 15) << 10) | mantissa;
}

/** Unpack `count` half-precision values from packed words. */
export function unpackHalfWords(words: Uint32Array, count: number): Float32Array {
  const output = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const word = words[index >> 1]!;
    output[index] = halfToFloat((index & 1) === 0 ? word & 0xffff : word >>> 16);
  }
  return output;
}

/** Pack values into half-precision pairs, rounding to nearest even. */
export function packHalfWords(values: Float32Array): Uint32Array {
  const words = new Uint32Array(Math.ceil(values.length / 2));
  for (let index = 0; index < values.length; index += 1) {
    const half = floatToHalf(values[index]!);
    words[index >> 1] = (words[index >> 1]! | (half << ((index & 1) * 16))) >>> 0;
  }
  return words;
}
