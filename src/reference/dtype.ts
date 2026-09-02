import type { BinaryTensorRecord } from "./tensor-store.js";

const WIDTHS = { float32: 4, float16: 2, int8: 1 } as const;

export function tensorElements(record: BinaryTensorRecord): number {
  const elements = record.shape.reduce((product, value) => product * value, 1);
  if (!Number.isSafeInteger(elements) || elements <= 0) throw new Error("tensor has an invalid element count");
  return elements;
}

/** Stored bytes occupied from byteOffset through the tensor's final scale. */
export function tensorByteLength(record: BinaryTensorRecord): number {
  const elements = tensorElements(record);
  if (record.dtype !== "int8") return elements * WIDTHS[record.dtype];
  const byteOffset = record.byteOffset ?? 0;
  const block = record.block;
  const scaleOffset = record.scaleOffset;
  if (block === undefined || !Number.isSafeInteger(block) || block <= 0) {
    throw new Error("int8 tensor has an invalid block size");
  }
  if (scaleOffset === undefined || !Number.isSafeInteger(scaleOffset) || scaleOffset < byteOffset + elements
    || scaleOffset % 2 !== 0) {
    throw new Error("int8 tensor has an invalid scale offset");
  }
  return scaleOffset - byteOffset + Math.ceil(elements / block) * 2;
}

export function float16ToNumber(bits: number): number {
  const sign = (bits & 0x8000) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const fraction = bits & 0x03ff;
  if (exponent === 0) return fraction === 0 ? sign * 0 : sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 0x1f) return fraction === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function uint16View(buffer: ArrayBufferLike, byteOffset: number, length: number): Uint16Array {
  if (byteOffset % 2 === 0) return new Uint16Array(buffer, byteOffset, length);
  return new Uint16Array(buffer.slice(byteOffset, byteOffset + length * 2));
}

/** Decode any supported storage representation into the float32 values consumed by WGSL. */
export function readTensor(record: BinaryTensorRecord, buffer: ArrayBufferLike,
  byteOffset = record.byteOffset ?? 0, copy = false): Float32Array {
  const elements = tensorElements(record);
  if (record.dtype === "float32" && copy) {
    const byteLength = tensorByteLength(record);
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
      throw new Error("tensor points outside its shard");
    }
    if (byteOffset % 4 !== 0) throw new Error("float32 tensor has an unaligned byte offset");
    return new Float32Array(buffer.slice(byteOffset, byteOffset + elements * 4));
  }
  return readTensorRange(record, buffer, byteOffset, 0, elements);
}

/**
 * Decode `count` values starting at value `start` of a stored tensor.
 *
 * Float32 comes back as a view of the shard; compressed formats decode into a
 * fresh array. Block scales are indexed by absolute position, so a range that
 * starts inside a block decodes exactly as the whole tensor would. Decoding on
 * demand is what lets a model stay stored in its compressed form: the float32
 * values of a block exist only while that block is being packed for the GPU.
 */
export function readTensorRange(record: BinaryTensorRecord, buffer: ArrayBufferLike,
  byteOffset: number, start: number, count: number): Float32Array {
  const elements = tensorElements(record);
  const byteLength = tensorByteLength(record);
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
    throw new Error("tensor points outside its shard");
  }
  if (![start, count].every(Number.isSafeInteger) || start < 0 || count < 0 || start + count > elements) {
    throw new RangeError("tensor range is out of bounds");
  }
  if (record.dtype === "float32") {
    if (byteOffset % 4 !== 0) throw new Error("float32 tensor has an unaligned byte offset");
    return new Float32Array(buffer, byteOffset + start * 4, count);
  }
  if (record.dtype === "float16") {
    const stored = uint16View(buffer, byteOffset + start * 2, count);
    const output = new Float32Array(count);
    for (let index = 0; index < count; index += 1) output[index] = float16ToNumber(stored[index]!);
    return output;
  }
  const block = record.block!;
  const codes = new Int8Array(buffer, byteOffset + start, count);
  const scaleOffset = byteOffset + record.scaleOffset! - (record.byteOffset ?? 0);
  const scales = uint16View(buffer, scaleOffset, Math.ceil(elements / block));
  const output = new Float32Array(count);
  let index = 0;
  while (index < count) {
    const absolute = start + index;
    const scale = float16ToNumber(scales[Math.floor(absolute / block)]!);
    const blockEnd = Math.min(count, index + block - (absolute % block));
    for (; index < blockEnd; index += 1) output[index] = codes[index]! * scale;
  }
  return output;
}
