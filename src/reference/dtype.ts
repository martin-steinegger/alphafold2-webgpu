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
  const byteLength = tensorByteLength(record);
  if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + byteLength > buffer.byteLength) {
    throw new Error("tensor points outside its shard");
  }
  if (record.dtype === "float32") {
    if (byteOffset % 4 !== 0) throw new Error("float32 tensor has an unaligned byte offset");
    return copy
      ? new Float32Array(buffer.slice(byteOffset, byteOffset + elements * 4))
      : new Float32Array(buffer, byteOffset, elements);
  }
  if (record.dtype === "float16") {
    const stored = uint16View(buffer, byteOffset, elements);
    const output = new Float32Array(elements);
    for (let index = 0; index < elements; index += 1) output[index] = float16ToNumber(stored[index]!);
    return output;
  }
  const codes = new Int8Array(buffer, byteOffset, elements);
  const scaleOffset = byteOffset + record.scaleOffset! - (record.byteOffset ?? 0);
  const scales = uint16View(buffer, scaleOffset, Math.ceil(elements / record.block!));
  const output = new Float32Array(elements);
  for (let index = 0; index < elements; index += 1) {
    output[index] = codes[index]! * float16ToNumber(scales[Math.floor(index / record.block!)]!);
  }
  return output;
}
