import { describe, expect, it } from "vitest";
import { float16ToNumber, readTensor, tensorByteLength } from "../src/reference/dtype.js";
import type { BinaryTensorRecord } from "../src/reference/tensor-store.js";

describe("mixed model tensor storage", () => {
  it("decodes IEEE float16 values without requiring Float16Array", () => {
    expect(float16ToNumber(0x3c00)).toBe(1);
    expect(float16ToNumber(0xc000)).toBe(-2);
    expect(float16ToNumber(0x0001)).toBe(2 ** -24);
    expect(float16ToNumber(0x7c00)).toBe(Number.POSITIVE_INFINITY);
  });

  it("decodes symmetric int8 blocks using stored float16 scales", () => {
    const buffer = new ArrayBuffer(12);
    new Int8Array(buffer, 0, 5).set([1, -2, 3, -4, 5]);
    new Uint16Array(buffer, 6, 2).set([0x3800, 0x4000]); // 0.5, 2.0
    const record: BinaryTensorRecord = {
      file: "weights.bin", shape: [5], dtype: "int8", byteOffset: 0,
      block: 4, scaleOffset: 6,
    };
    expect(tensorByteLength(record)).toBe(10);
    expect([...readTensor(record, buffer)]).toEqual([0.5, -1, 1.5, -2, 10]);
  });

  it("rejects int8 metadata that overlaps codes and scales", () => {
    const record: BinaryTensorRecord = {
      file: "weights.bin", shape: [8], dtype: "int8", byteOffset: 4,
      block: 4, scaleOffset: 10,
    };
    expect(() => tensorByteLength(record)).toThrow(/scale offset/);
  });
});
