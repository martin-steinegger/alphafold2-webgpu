import { describe, expect, it } from "vitest";
import { readTensor, readTensorRange } from "../src/reference/dtype.js";
import { blockRange } from "../src/reference/tensor-store.js";

function float16Bits(value: number): number {
  // Enough for the small positive scales used here: normal range only.
  const exponent = Math.floor(Math.log2(value));
  const mantissa = Math.round((value / 2 ** exponent - 1) * 1024);
  return ((exponent + 15) << 10) | mantissa;
}

describe("tensor range decoding", () => {
  it("decodes int8 block-scaled ranges exactly as the whole tensor", () => {
    const elements = 200; const block = 64;
    const codes = Int8Array.from({ length: elements }, (_, index) => ((index * 37) % 255) - 127);
    const scales = Uint16Array.from({ length: Math.ceil(elements / block) }, (_, index) => float16Bits(0.5 + index * 0.25));
    const buffer = new ArrayBuffer(8 + elements + scales.byteLength);
    new Int8Array(buffer, 8, elements).set(codes);
    new Uint16Array(buffer, 8 + elements, scales.length).set(scales);
    const record = { file: "shard", shape: [4, 50], dtype: "int8" as const, byteOffset: 8, block, scaleOffset: 8 + elements };
    const whole = readTensor(record, buffer);
    for (const [start, count] of [[0, 200], [0, 64], [60, 10], [63, 2], [127, 73], [199, 1]] as const) {
      expect(Array.from(readTensorRange(record, buffer, 8, start, count))).toEqual(Array.from(whole.subarray(start, start + count)));
    }
    expect(() => readTensorRange(record, buffer, 8, 190, 11)).toThrow(RangeError);
  });

  it("returns views of float32 shards and slices stacked blocks", () => {
    const values = Float32Array.from({ length: 12 }, (_, index) => index * 1.5);
    const record = { file: "shard", shape: [3, 4], dtype: "float32" as const, byteOffset: 0 };
    const view = readTensorRange(record, values.buffer, 0, 4, 4);
    expect(view.buffer).toBe(values.buffer);
    expect(Array.from(view)).toEqual([6, 7.5, 9, 10.5]);
    expect(blockRange(12, 1, 3)).toEqual([4, 4]);
    expect(blockRange(12)).toEqual([0, 12]);
    expect(() => blockRange(12, 3, 3)).toThrow(RangeError);
    expect(() => blockRange(10, 0, 3)).toThrow(RangeError);
  });
});
