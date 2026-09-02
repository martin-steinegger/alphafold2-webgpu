import { describe, expect, it } from "vitest";
import { packHalfWords, storageWords, unpackHalfWords } from "../src/runtime/storage.js";

describe("packed half-precision storage", () => {
  it("round-trips values within half precision, including odd counts and special values", () => {
    const values = new Float32Array([0, -0, 1, -1, 0.1, -3.14159, 65504, 1e-5, 2 ** -24, 1234.5, -0.000061, 42]);
    const words = packHalfWords(values);
    expect(words.length).toBe(storageWords(values.length, "f16"));
    const back = unpackHalfWords(words, values.length);
    for (let index = 0; index < values.length; index += 1) {
      const value = values[index]!;
      const tolerance = Math.max(Math.abs(value) * 2 ** -11, 2 ** -25);
      expect(Math.abs(back[index]! - value), `value ${value}`).toBeLessThanOrEqual(tolerance);
    }
    expect(Object.is(back[1], -0)).toBe(true);
    expect(unpackHalfWords(packHalfWords(new Float32Array([1e6])), 1)[0]).toBe(Number.POSITIVE_INFINITY);
  });

  it("packs exact halves bit for bit", () => {
    // 1.0 is 0x3c00, -2.0 is 0xc000; the first value takes the low half-word.
    expect(packHalfWords(new Float32Array([1, -2]))[0]).toBe((0xc000 << 16 | 0x3c00) >>> 0);
    expect(storageWords(7, "f32")).toBe(7);
    expect(storageWords(7, "f16")).toBe(4);
  });
});
