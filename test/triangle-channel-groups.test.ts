import { describe, expect, it } from "vitest";
import { wholeChannelGroups } from "../src/evoformer/block.js";
import { WHOLE_CHANNEL_ALIGNMENT, wholeProjectionStride } from "../src/triangle/shaders.js";

/** WebGPU's guaranteed storage binding offset alignment. */
const OFFSET_BYTES = 256;
const HIDDEN = 128;

describe("the channels the contraction covers in one dispatch", () => {
  it("pads a channel so every one of them starts where a binding may start", () => {
    for (const length of [1, 59, 118, 177, 256, 825, 1650, 3300, 6688]) {
      const stride = wholeProjectionStride(length);
      expect(stride).toBeGreaterThanOrEqual(length * length);
      expect(stride % WHOLE_CHANNEL_ALIGNMENT).toBe(0);
      // Both storages: 128 halves are 256 bytes, 128 words are 512.
      for (const bytes of [2, 4]) expect(stride * bytes % OFFSET_BYTES).toBe(0);
    }
  });

  it("pads by less than a channel, so the tensor does not grow", () => {
    for (const length of [59, 825, 3300]) {
      const grown = wholeProjectionStride(length) - length * length;
      expect(grown).toBeLessThan(WHOLE_CHANNEL_ALIGNMENT);
      // Over all the channels that is kilobytes, whatever the length.
      expect(grown * HIDDEN * 4).toBeLessThan(64 * 1024);
    }
  });

  it("covers every channel exactly once, in order", () => {
    for (const bindingBytes of [4 << 20, 16 << 20, 512 << 20, 2_147_483_644]) {
      const groups = wholeChannelGroups(HIDDEN, wholeProjectionStride(825), bindingBytes, 2);
      expect(groups[0]?.first).toBe(0);
      let next = 0;
      for (const group of groups) {
        expect(group.first).toBe(next);
        expect(group.count).toBeGreaterThan(0);
        next += group.count;
      }
      expect(next).toBe(HIDDEN);
    }
  });

  it("keeps every group inside one binding", () => {
    const stride = wholeProjectionStride(3300);
    for (const bindingBytes of [64 << 20, 512 << 20, 2_147_483_644]) {
      for (const group of wholeChannelGroups(HIDDEN, stride, bindingBytes, 2)) {
        expect(group.count * stride * 2).toBeLessThanOrEqual(bindingBytes);
      }
    }
  });

  it("takes one group where the whole projection fits, which is the common case", () => {
    // 1,650 residues packed is 665 MiB, and a native adapter binds 2 GiB.
    expect(wholeChannelGroups(HIDDEN, wholeProjectionStride(1650), 2_147_483_644, 2))
      .toEqual([{ first: 0, count: HIDDEN }]);
    // A 3,300-residue tetramer is 2.6 GiB, so it takes two.
    expect(wholeChannelGroups(HIDDEN, wholeProjectionStride(3300), 2_147_483_644, 2).length).toBe(2);
  });

  it("says which sequence is too long rather than binding past the limit", () => {
    // One channel of a 16,384-residue pair is 512 MiB packed, past a browser
    // default binding of 128 MiB, and no grouping can help that.
    expect(() => wholeChannelGroups(HIDDEN, wholeProjectionStride(16_384), 128 << 20, 2))
      .toThrow(/past the .* bytes one binding may cover/);
  });
});
