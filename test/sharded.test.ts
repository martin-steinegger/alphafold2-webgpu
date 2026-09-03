import { describe, expect, it } from "vitest";
import { planShards, shardBindings, shardLoader, shardStorer, shardWordLoader } from "../src/runtime/sharded.js";

describe("binding shards", () => {
  it("keeps a tensor that fits in one binding", () => {
    const layout = planShards(1000, 128, 128 * 1024 ** 2, 2);
    expect(layout).toEqual({ count: 1, shardElements: 1000, totalElements: 1000 });
    expect(shardLoader(layout, "pair", "f16")).toContain("pair_0");
    expect(shardLoader(layout, "pair", "f16")).not.toContain("shard");
  });

  it("splits a pair too large for one binding on row boundaries", () => {
    // 1500 residues, 128 channels, packed: 288 million elements at two bytes.
    const elements = 1500 * 1500 * 128;
    const layout = planShards(elements, 128, 128 * 1024 ** 2, 2);
    expect(layout.count).toBe(5);
    expect(layout.shardElements % 128).toBe(0);
    expect(layout.shardElements * 2).toBeLessThanOrEqual(128 * 1024 ** 2);
    expect(layout.shardElements * layout.count).toBeGreaterThanOrEqual(elements);
  });

  it("declares one binding a shard and reads through the right one", () => {
    const layout = planShards(400, 64, 1024, 4);
    expect(layout).toEqual({ count: 2, shardElements: 256, totalElements: 400 });
    const declarations = shardBindings(layout, "pair", "f32", 3, false);
    expect(declarations).toContain("@binding(3) var<storage, read> pair_0");
    expect(declarations).toContain("@binding(4) var<storage, read> pair_1");
    const loader = shardLoader(layout, "pair", "f32");
    expect(loader).toContain("const PAIR_SHARD: u32 = 256u;");
    expect(loader).toContain("if (shard == 0u) { return pair_0[local]; }");
    expect(loader).toContain("return pair_1[local];");
  });

  it("stores by element when exact and by word when packed", () => {
    const layout = planShards(400, 64, 1024, 4);
    expect(shardStorer(layout, "pair", "f32")).toContain("fn pair_store(index: u32, value: f32)");
    const packed = shardStorer(layout, "pair", "f16");
    expect(packed).toContain("fn pair_store(word: u32, value: u32)");
    // A packed shard holds half as many words as elements.
    expect(packed).toContain("const PAIR_STORE_SHARD: u32 = 128u;");
    expect(shardWordLoader(layout, "pair")).toContain("fn pair_load_word(word: u32) -> u32");
  });

  it("starts every shard on a 256-byte boundary, which a binding offset needs", () => {
    // Channel counts and storages the model actually uses, at a limit small
    // enough to split them: a row alone is not enough to place a binding.
    for (const [channels, bytes] of [[128, 2], [128, 4], [256, 2], [64, 2], [32, 4]] as const) {
      const layout = planShards(channels * 100_000, channels, 1024 * 1024, bytes);
      expect(layout.count).toBeGreaterThan(1);
      expect((layout.shardElements * bytes) % 256).toBe(0);
      expect(layout.shardElements % channels).toBe(0);
      expect(layout.shardElements * bytes).toBeLessThanOrEqual(1024 * 1024);
    }
  });

  it("refuses a row that cannot fit a binding at all", () => {
    expect(() => planShards(1000, 512, 256, 4)).toThrow(/exceeds the binding limit/);
  });
});
