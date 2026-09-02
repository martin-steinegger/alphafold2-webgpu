import { CLUSTERED_MSA_CHANNELS, expandClusteredMsaFeatures } from "../src/input/msa-features.js";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { iterateA3mFeatures, makeA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";

describe("A3M model feature preprocessing", () => {
  it("streams deterministic recycles and shares immutable sequence tensors", async () => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open(
      "test/fixtures/evoformer/model1-query-59-stack/manifest.json",
    ));
    const source = iterateA3mFeatures(">query\nACGG\n>homolog\nA-GG\n",
      await fixture.queryOnlyFeatureTables(), { recycles: 2, randomSeed: 7 });
    expect(source.length).toBe(3);
    const iterator = source[Symbol.iterator]();
    const first = iterator.next().value!;
    const second = iterator.next().value!;
    expect(first.aatype).toBe(second.aatype);
    expect(first.targetFeatures).toBe(second.targetFeatures);
    expect([...source].map((features) => [...features.msaFeatures]))
      .toEqual([...source].map((features) => [...features.msaFeatures]));
  });

  it("keeps block padding as gaps while excluding it from masked-MSA augmentation", async () => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open("test/fixtures/evoformer/model1-query-59-stack/manifest.json"));
    const result = makeA3mFeatures(">query\nACGG\n>chain\nA---\n", await fixture.queryOnlyFeatureTables(), {
      recycles: 0, randomSeed: 0, maxMsaSequences: 2,
      alignmentMask: Float32Array.of(1, 1, 1, 1, 1, 1, 0, 0),
    });
    const features = result[0]!;
    expect([...features.msaMask]).toEqual(new Array(8).fill(1));
    // Rows six and seven are block padding: gaps, and never masked-augmented.
    const dense = expandClusteredMsaFeatures(features.msaFeatures, features.msaSequences * 4);
    expect(dense[6 * 49 + 21]).toBe(1);
    expect(dense[7 * 49 + 21]).toBe(1);
  });

  it("clusters the uploaded 8,076-row alignment into model-1 tensors", async () => {
    const fixture = AlphaFoldFixture.fromStore(await FileTensorStore.open("test/fixtures/evoformer/model1-query-59-stack/manifest.json"));
    const result = makeA3mFeatures(await readFile("test.a3m", "utf8"), await fixture.queryOnlyFeatureTables(), {
      recycles: 0, randomSeed: 0,
    });
    const features = result[0]!;
    expect(features.msaSequences).toBe(508);
    expect(features.extraSequences).toBe(1024);
    expect(features.msaFeatures.length).toBe(508 * 59 * CLUSTERED_MSA_CHANNELS);
    expect(features.extraMsa.length).toBe(1024 * 59);
    expect(features.msaMask.every((value) => value === 1)).toBe(true);
    expect(features.extraMsaMask.every((value) => value === 1)).toBe(true);
  }, 30_000);
});
