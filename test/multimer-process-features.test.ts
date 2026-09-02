import { expandClusteredMsaFeatures } from "../src/input/msa-features.js";
import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";
import { makeMultimerA3mFeatures } from "../src/input/multimer-features.js";
import { assembleComplexA3m } from "../src/input/mmseqs2-api.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const ROOT = "test/fixtures/multimer-process";

describe("ColabFold Multimer process_features", () => {
  it("matches official paired/unpaired tensors through recycle 3", async () => {
    const [unpairedA, unpairedB, pairedA, pairedB] = await Promise.all([
      "unpaired-a.a3m", "unpaired-b.a3m", "paired-a.a3m", "paired-b.a3m",
    ].map((name) => readFile(`${ROOT}/${name}`, "utf8")));
    const assembled = assembleComplexA3m(
      ["ACDE", "GHIK"], ["ACDE", "GHIK"], [unpairedA!, unpairedB!], [pairedA!, pairedB!],
    );
    const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(
      "test/fixtures/evoformer/model1-query-59-stack/manifest.json",
    ));
    const features = makeMultimerA3mFeatures(
      ["ACDE", "GHIK"], assembled.a3m, assembled.mask, await model.queryOnlyFeatureTables(),
      { recycles: 3, randomSeed: 0, maxMsaSequences: 3, maxExtraSequences: 4 },
    );
    const official = await FileTensorStore.open(`${ROOT}/official/manifest.json`);
    expect(features).toHaveLength(4);
    for (let recycle = 0; recycle < features.length; recycle += 1) {
      const actual = features[recycle]!;
      const tensor = (name: string): Promise<Float32Array> =>
        official.tensor(`feature_${name}_recycle${recycle}`);
      expect(actual.msaSequences).toBe(official.shape(`feature_msa_feat_recycle${recycle}`)[0]);
      expect(actual.extraSequences).toBe(official.shape(`feature_extra_msa_recycle${recycle}`)[0]);
      expect(actual.msaMask).toEqual(await tensor("msa_mask"));
      expect(actual.extraMsa).toEqual(await tensor("extra_msa"));
      expect(actual.extraMsaMask).toEqual(await tensor("extra_msa_mask"));
      const expectedMsa = await tensor("msa_feat");
      const msaError = errorMetrics(
        expandClusteredMsaFeatures(actual.msaFeatures, actual.msaSequences * actual.aatype.length), expectedMsa);
      // JAX's GPU einsum uses reduced-precision accumulation for a three-way
      // nearest-neighbor tie; discrete rows/codes above must still match exactly.
      expect(msaError.meanAbsoluteError).toBeLessThan(2e-6);
      expect(msaError.maxAbsoluteError).toBeLessThan(2e-4);
      expect(errorMetrics(actual.extraHasDeletion,
        await tensor("extra_has_deletion")).maxAbsoluteError).toBeLessThan(1e-6);
      expect(errorMetrics(actual.extraDeletionValue,
        await tensor("extra_deletion_value")).maxAbsoluteError).toBeLessThan(1e-6);
    }
  });
});
