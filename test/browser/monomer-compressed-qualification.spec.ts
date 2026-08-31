import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_BROWSER_MONOMER_QUALIFICATION === "1";

interface Prediction {
  readonly recycles: readonly { readonly meanPlddt: number; readonly ptm: number }[];
  readonly plddt: readonly number[]; readonly pae: readonly number[]; readonly atom37: readonly number[];
}

function mae(left: readonly number[], right: readonly number[]): number {
  return left.reduce((sum, value, index) => sum + Math.abs(value - right[index]!), 0) / left.length;
}

function rms(left: readonly number[], right: readonly number[]): number {
  return Math.sqrt(left.reduce(
    (sum, value, index) => sum + (value - right[index]!) ** 2, 0,
  ) / left.length);
}

function compare(label: string, compressed: Prediction, float32: Prediction): void {
  expect(compressed.recycles, `${label} recycle count`).toHaveLength(float32.recycles.length);
  for (let recycle = 0; recycle < compressed.recycles.length; recycle += 1) {
    expect(Math.abs(compressed.recycles[recycle]!.meanPlddt - float32.recycles[recycle]!.meanPlddt),
      `${label} recycle ${recycle} mean pLDDT`).toBeLessThan(0.25);
    expect(Math.abs(compressed.recycles[recycle]!.ptm - float32.recycles[recycle]!.ptm),
      `${label} recycle ${recycle} pTM`).toBeLessThan(0.005);
  }
  expect(mae(compressed.plddt, float32.plddt), `${label} pLDDT MAE`).toBeLessThan(0.5);
  expect(mae(compressed.pae, float32.pae), `${label} PAE MAE`).toBeLessThan(0.5);
  expect(rms(compressed.atom37, float32.atom37), `${label} atom37 RMS`).toBeLessThan(0.5);
}

test.skip(!enabled, "set AFWEBGPU_BROWSER_MONOMER_QUALIFICATION=1 for full q8 qualification");
test("keeps the q8 monomer comparable for query-only and deep-MSA inputs", async ({ page }) => {
  test.setTimeout(40 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  const results = await page.evaluate(async () => {
    const qualificationUrl = "/monomer-qualification.ts";
    const module = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMonomer(modelManifestUrl: string, deepA3m: string): Promise<{
        bundleId: string; queryOnly: Prediction; deepMsa: Prediction;
      }>;
    };
    const deepA3m = await (await fetch("/qualification-assets/acceptance/test.a3m")).text();
    const float32 = await module.qualifyMonomer(
      "/qualification-assets/afwebgpu-monomer-model1-f32-v2/manifest.json", deepA3m,
    );
    const compressed = await module.qualifyMonomer(
      "/qualification-assets/model/manifest.json", deepA3m,
    );
    return { float32, compressed };
  });
  expect(results.float32.bundleId).toBe("model_1_ptm-f32-v2");
  expect(results.compressed.bundleId).toBe("model_1_ptm-q8-v1");
  compare("query-only", results.compressed.queryOnly, results.float32.queryOnly);
  compare("deep-MSA", results.compressed.deepMsa, results.float32.deepMsa);
  console.log("Chrome monomer q8 qualification passed", {
    float32: results.float32.bundleId, compressed: results.compressed.bundleId,
    queryOnlyRecycles: results.float32.queryOnly.recycles.length,
    deepMsaRecycles: results.float32.deepMsa.recycles.length,
  });
});
