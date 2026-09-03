import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_BROWSER_MONOMER_Q8_REGRESSION === "1";

interface RecycleConfidence {
  readonly meanPlddt: number;
  readonly ptm: number;
}

test.skip(!enabled, "set AFWEBGPU_BROWSER_MONOMER_Q8_REGRESSION=1 for the q8 Metal regression");
test("keeps exact and packed q8 deep-MSA confidence correct", async ({ page }) => {
  test.setTimeout(20 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  const results = await page.evaluate(async () => {
    const qualificationUrl = "/monomer-qualification.ts";
    const module = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMonomer(modelManifestUrl: string, deepA3m: string, storageOptions?: {
        readonly triangleWholeStorage?: "f32" | "f16";
        readonly msaStorage?: "f32" | "f16";
        readonly pairStorage?: "f32" | "f16";
      }): Promise<{
        readonly bundleId: string;
        readonly deepMsa: { readonly recycles: readonly RecycleConfidence[] };
      }>;
    };
    const deepA3m = await (await fetch("/qualification-assets/acceptance/test.a3m")).text();
    const exact = await module.qualifyMonomer("/qualification-assets/model/manifest.json", deepA3m);
    // The storages every prediction runs with.
    const packed = await module.qualifyMonomer("/qualification-assets/model/manifest.json", deepA3m, {
      triangleWholeStorage: "f16", msaStorage: "f16", pairStorage: "f16",
    });
    return { exact, packed };
  });
  expect(results.exact.bundleId).toBe("model_1_ptm-q8-v1");
  expect(results.packed.bundleId).toBe("model_1_ptm-q8-v1");
  const exact = results.exact.deepMsa.recycles[0]!;
  const packed = results.packed.deepMsa.recycles[0]!;
  expect(exact.meanPlddt).toBeGreaterThan(90);
  expect(exact.ptm).toBeGreaterThan(0.65);
  expect(Math.abs(packed.meanPlddt - exact.meanPlddt)).toBeLessThan(0.25);
  expect(Math.abs(packed.ptm - exact.ptm)).toBeLessThan(0.005);
  console.log("q8 deep-MSA storage qualification", { exact, packed });
});
