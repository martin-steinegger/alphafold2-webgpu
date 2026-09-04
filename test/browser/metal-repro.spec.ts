import { expect, test } from "@playwright/test";

interface Prediction {
  readonly recycles: readonly { readonly meanPlddt: number; readonly ptm: number }[];
}

test("keeps the q8 deep-MSA prediction correct on Apple Metal", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");

  const result = await page.evaluate(async () => {
    if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    const info = adapter.info;
    const qualificationUrl = "/monomer-qualification.ts";
    const module = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMonomer(modelManifestUrl: string, deepA3m: string): Promise<{
        bundleId: string; deepMsa: Prediction;
      }>;
    };
    const a3m = await (await fetch("/qualification-assets/acceptance/test.a3m")).text();
    const qualification = await module.qualifyMonomer(
      "/qualification-assets/model/manifest.json", a3m,
    );
    return {
      adapterInfo: {
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
      },
      bundleId: qualification.bundleId,
      deepMsa: qualification.deepMsa,
    };
  });

  console.log("Chrome WebGPU adapter", result.adapterInfo);
  console.log("Metal deep-MSA recycles", result.deepMsa.recycles);
  expect(result.bundleId).toBe("model_1_ptm-q8-v1");
  expect(result.deepMsa.recycles[0]!.meanPlddt).toBeGreaterThan(90);
  expect(result.deepMsa.recycles[0]!.ptm).toBeGreaterThan(0.65);
});
