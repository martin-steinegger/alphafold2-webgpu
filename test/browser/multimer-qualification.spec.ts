import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_BROWSER_MULTIMER_QUALIFICATION === "1";

test.skip(!enabled, "set AFWEBGPU_BROWSER_MULTIMER_QUALIFICATION=1 for the full model qualification");
test("runs model_1_multimer_v3 in Chrome WebGPU against official JAX", async ({ page }) => {
  test.setTimeout(20 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  const results = await page.evaluate(async () => {
    const qualificationUrl = "/multimer-qualification.ts";
    const qualification = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMultimer(referenceManifestUrl: string, modelManifestUrl: string, enforceOfficial?: boolean): Promise<{
        meanPlddt: number; ptm: number; iptm: number; rankingConfidence: number;
        atom37MeanAbsoluteError: number; plddtMeanAbsoluteError: number; paeMeanAbsoluteError: number;
        plddt: number[]; pae: number[]; atom37: number[];
      }>;
    };
    const mae = (a: number[], b: number[]): number =>
      a.reduce((sum, value, index) => sum + Math.abs(value - b[index]!), 0) / a.length;
    const rms = (a: number[], b: number[]): number => Math.sqrt(
      a.reduce((sum, value, index) => sum + (value - b[index]!) ** 2, 0) / a.length,
    );
    const results = [];
    for (const reference of [
      "/qualification-assets/afwebgpu-multimer-reference-query-v2/manifest.json",
      "/qualification-assets/afwebgpu-multimer-paired-reference-v1/manifest.json",
    ]) {
      const f32 = await qualification.qualifyMultimer(
        reference, "/qualification-assets/afwebgpu-multimer-model1-f32-v2/manifest.json",
      );
      const compressed = await qualification.qualifyMultimer(
        reference, "/qualification-assets/model-multimer/manifest.json", false,
      );
      results.push({ reference,
        f32AtomMae: f32.atom37MeanAbsoluteError, f32PlddtMae: f32.plddtMeanAbsoluteError,
        f32PaeMae: f32.paeMeanAbsoluteError,
        meanPlddtDifference: Math.abs(compressed.meanPlddt - f32.meanPlddt),
        ptmDifference: Math.abs(compressed.ptm - f32.ptm),
        iptmDifference: Math.abs(compressed.iptm - f32.iptm),
        plddtMae: mae(compressed.plddt, f32.plddt), paeMae: mae(compressed.pae, f32.pae),
        atomRms: rms(compressed.atom37, f32.atom37) });
    }
    return results;
  });
  for (const result of results) {
    expect(result.f32AtomMae).toBeLessThan(0.25);
    expect(result.f32PlddtMae).toBeLessThan(0.5);
    expect(result.f32PaeMae).toBeLessThan(0.5);
    expect(result.meanPlddtDifference).toBeLessThan(0.25);
    expect(result.ptmDifference).toBeLessThan(0.005);
    expect(result.iptmDifference).toBeLessThan(0.005);
    expect(result.plddtMae).toBeLessThan(0.5);
    expect(result.paeMae).toBeLessThan(0.5);
    expect(result.atomRms).toBeLessThan(0.5);
  }
  console.log("Chrome Multimer qualification:", results);
});
