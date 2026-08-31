import { expect, test } from "@playwright/test";

const reference = process.env.AFWEBGPU_MULTIMER_LIVE_REFERENCE_URL?.trim();
const model = process.env.AFWEBGPU_MULTIMER_LIVE_MODEL_URL?.trim();
const enabled = reference !== undefined && reference !== "" && model !== undefined && model !== "";

test.skip(!enabled, "set the live Multimer reference and model URLs for a full-shape differential");
test("matches a full-shape official Multimer reference in Chrome WebGPU", async ({ page }) => {
  test.setTimeout(20 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  const result = await page.evaluate(async ({ referenceUrl, modelUrl }) => {
    const qualificationUrl = "/multimer-qualification.ts";
    const qualification = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMultimer(referenceManifestUrl: string, modelManifestUrl: string): Promise<{
        meanPlddt: number; ptm: number; iptm: number;
        atom37MeanAbsoluteError: number; plddtMeanAbsoluteError: number; paeMeanAbsoluteError: number;
      }>;
    };
    return qualification.qualifyMultimer(referenceUrl, modelUrl);
  }, { referenceUrl: reference!, modelUrl: model! });
  expect(result.atom37MeanAbsoluteError).toBeLessThan(0.25);
  expect(result.plddtMeanAbsoluteError).toBeLessThan(0.5);
  expect(result.paeMeanAbsoluteError).toBeLessThan(0.5);
  console.log("Chrome full-shape Multimer qualification:", result);
});
