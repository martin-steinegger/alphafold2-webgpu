import { expect, test } from "@playwright/test";

const a3mPath = process.env.AFWEBGPU_CUSTOM_MULTIMER_A3M?.trim();

test.skip(a3mPath === undefined || a3mPath === "", "set AFWEBGPU_CUSTOM_MULTIMER_A3M to a ColabFold complex A3M");
test("predicts a ColabFold serialized complex A3M", async ({ page }) => {
  test.setTimeout(30 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/?compact=1");
  await page.getByText("Advanced settings").click();
  await page.locator("#multimer-model-url").fill("/qualification-assets/model-multimer/manifest.json");
  await page.locator("#input-mode").selectOption("custom");
  await page.locator("#a3m-file").setInputFiles(a3mPath!);
  await expect(page.locator("#max-extra")).toHaveValue("2048");
  await page.locator("#recycles").selectOption("3");
  await page.locator("#predict").click();
  await page.waitForFunction(() => {
    const results = document.querySelector<HTMLElement>("#results-section");
    const status = document.querySelector<HTMLElement>("#prediction-status");
    return results?.hidden === false || status?.dataset.state === "failed";
  }, undefined, { timeout: 30 * 60_000 });
  await expect(page.locator("#prediction-status")).toHaveAttribute("data-state", "passed");
  await expect(page.locator("#result-length")).toHaveText("210");
  const prediction = await page.evaluate(() => window.__AFWEBGPU_PREDICTION__);
  expect(prediction?.meanPlddt).toBeGreaterThan(0);
  expect(prediction?.iptm).toBeGreaterThanOrEqual(0);
  console.log("Custom ColabFold Multimer acceptance:", prediction);
});
