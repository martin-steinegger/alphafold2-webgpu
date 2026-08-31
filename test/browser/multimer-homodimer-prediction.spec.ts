import { expect, test, type Page } from "@playwright/test";

const enabled = process.env.AFWEBGPU_BROWSER_MULTIMER_QUALIFICATION === "1";
const CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

test.skip(!enabled, "set AFWEBGPU_BROWSER_MULTIMER_QUALIFICATION=1 for the full model qualification");
async function runQueryOnlyOligomer(page: Page, copies: number): Promise<void> {
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  await page.getByText("Advanced settings").click();
  await page.locator("#multimer-model-url").fill(
    "/qualification-assets/model-multimer/manifest.json",
  );
  await page.locator("#input-mode").selectOption("single");
  await page.locator("#recycles").selectOption("1");
  await page.locator("#sequence").fill(Array.from({ length: copies }, () => CHAIN).join(":"));
  await page.locator("#predict").click();
  await expect(page.locator("#results-section")).toBeVisible({ timeout: 20 * 60_000 });
  await expect(page.locator("#result-length")).toHaveText(String(CHAIN.length * copies));
  await expect(page.locator("#mean-plddt")).not.toHaveText("—");
  await expect(page.locator("#run-log")).toContainText("recycle=1");
  const prediction = await page.evaluate(() => window.__AFWEBGPU_PREDICTION__);
  expect(prediction?.meanPlddt).toBeGreaterThan(0);
  expect(prediction?.meanPlddt).toBeLessThanOrEqual(100);
  expect(prediction?.iptm).toBeGreaterThanOrEqual(0);
  expect(prediction?.iptm).toBeLessThanOrEqual(1);
  console.log(`GB10/Apple ${copies}-mer smoke:`, prediction);
}

test("completes the 118-residue homodimer through the unified UI", async ({ page }) => {
  test.setTimeout(20 * 60_000);
  await runQueryOnlyOligomer(page, 2);
});

test("completes the 177-residue homotrimer through the unified UI", async ({ page }) => {
  test.setTimeout(20 * 60_000);
  await runQueryOnlyOligomer(page, 3);
});
