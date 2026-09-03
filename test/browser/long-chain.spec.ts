import { expect, test } from "@playwright/test";

/**
 * A 1500-residue prediction through the page.
 *
 * At that length the pair and the triangle multiplication's whole projection
 * are 549 MiB each, well past the 128 MiB storage binding a WebGPU device gets
 * by default, so this checks that the plan asks the adapter for what the shape
 * needs and that the run survives it. Single-sequence input keeps the MSA side
 * small; the pair side, which is what the limits are about, is unchanged.
 */
const manifest = process.env.AFWEBGPU_BROWSER_MONOMER_MANIFEST;
const enabled = process.env.AFWEBGPU_BROWSER_LONG_CHAIN === "1" && manifest !== undefined;
const RESIDUES = Number(process.env.AFWEBGPU_LONG_CHAIN_LENGTH ?? "1500");

test.skip(!enabled, "set AFWEBGPU_BROWSER_LONG_CHAIN=1 and AFWEBGPU_BROWSER_MONOMER_MANIFEST to run this");

test("predicts a 1500-residue chain through the page", async ({ page }) => {
  test.setTimeout(45 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  const amino = "ACDEFGHIKLMNPQRSTVWY";
  let state = 0x5eed;
  const sequence = Array.from({ length: RESIDUES }, () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return amino[state % 20];
  }).join("");
  await page.goto("/");
  await page.getByText("Advanced settings").click();
  await page.locator("#monomer-model-url").fill(manifest!);
  await page.locator("#input-mode").selectOption("single");
  await page.locator("#recycles").selectOption("0");
  await page.locator("#sequence").fill(sequence);
  await page.locator("#predict").click();
  // A recycle at this length is minutes of GPU work, so the stage line has to
  // move while it runs: blocks completed, and once both stacks have been timed
  // the minutes left. Without it a working run is indistinguishable from a
  // stalled one, which is what users report.
  await expect(page.locator('[data-stage="inference"] small'))
    .toHaveText(/Evoformer block \d+\/\d+/, { timeout: 20 * 60_000 });
  await expect(page.locator("#results-section")).toBeVisible({ timeout: 40 * 60_000 });
  await expect(page.locator("#result-length")).toHaveText(String(RESIDUES));
  const prediction = await page.evaluate(() => window.__AFWEBGPU_PREDICTION__);
  expect(prediction?.meanPlddt).toBeGreaterThan(0);
  const log = await page.locator("#run-log").textContent();
  console.log("LONG CHAIN:", JSON.stringify(prediction), "\nlimits:",
    log?.split("\n").filter((line) => line.includes("limits") || line.includes("Estimated peak")).join(" | "));
});
