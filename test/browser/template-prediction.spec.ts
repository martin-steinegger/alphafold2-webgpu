import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

/**
 * A whole prediction with a template, in a browser, through the worker.
 *
 * The unit and GPU tests reach the template module directly; this is the only
 * thing that exercises the path the page actually takes — a file read in the
 * page, the structure text crossing to the worker, the alignment and the
 * features built there, and a structure coming back.
 *
 * Needs a served model bundle:
 *
 *   AFWEBGPU_QUALIFICATION_ASSET_ROOT=/tmp/afwebgpu-model-q8-v2 \
 *   AFWEBGPU_BROWSER_TEMPLATE=1 npx playwright test template-prediction
 */
const enabled = process.env.AFWEBGPU_BROWSER_TEMPLATE === "1";
test.skip(!enabled, "set AFWEBGPU_BROWSER_TEMPLATE=1 and serve a model bundle");

const UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";
const STRUCTURE = process.env.AFWEBGPU_TEMPLATE_STRUCTURE
  ?? "/tmp/afwebgpu-template-reference-v1/1ubq.pdb";

test("folds a single sequence against an uploaded structure", async ({ page }) => {
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  await page.getByText("Advanced settings").click();
  await page.locator("#monomer-model-url").fill("/qualification-assets/model/manifest.json");
  await page.locator("#input-mode").selectOption("single");
  await page.locator("#recycles").selectOption("1");
  await page.locator("#sequence").fill(UBIQUITIN);
  await page.locator("#template-file").setInputFiles(resolve(STRUCTURE));
  await expect(page.locator("#template-detail")).toContainText("100% of the query");

  await page.locator("#predict").click();
  await expect(page.locator("#results-section")).toBeVisible({ timeout: 20 * 60_000 });
  await expect(page.locator("#result-length")).toHaveText(String(UBIQUITIN.length));

  // Ubiquitin from one sequence is a coin toss without a template and a
  // confident answer with one, so the confidence is the assertion.
  const plddt = Number(await page.locator("#mean-plddt").textContent());
  console.log(`TEMPLATE PREDICTION: pLDDT ${plddt}`);
  expect(plddt).toBeGreaterThan(85);
  await expect(page.locator("#run-log")).toContainText("Template: 1ubq.pdb chain A");
  await expect(page.locator("#run-log")).toContainText("100% coverage");
});
