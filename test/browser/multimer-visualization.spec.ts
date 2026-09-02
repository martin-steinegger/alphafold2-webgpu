import { expect, test } from "@playwright/test";

/**
 * Checks that a complex reads as a complex: chains marked in the plots, a
 * colour key under the viewer, and a table of per-chain confidence and
 * chain-pair alignment error.
 *
 * Serve a Multimer-v3 model through the dev server's qualification-asset route:
 *   AFWEBGPU_QUALIFICATION_ASSET_ROOT=/path/to/models \
 *   AFWEBGPU_BROWSER_MULTIMER_MANIFEST=/qualification-assets/model-multimer/manifest.json \
 *   npx playwright test multimer-visualization
 */
const manifest = process.env.AFWEBGPU_BROWSER_MULTIMER_MANIFEST;
const monomerManifest = process.env.AFWEBGPU_BROWSER_MONOMER_MANIFEST;
const CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

test.skip(manifest === undefined, "set AFWEBGPU_BROWSER_MULTIMER_MANIFEST to run this");

test("marks the chains of a complex in every result view", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  await page.getByText("Advanced settings").click();
  await page.locator("#multimer-model-url").fill(manifest!);
  await page.locator("#input-mode").selectOption("single");
  await page.locator("#recycles").selectOption("0");
  await page.locator("#sequence").fill(`${CHAIN}:${CHAIN}`);
  await page.locator("#predict").click();
  await expect(page.locator("#results-section")).toBeVisible({ timeout: 9 * 60_000 });

  // The viewer opens coloured by chain, and the key names both chains.
  await expect(page.locator("#viewer-color")).toHaveValue("chain");
  await expect(page.locator("#structure-viewer canvas")).toBeVisible({ timeout: 60_000 });
  const legend = page.locator("#chain-legend .chain-swatch");
  await expect(legend).toHaveCount(2);
  await expect(legend.first()).toContainText(`A · ${CHAIN.length} aa · pLDDT`);
  await expect(legend.nth(1)).toContainText(`B · ${CHAIN.length} aa · pLDDT`);

  // Chain, residues, pLDDT, then one alignment-error column per chain.
  await expect(page.locator("#chain-card")).toBeVisible();
  await expect(page.locator("#chain-head th")).toHaveText(["Chain", "Residues", "pLDDT", "PAE A", "PAE B"]);
  const rows = page.locator("#chain-rows tr");
  await expect(rows).toHaveCount(2);
  const cells = await rows.first().locator("td").allTextContents();
  expect(cells[0]).toBe("A");
  expect(cells[1]).toBe(String(CHAIN.length));
  expect(Number(cells[2])).toBeGreaterThan(0);
  // A homodimer's two chains fold alike, so each diagonal block beats the
  // interface block that places one chain against the other.
  expect(Number(cells[3])).toBeLessThan(Number(cells[4]));

  // The ranking score ColabFold sorts complexes by.
  await expect(page.locator("#ranking-card")).toBeVisible();
  expect(Number(await page.locator("#ranking").textContent())).toBeGreaterThan(0);

  // Clicking an off-diagonal block of the matrix isolates that interface.
  const canvas = page.locator("#pae-plot");
  const box = await canvas.boundingBox();
  const scale = box!.width / 560;
  await expect(page.locator("#pae-hint")).toContainText("Click a block to isolate");
  await canvas.click({ position: { x: 371 * scale, y: 141 * scale } });
  await expect(page.locator("#pae-hint")).toContainText("Chains A and B");
  await page.screenshot({ path: "/tmp/claude-1000/-home-steineggerlab-martin-fsinterface-afwebgl/c5c210f9-d817-4636-aef1-df98e3e6bdef/scratchpad/multimer-interface.png", fullPage: true });
  await canvas.click({ position: { x: 371 * scale, y: 141 * scale } });
  await expect(page.locator("#pae-hint")).toContainText("Click a block to isolate");
  // A diagonal block isolates the one chain.
  await canvas.click({ position: { x: 141 * scale, y: 141 * scale } });
  await expect(page.locator("#pae-hint")).toContainText("Chain A alone");

  await page.locator("#viewer-color").selectOption("plddt");
  await expect(page.locator("#viewer-color")).toHaveValue("plddt");
});

test("leaves the chain views out of a single-chain prediction", async ({ page }) => {
  test.skip(monomerManifest === undefined, "set AFWEBGPU_BROWSER_MONOMER_MANIFEST to run this");
  test.setTimeout(10 * 60_000);
  await page.goto("/");
  await page.getByText("Advanced settings").click();
  await page.locator("#monomer-model-url").fill(monomerManifest!);
  await page.locator("#input-mode").selectOption("single");
  await page.locator("#recycles").selectOption("0");
  await page.locator("#sequence").fill(CHAIN);
  await page.locator("#predict").click();
  await expect(page.locator("#results-section")).toBeVisible({ timeout: 9 * 60_000 });
  await expect(page.locator("#chain-card")).toBeHidden();
  await expect(page.locator("#chain-legend")).toBeHidden();
  await expect(page.locator("#viewer-color")).toHaveValue("plddt");
});
