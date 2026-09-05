import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, test } from "@playwright/test";

/**
 * The template field, without running a prediction.
 *
 * What it guards is the wiring: that choosing a file produces the alignment
 * summary someone decides on, that a structure covering almost none of the
 * query says so before the run rather than after it, and that the field goes
 * away for a complex, where a template is not supported.
 */
const FRAGMENT = resolve(import.meta.dirname, "../fixtures/template/ubiquitin-fragment.pdb");
const FRAGMENT_SEQUENCE = "MQIFVKTLTGKT";
const UNRELATED = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

test("reports what the query makes of an uploaded template", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#template-field")).toBeVisible();
  await expect(page.locator("#template-summary")).toBeHidden();

  await page.locator("#sequence").fill(FRAGMENT_SEQUENCE);
  await page.locator("#template-file").setInputFiles(FRAGMENT);
  const summary = page.locator("#template-summary");
  await expect(summary).toBeVisible();
  await expect(page.locator("#template-detail"))
    .toHaveText("ubiquitin-fragment.pdb · chain A covers 100% of the query, 100% identical");
  await expect(summary).toHaveAttribute("data-state", "");
  // One chain, so there is nothing to pick between.
  await expect(page.locator("#template-chain-field")).toBeHidden();

  // A structure covering almost none of the query is the wrong file, and the
  // page says so rather than waiting for a bad prediction to say it. Twelve
  // residues against a query of 177 is under a tenth of it however they align.
  await page.locator("#sequence").fill(UNRELATED.repeat(3));
  await expect(summary).toHaveAttribute("data-state", "warning");

  await page.locator("#template-clear").click();
  await expect(summary).toBeHidden();
  await expect(page.locator("#template-file-name")).toHaveText("Choose a PDB or mmCIF file");
});

test("offers the chains of a structure that has more than one", async ({ page }) => {
  await page.goto("/");
  // Two copies of the fragment under different chain identifiers.
  const atoms = readFileSync(FRAGMENT, "utf8").split("\n").filter((line) => line.startsWith("ATOM"));
  const second = atoms.map((line) => `${line.slice(0, 21)}B${line.slice(22)}`);
  await page.locator("#template-file").setInputFiles({
    name: "two-chains.pdb", mimeType: "text/plain",
    buffer: Buffer.from([...atoms, ...second, "END"].join("\n")),
  });
  await expect(page.locator("#template-chain-field")).toBeVisible();
  await expect(page.locator("#template-chain")).toHaveValue("A");
  await expect(page.locator("#template-chain option")).toHaveCount(2);
  await page.locator("#template-chain").selectOption("B");
  await expect(page.locator("#template-detail")).toContainText("chain B");
});

test("has no template field for a complex", async ({ page }) => {
  await page.goto("/");
  await page.locator("#sequence").fill(`${UNRELATED}:${UNRELATED}`);
  await expect(page.locator("#template-field")).toBeHidden();
  await expect(page.locator("#template-summary")).toBeHidden();
});
