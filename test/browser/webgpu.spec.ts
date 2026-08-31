import { expect, test } from "@playwright/test";

const HOMODIMER = [
  "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
  "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
].join(":");

test("shows monomer and multimer input modes and switches to custom A3M", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Fold a protein in your browser." })).toBeVisible();
  await expect(page.locator("#sequence-length")).toHaveText("59 residues");
  await expect(page.locator("#input-mode")).toHaveValue("mmseqs2");
  await expect(page.locator("#msa-api-url")).toHaveValue("https://api.colabfold.com");
  await expect(page.locator("#predict-label")).toHaveText("Generate MSA & predict");
  await expect(page.locator("#model-url")).toHaveValue("./model/manifest.json");
  await expect(page.locator("#sequence")).toHaveCSS("min-height", "112px");
  await page.getByText("Advanced settings").click();
  await expect(page.getByRole("button", { name: "Clear downloaded model" })).toBeVisible();
  await page.locator("#input-mode").selectOption("multimer");
  await expect(page.locator("#predict-label")).toHaveText("Run Multimer-v3");
  await expect(page.locator("#recycles")).toHaveValue("20");
  await expect(page.locator("#max-msa")).toBeDisabled();
  await page.locator("#sequence").fill(HOMODIMER);
  await expect(page.locator("#sequence-length")).toHaveText("118 residues · 2 chains");
  await page.locator("#input-mode").selectOption("custom");
  await expect(page.locator("#a3m-field")).toBeVisible();
  await expect(page.locator("#sequence-field")).toBeHidden();
});
