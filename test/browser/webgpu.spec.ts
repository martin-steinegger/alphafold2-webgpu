import { expect, test } from "@playwright/test";


test("shows the one-model prediction UI and switches to custom A3M", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Fold a protein in your browser." })).toBeVisible();
  await expect(page.locator("#sequence-length")).toHaveText("59 residues");
  await expect(page.locator("#input-mode")).toHaveValue("mmseqs2");
  await expect(page.locator("#msa-api-url")).toHaveValue("https://api.colabfold.com");
  await expect(page.locator("#predict-label")).toHaveText("Generate MSA & predict");
  await expect(page.locator("#model-url")).toHaveValue("./model/manifest.json");
  await expect(page.locator("#sequence")).toHaveCSS("min-height", "160px");
  await page.getByText("Advanced settings").click();
  await expect(page.getByRole("button", { name: "Clear downloaded model" })).toBeVisible();
  await page.locator("#input-mode").selectOption("custom");
  await expect(page.locator("#a3m-field")).toBeVisible();
  await expect(page.locator("#sequence-field")).toBeHidden();
});
