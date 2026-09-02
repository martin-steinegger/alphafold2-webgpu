import { expect, test } from "@playwright/test";

const HOMOMER_CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const HOMODIMER = new Array(2).fill(HOMOMER_CHAIN).join(":");
const HOMOTRIMER = new Array(3).fill(HOMOMER_CHAIN).join(":");

test("auto-detects monomers and multimers and switches to custom A3M", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Fold a protein in your browser." })).toBeVisible();
  await expect(page.locator("#sequence-length")).toHaveText("59 residues");
  await expect(page.locator("#input-mode")).toHaveValue("mmseqs2");
  await expect(page.locator("#msa-api-url")).toHaveValue("https://api.colabfold.com");
  await expect(page.locator("#predict-label")).toHaveText("Generate MSA & predict");
  await expect(page.locator("#monomer-model-url")).toHaveValue("./model/manifest.json");
  await expect(page.locator("#multimer-model-url")).toHaveValue("./model-multimer/manifest.json");
  await expect(page.locator("#sequence")).toHaveCSS("min-height", "80px");
  await page.getByText("Advanced settings").click();
  await expect(page.getByRole("button", { name: "Clear downloaded model" })).toBeVisible();
  await expect(page.locator("#monomer-storage")).toHaveValue("f32");
  await expect(page.locator("#monomer-storage")).toBeEnabled();
  await page.locator("#monomer-storage").selectOption("f16");
  await page.locator("#sequence").fill(HOMODIMER);
  await expect(page.locator("#sequence-length")).toHaveText("118 residues · 2 chains");
  await expect(page.locator("#predict-label")).toHaveText("Generate complex MSA & predict");
  await expect(page.locator("#recycles")).toHaveValue("20");
  await expect(page.locator("#max-msa")).toBeEnabled();
  await expect(page.locator("#max-extra")).toHaveValue("2048");
  // Multimer can pack the triangle projection, so the control stays available.
  await expect(page.locator("#monomer-storage")).toBeEnabled();
  await page.locator("#input-mode").selectOption("single");
  await expect(page.locator("#predict-label")).toHaveText("Run Multimer-v3");
  await expect(page.locator("#max-msa")).toBeDisabled();
  await page.locator("#sequence").fill(HOMOTRIMER);
  await expect(page.locator("#sequence-length")).toHaveText("177 residues · 3 chains");
  await page.locator("#input-mode").selectOption("custom");
  await expect(page.locator("#a3m-field")).toBeVisible();
  await expect(page.locator("#sequence-field")).toBeHidden();

  await page.goto("/?precision=f16");
  await page.getByText("Advanced settings").click();
  await expect(page.locator("#monomer-storage")).toHaveValue("f16");
});

test("reports a WebGPU compatibility verdict before any prediction starts", async ({ page }) => {
  await page.goto("/");
  const summary = page.locator("#gpu-summary");
  await expect(summary).not.toHaveAttribute("data-state", "checking", { timeout: 20_000 });
  const state = await summary.getAttribute("data-state");
  expect(["ready", "warning", "failed"]).toContain(state);
  await expect(page.locator("#gpu-summary-text")).not.toHaveText("");
  if (state === "ready") {
    await expect(page.locator("#gpu-summary-text")).toContainText("WebGPU ready");
    await expect(page.locator("#gpu-details")).toBeHidden();
    return;
  }
  // Every non-ready verdict must tell the user what to change.
  await expect(page.locator("#gpu-details")).toBeVisible();
  expect(await page.locator("#gpu-remedies li").count()).toBeGreaterThan(0);
});
