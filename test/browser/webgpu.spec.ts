import { expect, test } from "@playwright/test";

test("runs the triangle kernel in a browser WebGPU implementation", async ({ page }) => {
  await page.goto("/?autorun=0&length=5&cz=7&hidden=6&precision=f32");
  const available = await page.evaluate(async () => {
    if (navigator.gpu === undefined) return false;
    return (await navigator.gpu.requestAdapter()) !== null;
  });
  test.skip(!available, "the installed browser has no usable WebGPU adapter");
  await page.getByText("WebGPU kernel diagnostic").click();
  await page.getByRole("button", { name: "Run kernel" }).click();
  await expect(page.locator("#status")).toHaveAttribute("data-state", "passed");
  const metrics = await page.evaluate(() => window.__AFWEBGPU_RESULT__);
  expect(metrics?.meanAbsoluteError).toBeLessThan(1e-5);
  expect(metrics?.maxAbsoluteError).toBeLessThan(1e-4);
});

test("runs fp16 inputs and weights when the browser exposes shader-f16", async ({ page }) => {
  await page.goto("/?autorun=0&length=8&cz=8&hidden=8&precision=f16");
  const hasF16 = await page.evaluate(async () => {
    const adapter = await navigator.gpu?.requestAdapter();
    return adapter?.features.has("shader-f16") ?? false;
  });
  test.skip(!hasF16, "the browser adapter does not expose shader-f16");
  await page.getByText("WebGPU kernel diagnostic").click();
  await page.getByRole("button", { name: "Run kernel" }).click();
  await expect(page.locator("#status")).toHaveAttribute("data-state", "passed");
  const metrics = await page.evaluate(() => window.__AFWEBGPU_RESULT__);
  expect(metrics?.meanAbsoluteError).toBeLessThan(1e-3);
  expect(metrics?.maxAbsoluteError).toBeLessThan(1e-2);
});

test("shows the one-model prediction UI and switches to custom A3M", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Fold a protein in your browser." })).toBeVisible();
  await expect(page.locator("#sequence-length")).toHaveText("59 residues");
  await expect(page.locator("#input-mode")).toHaveValue("mmseqs2");
  await expect(page.locator("#msa-api-url")).toHaveValue("https://api.colabfold.com");
  await expect(page.locator("#predict-label")).toHaveText("Generate MSA & predict");
  await expect(page.locator("#model-url")).toHaveValue("./model/manifest.json");
  await page.getByText("Advanced settings").click();
  await expect(page.getByRole("button", { name: "Clear downloaded model" })).toBeVisible();
  await page.locator("#input-mode").selectOption("custom");
  await expect(page.locator("#a3m-field")).toBeVisible();
  await expect(page.locator("#sequence-field")).toBeHidden();
});
