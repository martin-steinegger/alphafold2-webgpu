import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_LIVE_MMSEQS2 === "1";
const CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const HETERODIMER = [
  "MESAQAVAEPLDLVRLSLDEIVYVKLRGDRELNGRLHAYDEHLNMVLGDAEEIVTIFDDEETDKDKALKTIRKHYEMLFVRGDSVILIAPPRN",
  "MKLVRFLMKLTNETVSIELKNGTIVHGTITSVDMQMNTHLKAVKMTVKGREPVPVETLSIRGNNIRYYILPDSLPLDTLLIDDSTKPKQKKKEVVRGRGRGRGRGTRGRGRGASRG",
].join(":");

async function prepareLivePrediction(
  page: import("@playwright/test").Page,
  sequence: string,
  recycles = 1,
): Promise<void> {
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  await page.getByText("Advanced settings").click();
  await page.locator("#multimer-model-url").fill(
    "/qualification-assets/model-multimer/manifest.json",
  );
  await page.locator("#input-mode").selectOption("mmseqs2");
  await page.locator("#sequence").fill(sequence);
  await page.locator("#recycles").selectOption(String(recycles));
  await expect(page.locator("#recycles")).toHaveValue(String(recycles));
  await page.locator("#predict").click();
  await expect(page.locator("#results-section")).toBeVisible({ timeout: 30 * 60_000 });
}

test.skip(!enabled, "set AFWEBGPU_LIVE_MMSEQS2=1 to query the public ColabFold API");
test("predicts the acceptance homodimer with a ColabFold MMseqs2 MSA", async ({ page }) => {
  test.setTimeout(30 * 60_000);
  await prepareLivePrediction(page, `${CHAIN}:${CHAIN}`);
  await expect(page.locator("#result-length")).toHaveText("118");
  await expect(page.locator("#run-log")).toContainText("MMseqs2: Unpaired: Alignment ready");
  const prediction = await page.evaluate(() => window.__AFWEBGPU_PREDICTION__);
  expect(prediction?.meanPlddt).toBeGreaterThan(0);
  expect(prediction?.iptm).toBeGreaterThanOrEqual(0);
  console.log("Live MMseqs2 homodimer acceptance:", prediction);
});

test("predicts a heterodimer with paired and unpaired ColabFold MSAs", async ({ page }) => {
  test.setTimeout(30 * 60_000);
  await prepareLivePrediction(page, HETERODIMER, 3);
  await expect(page.locator("#result-length")).toHaveText("209");
  await expect(page.locator("#run-log")).toContainText("MMseqs2: Paired: Alignment ready");
  await expect(page.locator("#run-log")).toContainText("MMseqs2: Unpaired: Alignment ready");
  const prediction = await page.evaluate(() => window.__AFWEBGPU_PREDICTION__);
  expect(prediction?.meanPlddt).toBeGreaterThan(0);
  expect(prediction?.iptm).toBeGreaterThanOrEqual(0);
  console.log("Live MMseqs2 heterodimer acceptance:", prediction);
});
