import { readFile } from "node:fs/promises";
import { expect, test } from "@playwright/test";
import { readZipArchive, zipText } from "../zip-reader.js";

/**
 * Downloads the result archive from a real browser run, so the ZIP the page
 * writes is checked by an extractor rather than by the writer's own tests.
 *
 * Serve a monomer model through the dev server's qualification-asset route:
 *   AFWEBGPU_QUALIFICATION_ASSET_ROOT=/path/to/models \
 *   AFWEBGPU_BROWSER_MONOMER_MANIFEST=/qualification-assets/model-q8/manifest.json \
 *   AFWEBGPU_BROWSER_RESULT_PACKAGE=1 npx playwright test result-package
 */
const manifest = process.env.AFWEBGPU_BROWSER_MONOMER_MANIFEST;
const enabled = process.env.AFWEBGPU_BROWSER_RESULT_PACKAGE === "1" && manifest !== undefined;
const CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

test.skip(!enabled, "set AFWEBGPU_BROWSER_RESULT_PACKAGE=1 and AFWEBGPU_BROWSER_MONOMER_MANIFEST to run this");

test("packages a finished prediction the way ColabFold does", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");
  await page.getByText("Advanced settings").click();
  await page.locator("#monomer-model-url").fill(manifest!);
  await page.locator("#input-mode").selectOption("single");
  await page.locator("#recycles").selectOption("0");
  await page.locator("#job-name").fill("zip check");
  await page.locator("#sequence").fill(CHAIN);
  await page.locator("#predict").click();
  await expect(page.locator("#results-section")).toBeVisible({ timeout: 9 * 60_000 });

  const button = page.locator("#download-results");
  await expect(button).toBeVisible();
  const [download] = await Promise.all([page.waitForEvent("download"), button.click()]);
  expect(download.suggestedFilename()).toBe("zip_check.result.zip");
  const path = await download.path();
  const files = readZipArchive(new Uint8Array(await readFile(path)));

  expect([...files.keys()]).toEqual([
    "zip_check/zip_check_unrelaxed_model_1.pdb",
    "zip_check/zip_check_scores.json",
    "zip_check/zip_check_predicted_aligned_error_v1.json",
    "zip_check/zip_check_plddt.png",
    "zip_check/zip_check_pae.png",
    "zip_check/zip_check_coverage.png",
    "zip_check/zip_check.a3m",
    "zip_check/zip_check.csv",
    "zip_check/config.json",
    "zip_check/log.txt",
    "zip_check/cite.bib",
  ]);

  const pdb = zipText(files, "zip_check/zip_check_unrelaxed_model_1.pdb");
  expect(pdb).toContain("ATOM");
  expect(pdb.trimEnd().endsWith("END")).toBe(true);
  const scores = JSON.parse(zipText(files, "zip_check/zip_check_scores.json")) as
    { sequence: string; plddt: number[]; mean_plddt: number };
  expect(scores.sequence).toBe(CHAIN);
  expect(scores.plddt).toHaveLength(CHAIN.length);
  expect(scores.mean_plddt).toBeGreaterThan(0);
  const pae = JSON.parse(zipText(files, "zip_check/zip_check_predicted_aligned_error_v1.json")) as
    readonly { predicted_aligned_error: number[][] }[];
  expect(pae[0]!.predicted_aligned_error).toHaveLength(CHAIN.length);
  const config = JSON.parse(zipText(files, "zip_check/config.json")) as
    { model_type: string; num_recycles: number; msa_mode: string; length: number; adapter: string };
  expect(config).toMatchObject({ model_type: "alphafold2_ptm", num_recycles: 0, msa_mode: "single_sequence", length: CHAIN.length });
  expect(config.adapter.length).toBeGreaterThan(0);
  expect(zipText(files, "zip_check/zip_check.csv")).toBe(`id,sequence\nzip_check,${CHAIN}\n`);
  expect(zipText(files, "zip_check/cite.bib")).toContain("jumper2021highly");
  // A single-sequence run never touched the MMseqs2 server, so it cites neither.
  expect(zipText(files, "zip_check/cite.bib")).not.toContain("mirdita2022colabfold");
  expect(zipText(files, "zip_check/log.txt")).toContain("Finished in");
  for (const plot of ["plddt", "pae", "coverage"]) {
    const png = files.get(`zip_check/zip_check_${plot}.png`)!;
    expect(Array.from(png.subarray(0, 4))).toEqual([0x89, 0x50, 0x4e, 0x47]);
    expect(png.length).toBeGreaterThan(1000);
  }
});
