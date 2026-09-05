/**
 * What narrowing the attention keys and values costs a prediction.
 *
 * The flash kernel reads a key and a value for every key index and every
 * query, which is most of what a long prediction does: triangle attention is
 * 57% of an Evoformer block at 708 residues and rises with length. Packing
 * those two operands as half words measured 1.29x on the kernel, and this is
 * the question that decides whether it can ship.
 *
 * It is the same gate the projections had to pass and for the same reasons.
 * Two inputs, because one sequence can pass by luck and the deep one is where
 * a reduction goes wrong. Recycle by recycle, because error compounds. And the
 * comparison is checked in both directions: half precision has to change
 * something, or the toggle never reached the kernel and the agreement means
 * nothing.
 */
import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_ATTENTION_DIFFERENTIAL === "1";

/** The same gate the projections passed. */
const PLDDT_GATE = 0.05;
const PTM_GATE = 0.005;

interface Prediction {
  readonly recycles: readonly { readonly meanPlddt: number; readonly ptm: number }[];
  readonly plddt: readonly number[];
  readonly pae: readonly number[];
  readonly atom37: readonly number[];
}

interface Side {
  readonly storage: string;
  readonly seconds: number;
  readonly bundleId: string;
  readonly queryOnly: Prediction;
  readonly deepMsa: Prediction;
}

function mae(left: readonly number[], right: readonly number[]): number {
  return left.reduce((sum, value, index) => sum + Math.abs(value - right[index]!), 0) / left.length;
}

function rms(left: readonly number[], right: readonly number[]): number {
  return Math.sqrt(left.reduce((sum, value, index) => sum + (value - right[index]!) ** 2, 0) / left.length);
}

test.skip(!enabled, "set AFWEBGPU_ATTENTION_DIFFERENTIAL=1 and point AFWEBGPU_QUALIFICATION_ASSET_ROOT at a model");
test("half-precision attention keys and values predict what single precision predicts", async ({ page }) => {
  test.setTimeout(60 * 60_000);
  page.on("console", (message) => console.log(message.text()));
  page.on("pageerror", (error) => console.log(`page error: ${error.message}`));
  const root = process.cwd();
  await page.goto("/");

  const sides = await page.evaluate(async ({ root }) => {
    const attention = await import(/* @vite-ignore */
      `/@fs${root}/src/evoformer/attention.ts`) as {
        forceAttentionKeyValueStorage(storage: string | undefined): void;
      };
    const qualificationUrl = "/monomer-qualification.ts";
    const qualification = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMonomer(modelManifestUrl: string, deepA3m: string): Promise<{
        bundleId: string; queryOnly: unknown; deepMsa: unknown;
      }>;
    };
    const deepA3m = await (await fetch("/qualification-assets/acceptance/test.a3m")).text();
    const results: unknown[] = [];
    for (const storage of ["f32", "f16"]) {
      attention.forceAttentionKeyValueStorage(storage);
      const started = performance.now();
      const result = await qualification.qualifyMonomer(
        "/qualification-assets/model/manifest.json", deepA3m,
      ) as unknown as Record<string, unknown>;
      results.push({ ...result, storage, seconds: (performance.now() - started) / 1000 });
    }
    attention.forceAttentionKeyValueStorage(undefined);
    return results;
  }, { root });

  const measured = sides as unknown as Side[];
  const exact = measured.find((side) => side.storage === "f32");
  const half = measured.find((side) => side.storage === "f16");
  expect(exact, "the f32 side ran").toBeDefined();
  expect(half, "the f16 side ran").toBeDefined();
  if (exact === undefined || half === undefined) return;

  const lines = [`bundle: ${exact.bundleId}`,
    `f32 keys and values: ${exact.seconds.toFixed(1)} s      `
    + `f16: ${half.seconds.toFixed(1)} s      ${(exact.seconds / half.seconds).toFixed(3)}x`];
  let identical = true;
  for (const [label, before, after] of [
    ["query-only, 4 recycles", exact.queryOnly, half.queryOnly],
    ["deep MSA, 508 + 1024 rows", exact.deepMsa, half.deepMsa],
  ] as const) {
    lines.push(`  ${label}`);
    for (let recycle = 0; recycle < before.recycles.length; recycle += 1) {
      const from = before.recycles[recycle]!;
      const to = after.recycles[recycle]!;
      lines.push(`    recycle ${recycle}: pLDDT ${from.meanPlddt.toFixed(4)} -> ${to.meanPlddt.toFixed(4)}`
        + `   pTM ${from.ptm.toFixed(5)} -> ${to.ptm.toFixed(5)}`);
      expect(Number.isFinite(to.meanPlddt) && Number.isFinite(to.ptm),
        `${label} recycle ${recycle} stayed finite`).toBe(true);
      expect(Math.abs(to.meanPlddt - from.meanPlddt),
        `${label} recycle ${recycle} mean pLDDT`).toBeLessThan(PLDDT_GATE);
      expect(Math.abs(to.ptm - from.ptm), `${label} recycle ${recycle} pTM`).toBeLessThan(PTM_GATE);
    }
    const plddtMae = mae(after.plddt, before.plddt);
    const paeMae = mae(after.pae, before.pae);
    const atomRms = rms(after.atom37, before.atom37);
    lines.push(`    per-residue pLDDT MAE ${plddtMae.toFixed(4)}   PAE MAE ${paeMae.toFixed(4)}`
      + `   atom37 RMS ${atomRms.toFixed(4)} A`);
    identical &&= plddtMae === 0 && paeMae === 0 && atomRms === 0;
  }
  console.log(`\nATTENTION DIFFERENTIAL\n${lines.join("\n")}\n`);
  // Half precision has to change something, or the toggle did not reach the
  // kernel and the agreement above is one kernel measured against itself.
  expect(identical, "f16 keys and values changed nothing at all").toBe(false);
});
