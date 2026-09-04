/**
 * What the projection kernel is worth across a whole prediction.
 *
 * The microbenchmark times one kernel in isolation; this times the model. The
 * two answer different questions, because the projections are only part of a
 * recycle: a kernel worth 1.2x on its own is worth that much only of the
 * fraction of the run it occupies.
 *
 * Whole-prediction wall clock on a laptop drifts, and badly: run eight full
 * predictions back to back and the later ones are several percent slower than
 * the first whatever kernel they use, which is enough to invert a ranking this
 * close. So the variants are interleaved rather than run in blocks, and each
 * is scored by its fastest round, the one least touched by whatever the
 * machine was doing. A single pass in a fixed order measures the order.
 */
import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_GEMM_TIMING === "1";

const ROUNDS = 4;

const VARIANTS: readonly { readonly precision: string; readonly inner: number }[] = [
  { precision: "f32", inner: 8 },
  { precision: "f32", inner: 16 },
  { precision: "f16-chunked", inner: 8 },
  { precision: "f16-chunked", inner: 16 },
  { precision: "f16-mixed", inner: 8 },
];

const name = (variant: { precision: string; inner: number }): string =>
  `${variant.precision}-k${variant.inner}`;

test.skip(!enabled, "set AFWEBGPU_GEMM_TIMING=1 and point AFWEBGPU_QUALIFICATION_ASSET_ROOT at a model");
test("times each projection variant across a whole prediction", async ({ page }) => {
  test.setTimeout(60 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  const root = process.cwd();
  await page.goto("/");

  const rounds = await page.evaluate(async ({ root, variants, count }) => {
    const selection = await import(/* @vite-ignore */
      `/@fs${root}/src/runtime/gemm-selection.ts`) as {
        forceGemmVariant(variant: { precision: string; inner: number } | undefined): void;
      };
    const qualificationUrl = "/monomer-qualification.ts";
    const qualification = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMonomer(modelManifestUrl: string, deepA3m: string): Promise<unknown>;
    };
    const deepA3m = await (await fetch("/qualification-assets/acceptance/test.a3m")).text();
    const timings: { name: string; round: number; seconds: number }[] = [];
    for (let round = 0; round < count; round += 1) {
      // Interleaved, and the order reversed on alternate rounds so no variant
      // always follows the same neighbour.
      const order = round % 2 === 0 ? variants : [...variants].reverse();
      for (const variant of order) {
        selection.forceGemmVariant(variant);
        const started = performance.now();
        await qualification.qualifyMonomer(
          "/qualification-assets/model/manifest.json", deepA3m,
        );
        timings.push({
          name: `${variant.precision}-k${variant.inner}`, round,
          seconds: (performance.now() - started) / 1000,
        });
      }
    }
    selection.forceGemmVariant(undefined);
    return timings;
  }, { root, variants: VARIANTS.map((variant) => ({ ...variant })), count: ROUNDS });

  const best = new Map<string, number>();
  const all = new Map<string, number[]>();
  for (const timing of rounds) {
    best.set(timing.name, Math.min(best.get(timing.name) ?? Infinity, timing.seconds));
    all.set(timing.name, [...(all.get(timing.name) ?? []), timing.seconds]);
  }
  const baseline = best.get("f32-k8");
  expect(baseline, "the f32 baseline ran").toBeDefined();
  if (baseline === undefined) return;

  const lines = [`${ROUNDS} interleaved rounds, whole prediction (query-only + deep MSA)`, "",
    "variant            best     vs f32   every round"];
  for (const variant of VARIANTS) {
    const key = name(variant);
    const fastest = best.get(key)!;
    lines.push(`${key.padEnd(18)} ${fastest.toFixed(2)} s   ${(baseline / fastest).toFixed(3)}x   `
      + (all.get(key) ?? []).map((seconds) => seconds.toFixed(2)).join("  "));
  }
  console.log(`\nGEMM WHOLE-PREDICTION TIMING\n${lines.join("\n")}\n`);
});
