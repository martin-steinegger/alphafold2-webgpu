/**
 * What half precision costs a prediction.
 *
 * The microbenchmark in `gemm-calibration.spec.ts` measures one kernel against
 * random inputs, which does not answer the question that matters: real
 * activations are not random, rounding partly cancels, and recycling is
 * self-correcting, so a worst-case figure there can be either pessimistic or
 * optimistic about the model. This predicts the same input with the same
 * weights once per arrangement and compares confidence recycle by recycle
 * against the f32 kernel the project already ships.
 *
 * Two inputs, because one sequence can pass by luck: the 59-residue query on
 * its own for four recycles, and the same query against a 508-row clustered
 * MSA with 1,024 extra rows. The deep case is the one that matters. Pure f16
 * passes the shallow one and takes the deep one to NaN.
 *
 * Every arrangement the per-device calibration may install has to pass, since
 * it picks on hardware neither of us has. The comparison is also checked in
 * the other direction: a variant that rounds must *differ* from f32 somewhere,
 * or a toggle that silently failed to reach the kernel would report perfect
 * agreement, the one result that looks like success while proving nothing. An
 * exact variant is exempt from that: agreeing to the bit is what it is for.
 */
import { expect, test } from "@playwright/test";


const enabled = process.env.AFWEBGPU_GEMM_DIFFERENTIAL === "1";

/** The brief's gate: the change must not move the prediction this much. */
const PLDDT_GATE = 0.05;
const PTM_GATE = 0.005;

interface Candidate {
  readonly name: string;
  readonly precision: string;
  readonly inner: number;
  readonly fallback?: string;
  /** Whether the per-device selection may install it, so whether it is gated. */
  readonly shippable: boolean;
  /** Whether it rounds. An exact kernel is allowed to agree with f32 exactly. */
  readonly approximate: boolean;
}

/** Pure f16 is measured for the record; it is not in the shippable set. */
const MEASURED: readonly Candidate[] = [
  ...([8, 16] as const).flatMap((inner) => [
    { name: `f32-k${inner}`, precision: "f32", inner, shippable: true, approximate: false },
    { name: `f16-k${inner}`, precision: "f16", inner, shippable: false, approximate: true },
    { name: `f16-chunked-k${inner}`, precision: "f16-chunked", inner, shippable: true, approximate: true },
    { name: `f16-mixed-k${inner}`, precision: "f16-mixed", inner, shippable: true, approximate: true },
  ]),
  // The matrix units, with each fallback the selection can pair them with.
  // Only some callers can reach them, so the fallback is what everyone else
  // computes and both halves have to hold.
  { name: "matrix+f32", precision: "matrix", inner: 8, fallback: "f32",
    shippable: true, approximate: false },
  { name: "matrix+chunked", precision: "matrix", inner: 16, fallback: "f16-chunked",
    shippable: true, approximate: true },
];

interface Prediction {
  readonly recycles: readonly { readonly meanPlddt: number; readonly ptm: number }[];
  readonly plddt: readonly number[];
  readonly pae: readonly number[];
  readonly atom37: readonly number[];
}

interface Side {
  readonly name: string;
  readonly precision: string;
  readonly inner: number;
  readonly bundleId: string;
  readonly seconds: number;
  readonly queryOnly: Prediction;
  readonly deepMsa: Prediction;
}

function mae(left: readonly number[], right: readonly number[]): number {
  return left.reduce((sum, value, index) => sum + Math.abs(value - right[index]!), 0) / left.length;
}

function rms(left: readonly number[], right: readonly number[]): number {
  return Math.sqrt(left.reduce((sum, value, index) => sum + (value - right[index]!) ** 2, 0) / left.length);
}

interface Divergence {
  readonly plddt: number;
  readonly ptm: number;
  readonly plddtMae: number;
  readonly paeMae: number;
  readonly atomRms: number;
  readonly finite: boolean;
  readonly identical: boolean;
}

/** How far one case moved, worst over the recycles. */
function diverge(exact: Prediction, half: Prediction): Divergence {
  let plddt = 0;
  let ptm = 0;
  let finite = true;
  for (let recycle = 0; recycle < exact.recycles.length; recycle += 1) {
    const before = exact.recycles[recycle]!;
    const after = half.recycles[recycle]!;
    finite = finite && Number.isFinite(after.meanPlddt) && Number.isFinite(after.ptm);
    plddt = Math.max(plddt, Math.abs(after.meanPlddt - before.meanPlddt));
    ptm = Math.max(ptm, Math.abs(after.ptm - before.ptm));
  }
  const plddtMae = mae(half.plddt, exact.plddt);
  const paeMae = mae(half.pae, exact.pae);
  const atomRms = rms(half.atom37, exact.atom37);
  return {
    plddt, ptm, plddtMae, paeMae, atomRms,
    finite: finite && [plddtMae, paeMae, atomRms].every((value) => Number.isFinite(value)),
    identical: plddtMae === 0 && paeMae === 0 && atomRms === 0,
  };
}

function show(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : "NaN";
}

test.skip(!enabled, "set AFWEBGPU_GEMM_DIFFERENTIAL=1 and point AFWEBGPU_QUALIFICATION_ASSET_ROOT at a model");
test("every shippable projection variant predicts what the f32 kernel predicts", async ({ page }) => {
  test.setTimeout(60 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  page.on("pageerror", (error) => console.log(`page error: ${error.message}`));
  const root = process.cwd();
  await page.goto("/");

  const sides = await page.evaluate(async ({ root, variants }) => {
    // Served through /@fs because the dev server's root is web/. This is the
    // same module instance the model itself imports, so pinning here is what
    // the prediction below sees.
    const selection = await import(/* @vite-ignore */
      `/@fs${root}/src/runtime/gemm-selection.ts`) as {
        forceGemmVariant(variant: {
          precision: string; inner: number; fallback?: string;
        } | undefined): void;
      };
    const qualificationUrl = "/monomer-qualification.ts";
    const qualification = await import(/* @vite-ignore */ qualificationUrl) as {
      qualifyMonomer(modelManifestUrl: string, deepA3m: string): Promise<{
        bundleId: string; queryOnly: unknown; deepMsa: unknown;
      }>;
    };
    const deepA3m = await (await fetch("/qualification-assets/acceptance/test.a3m")).text();
    const results: unknown[] = [];
    for (const variant of variants) {
      selection.forceGemmVariant(variant);
      const started = performance.now();
      try {
        const result = await qualification.qualifyMonomer(
          "/qualification-assets/model/manifest.json", deepA3m,
        ) as unknown as Record<string, unknown>;
        results.push({
          ...result, precision: variant.precision, inner: variant.inner,
          name: variant.name, seconds: (performance.now() - started) / 1000,
        });
      } catch (error) {
        console.log(`${variant.name} threw: ${String(error).slice(0, 200)}`);
      }
    }
    selection.forceGemmVariant(undefined);
    return results;
  }, { root, variants: MEASURED.map((variant) => ({ ...variant })) });

  const measured = sides as unknown as Side[];
  const baseline = measured.find((side) => side.name === "f32-k8");
  expect(baseline, "the f32 baseline ran").toBeDefined();
  if (baseline === undefined) return;

  const lines = [`bundle: ${baseline.bundleId}`, "",
    "variant            whole run   vs f32   worst dPLDDT   worst dPTM   PAE MAE   atom RMS"];
  const divergences = new Map<string, { queryOnly: Divergence; deepMsa: Divergence }>();
  for (const side of measured) {
    const queryOnly = diverge(baseline.queryOnly, side.queryOnly);
    const deepMsa = diverge(baseline.deepMsa, side.deepMsa);
    divergences.set(side.name, { queryOnly, deepMsa });
    const shippable = MEASURED.find((entry) => entry.name === side.name)?.shippable === true;
    lines.push(`${side.name.padEnd(18)} ${side.seconds.toFixed(1).padStart(6)} s   `
      + `${(baseline.seconds / side.seconds).toFixed(2)}x   `
      + `${show(Math.max(queryOnly.plddt, deepMsa.plddt)).padStart(12)}   `
      + `${show(Math.max(queryOnly.ptm, deepMsa.ptm)).padStart(10)}   `
      + `${show(Math.max(queryOnly.paeMae, deepMsa.paeMae)).padStart(7)}   `
      + `${show(Math.max(queryOnly.atomRms, deepMsa.atomRms)).padStart(8)}`
      + (shippable ? "" : "   (not shippable)")
      + (queryOnly.finite && deepMsa.finite ? "" : "   NOT FINITE"));
  }

  // Recycle by recycle for the shippable arrangements, where compounding shows.
  for (const side of measured) {
    const entry = MEASURED.find((candidate) => candidate.name === side.name);
    if (entry === undefined || !entry.shippable || side.name === "f32-k8") continue;
    lines.push("", `${side.name}:`);
    for (const [label, exact, half] of [
      ["query-only, 4 recycles", baseline.queryOnly, side.queryOnly],
      ["deep MSA, 508 + 1024 rows", baseline.deepMsa, side.deepMsa],
    ] as const) {
      lines.push(`  ${label}`);
      for (let recycle = 0; recycle < exact.recycles.length; recycle += 1) {
        const before = exact.recycles[recycle]!;
        const after = half.recycles[recycle]!;
        lines.push(`    recycle ${recycle}: pLDDT ${before.meanPlddt.toFixed(4)} -> `
          + `${after.meanPlddt.toFixed(4)}   pTM ${before.ptm.toFixed(5)} -> ${after.ptm.toFixed(5)}`);
      }
    }
  }
  console.log(`\nGEMM DIFFERENTIAL\n${lines.join("\n")}\n`);

  for (const candidate of MEASURED) {
    if (!candidate.shippable) continue;
    const divergence = divergences.get(candidate.name);
    expect(divergence, `${candidate.name} produced a prediction`).toBeDefined();
    if (divergence === undefined) continue;
    for (const [label, result] of [
      ["query-only", divergence.queryOnly], ["deep MSA", divergence.deepMsa],
    ] as const) {
      expect(result.finite, `${candidate.name} ${label} stayed finite`).toBe(true);
      expect(result.plddt, `${candidate.name} ${label} worst mean-pLDDT shift`)
        .toBeLessThan(PLDDT_GATE);
      expect(result.ptm, `${candidate.name} ${label} worst pTM shift`).toBeLessThan(PTM_GATE);
    }
    // A kernel that rounds has to change something, or the agreement above is
    // measuring one kernel against itself. An exact one is under no such
    // obligation: agreeing with f32 to the bit is what it is for.
    if (candidate.approximate) {
      expect(divergence.queryOnly.identical && divergence.deepMsa.identical,
        `${candidate.name} changed nothing at all, so the toggle did not reach the model`)
        .toBe(false);
    }
  }
});
