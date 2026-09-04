/**
 * The longest complex this machine can predict, with and without half
 * precision.
 *
 * Two things are being checked, and they are not the same thing. A projection
 * kernel must not cost the project its longest runnable input: half precision
 * halves what the k tile stages, so it should if anything help, but nothing
 * about that is obvious and a capacity floor is not something to assume. And
 * the prediction gate wants a complex as its second input, since a single
 * chain can pass by luck, and a complex exercises chain breaks, the multimer
 * template and iptm.
 *
 * The chain count is a parameter because cost grows with the square of the
 * total length: a 24-mer of the 59-residue acceptance sequence is 1,416
 * residues and hours of work, while a dimer settles the differential in
 * minutes. `AFWEBGPU_COMPLEX_CHAINS` picks how many, and
 * `AFWEBGPU_COMPLEX_VARIANTS` which arrangements to run.
 */
import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_LONG_COMPLEX === "1";
const CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const copies = Number(process.env.AFWEBGPU_COMPLEX_CHAINS ?? "2");
const variants = (process.env.AFWEBGPU_COMPLEX_VARIANTS ?? "f32-k8,f16-chunked-k16")
  .split(",").map((entry) => entry.trim()).filter((entry) => entry.length > 0);

interface Summary {
  readonly name: string;
  readonly seconds: number;
  readonly length: number;
  readonly meanPlddt: number;
  readonly ptm: number;
  readonly iptm: number;
  readonly rankingConfidence: number;
  readonly recycles: readonly {
    readonly meanPlddt: number; readonly ptm: number; readonly iptm: number;
  }[];
  readonly plddt: readonly number[];
  readonly failure?: string;
}

test.skip(!enabled, "set AFWEBGPU_LONG_COMPLEX=1 and point AFWEBGPU_QUALIFICATION_ASSET_ROOT at a multimer model");
test(`predicts a ${copies}-chain complex with each projection variant`, async ({ page }) => {
  // A 24-mer is hours of GPU work, so this is deliberately generous.
  test.setTimeout(20 * 60 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  page.on("pageerror", (error) => console.log(`page error: ${error.message}`));
  const root = process.cwd();
  // No worker: the prediction has to run in this module graph for the pinned
  // variant to reach it.
  await page.goto("/?worker=0");

  const summaries = await page.evaluate(async ({ root, chain, copies, variants }) => {
    const selection = await import(/* @vite-ignore */
      `/@fs${root}/src/runtime/gemm-selection.ts`) as {
        forceGemmVariant(variant: { precision: string; inner: number } | undefined): void;
      };
    const expression = await import(/* @vite-ignore */
      `/@fs${root}/src/input/sequence-expression.ts`) as {
        parseSequenceExpression(value: string): {
          readonly chains: readonly string[]; readonly sequence: string; readonly multimer: boolean;
        };
      };
    const inferenceUrl = "/inference.ts";
    const inference = await import(/* @vite-ignore */ inferenceUrl) as {
      runInference(job: unknown, reporter: unknown): Promise<{
        prediction: Record<string, unknown>;
      }>;
      resetInferenceDevice(): void;
    };

    const text = Array.from({ length: copies }, () => chain).join(":");
    const parsed = expression.parseSequenceExpression(text);
    const input = {
      a3m: `>query\n${parsed.sequence}\n`, sequence: parsed.sequence,
      depth: 1, multimer: parsed.multimer, chains: parsed.chains,
    };
    const results: Summary[] = [];
    for (const requested of variants) {
      const match = /^(.*)-k(8|16)$/u.exec(requested);
      if (match === null) continue;
      const variant = { precision: match[1]!, inner: Number(match[2]) };
      const recycles: { meanPlddt: number; ptm: number; iptm: number }[] = [];
      const reporter = {
        stage: () => undefined,
        status: () => undefined,
        log: (text: string) => { if (/memory|budget|adapter|window/i.test(text)) console.log(text); },
        modelProgress: () => undefined,
        recycle: (summary: Record<string, unknown>, index: number) => {
          const confidence = (summary.confidence ?? summary) as Record<string, number>;
          recycles.push({
            meanPlddt: confidence.meanPlddt ?? Number.NaN,
            ptm: confidence.ptm ?? Number.NaN,
            iptm: confidence.iptm ?? Number.NaN,
          });
          console.log(`${requested} recycle ${index}: pLDDT `
            + `${(confidence.meanPlddt ?? Number.NaN).toFixed(4)} ptm `
            + `${(confidence.ptm ?? Number.NaN).toFixed(5)} iptm `
            + `${(confidence.iptm ?? Number.NaN).toFixed(5)}`);
        },
      };
      // A retained device would carry the previous variant's compiled
      // pipelines, and one cache key cannot describe two shaders: the
      // pipeline cache reports that as a collision rather than running. Each
      // variant therefore gets a device of its own, which is also what
      // production does, since the choice is made before a device exists.
      inference.resetInferenceDevice();
      selection.forceGemmVariant(variant);
      const started = performance.now();
      try {
        const outcome = await inference.runInference({
          manifestUrl: new URL(
            "/qualification-assets/model-multimer/manifest.json", location.href).href,
          input, maxMsaSequences: 1, maxExtraSequences: 1, recycles: 1,
          randomSeed: 0, compactPolicy: false,
        }, reporter);
        const prediction = outcome.prediction;
        const final = prediction.final as Record<string, Record<string, unknown>> | undefined;
        const confidence = final?.confidence as unknown as Record<string, number> | undefined;
        const plddt = final?.confidence?.plddt as Float32Array | undefined;
        results.push({
          name: requested, seconds: (performance.now() - started) / 1000,
          length: parsed.sequence.replace(/:/gu, "").length,
          meanPlddt: confidence?.meanPlddt ?? Number.NaN,
          ptm: confidence?.ptm ?? Number.NaN,
          iptm: confidence?.iptm ?? Number.NaN,
          rankingConfidence: confidence?.rankingConfidence ?? Number.NaN,
          recycles, plddt: plddt === undefined ? [] : Array.from(plddt),
        });
      } catch (error) {
        const failure = String(error);
        console.log(`${requested} failed after `
          + `${((performance.now() - started) / 1000).toFixed(0)} s: ${failure.slice(0, 300)}`);
        results.push({
          name: requested, seconds: (performance.now() - started) / 1000,
          length: parsed.sequence.replace(/:/gu, "").length,
          meanPlddt: Number.NaN, ptm: Number.NaN, iptm: Number.NaN,
          rankingConfidence: Number.NaN, recycles, plddt: [], failure,
        });
      }
    }
    selection.forceGemmVariant(undefined);
    return results;
  }, { root, chain: CHAIN, copies, variants });

  const results = summaries as unknown as Summary[];
  const lines = [`${copies} chains, ${results[0]?.length ?? "?"} residues`, "",
    "variant            elapsed    mean pLDDT      pTM     ipTM   ranking"];
  for (const result of results) {
    lines.push(`${result.name.padEnd(18)} ${result.seconds.toFixed(0).padStart(6)} s   `
      + `${result.meanPlddt.toFixed(4).padStart(10)}  ${result.ptm.toFixed(5)}  `
      + `${result.iptm.toFixed(5)}  ${result.rankingConfidence.toFixed(4)}`
      + (result.failure === undefined ? "" : `   FAILED: ${result.failure.slice(0, 120)}`));
  }
  const baseline = results.find((result) => result.name === "f32-k8" && result.failure === undefined);
  for (const result of results) {
    if (baseline === undefined || result === baseline || result.failure !== undefined) continue;
    lines.push("", `${result.name} against f32-k8:`);
    lines.push(`  ${(baseline.seconds / result.seconds).toFixed(3)}x`);
    for (let recycle = 0; recycle < Math.min(baseline.recycles.length, result.recycles.length); recycle += 1) {
      const before = baseline.recycles[recycle]!;
      const after = result.recycles[recycle]!;
      lines.push(`  recycle ${recycle}: dPLDDT ${Math.abs(after.meanPlddt - before.meanPlddt).toFixed(4)}`
        + `  dPTM ${Math.abs(after.ptm - before.ptm).toFixed(5)}`
        + `  dIPTM ${Math.abs(after.iptm - before.iptm).toFixed(5)}`);
    }
  }
  console.log(`\nLONG COMPLEX\n${lines.join("\n")}\n`);

  // Capacity first: whatever the kernel, this input has to come out finite.
  for (const result of results) {
    expect(result.failure, `${result.name} completed the ${copies}-chain complex`).toBeUndefined();
    expect(Number.isFinite(result.meanPlddt), `${result.name} produced a finite pLDDT`).toBe(true);
    expect(result.meanPlddt, `${result.name} pLDDT is in range`).toBeGreaterThan(0);
    expect(result.meanPlddt, `${result.name} pLDDT is in range`).toBeLessThanOrEqual(100);
    expect(Number.isFinite(result.iptm), `${result.name} produced a finite ipTM`).toBe(true);
  }
  // Then the gate, against the f32 kernel on the same input.
  for (const result of results) {
    if (baseline === undefined || result === baseline) continue;
    for (let recycle = 0; recycle < baseline.recycles.length; recycle += 1) {
      const before = baseline.recycles[recycle]!;
      const after = result.recycles[recycle]!;
      expect(Math.abs(after.meanPlddt - before.meanPlddt),
        `${result.name} recycle ${recycle} mean pLDDT`).toBeLessThan(0.05);
      expect(Math.abs(after.ptm - before.ptm),
        `${result.name} recycle ${recycle} pTM`).toBeLessThan(0.005);
      expect(Math.abs(after.iptm - before.iptm),
        `${result.name} recycle ${recycle} ipTM`).toBeLessThan(0.005);
    }
  }
});
