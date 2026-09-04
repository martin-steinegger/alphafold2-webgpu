/**
 * The longest complex this machine can predict at all.
 *
 * Deliberately free of anything this branch added, so the same file runs on
 * `main`: no pinned kernel, no import of `gemm-selection`, whatever the device
 * selects for itself. That is the only way to tell a capacity ceiling that was
 * always there from one a kernel change introduced, and the difference matters
 * more than the speedup does.
 *
 * Every stage is logged, because a renderer that dies mid-prediction leaves
 * nothing else behind to say how far it got.
 */
import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_CAPACITY === "1";
const CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const copies = Number(process.env.AFWEBGPU_COMPLEX_CHAINS ?? "24");

test.skip(!enabled, "set AFWEBGPU_CAPACITY=1 and point AFWEBGPU_QUALIFICATION_ASSET_ROOT at a multimer model");
test(`predicts a ${copies}-chain complex`, async ({ page }) => {
  test.setTimeout(6 * 60 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  page.on("pageerror", (error) => console.log(`page error: ${error.message}`));
  page.on("crash", () => console.log("PAGE CRASHED"));
  const root = process.cwd();
  await page.goto("/?worker=0");

  const outcome = await page.evaluate(async ({ root, chain, copies }) => {
    const expression = await import(/* @vite-ignore */
      `/@fs${root}/src/input/sequence-expression.ts`) as {
        parseSequenceExpression(value: string): {
          readonly chains: readonly string[]; readonly sequence: string; readonly multimer: boolean;
        };
      };
    const inferenceUrl = "/inference.ts";
    const inference = await import(/* @vite-ignore */ inferenceUrl) as {
      runInference(job: unknown, reporter: unknown): Promise<{ prediction: Record<string, unknown> }>;
    };
    const parsed = expression.parseSequenceExpression(
      Array.from({ length: copies }, () => chain).join(":"));
    const started = performance.now();
    const at = (): string => `[${((performance.now() - started) / 1000).toFixed(0)} s]`;
    const reporter = {
      stage: (stage: string, state: string, detail: string) =>
        console.log(`${at()} stage ${stage} ${state}: ${detail}`),
      status: () => undefined,
      log: (text: string) => console.log(`${at()} ${text}`),
      modelProgress: () => undefined,
      recycle: (summary: Record<string, unknown>, index: number) => {
        const confidence = (summary.confidence ?? summary) as Record<string, number>;
        console.log(`${at()} recycle ${index}: pLDDT ${confidence.meanPlddt?.toFixed(4)} `
          + `ptm ${confidence.ptm?.toFixed(5)} iptm ${confidence.iptm?.toFixed(5)}`);
      },
    };
    try {
      const result = await inference.runInference({
        manifestUrl: new URL(
          "/qualification-assets/model-multimer/manifest.json", location.href).href,
        input: {
          a3m: `>query\n${parsed.sequence}\n`, sequence: parsed.sequence,
          depth: 1, multimer: parsed.multimer, chains: parsed.chains,
        },
        maxMsaSequences: 1, maxExtraSequences: 1, recycles: 0,
        randomSeed: 0, compactPolicy: false,
      }, reporter);
      const final = result.prediction.final as Record<string, Record<string, number>>;
      return {
        seconds: (performance.now() - started) / 1000,
        residues: parsed.sequence.replace(/:/gu, "").length,
        meanPlddt: final.confidence.meanPlddt, ptm: final.confidence.ptm,
        iptm: final.confidence.iptm, failure: undefined as string | undefined,
      };
    } catch (error) {
      return {
        seconds: (performance.now() - started) / 1000,
        residues: parsed.sequence.replace(/:/gu, "").length,
        meanPlddt: Number.NaN, ptm: Number.NaN, iptm: Number.NaN, failure: String(error),
      };
    }
  }, { root, chain: CHAIN, copies });

  console.log(`\nCAPACITY\n${copies} chains, ${outcome.residues} residues, `
    + `${outcome.seconds.toFixed(0)} s\n`
    + (outcome.failure === undefined
      ? `pLDDT ${outcome.meanPlddt.toFixed(4)}  pTM ${outcome.ptm.toFixed(5)}  `
        + `ipTM ${outcome.iptm.toFixed(5)}\n`
      : `FAILED: ${outcome.failure}\n`));
  expect(outcome.failure, `the ${copies}-chain complex completed`).toBeUndefined();
  expect(Number.isFinite(outcome.meanPlddt)).toBe(true);
});
