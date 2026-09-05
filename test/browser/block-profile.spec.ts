/**
 * Where the time actually goes inside one Evoformer block.
 *
 * The projections were optimised on the assumption that they were worth
 * optimising, which is not the same as knowing it. At 1,416 residues the main
 * Evoformer stack is 92% of the run — 689 seconds of 746 — and everything
 * else, the structure module and the confidence heads included, is about one
 * percent. So the only question that matters is what a block is made of, and
 * `ExecutionContext` already timestamps every dispatch and labels it.
 *
 * This profiles one block of a real prediction at a real length. It is the
 * measurement that says whether any further work on the dense projections can
 * pay, or whether the time is somewhere else entirely.
 */
import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_BLOCK_PROFILE === "1";
const CHAIN = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";
const copies = Number(process.env.AFWEBGPU_COMPLEX_CHAINS ?? "24");

test.skip(!enabled, "set AFWEBGPU_BLOCK_PROFILE=1 and point AFWEBGPU_QUALIFICATION_ASSET_ROOT at a model");
test(`profiles one Evoformer block of a ${copies}-chain complex`, async ({ page }) => {
  test.setTimeout(6 * 60 * 60_000);
  const profileLines: string[] = [];
  page.on("console", (message) => {
    const text = message.text();
    if (/^browser: (profile|  )/u.test(text) || /profile /u.test(text)) profileLines.push(text);
    console.log(text);
  });
  page.on("pageerror", (error) => console.log(`page error: ${error.message}`));
  const root = process.cwd();
  await page.goto("/?worker=0");

  await page.evaluate(async ({ root, chain, copies }) => {
    const expression = await import(/* @vite-ignore */
      `/@fs${root}/src/input/sequence-expression.ts`) as {
        parseSequenceExpression(value: string): {
          readonly chains: readonly string[]; readonly sequence: string; readonly multimer: boolean;
        };
      };
    const inferenceUrl = "/inference.ts";
    const inference = await import(/* @vite-ignore */ inferenceUrl) as {
      runInference(job: unknown, reporter: unknown): Promise<unknown>;
    };
    const parsed = expression.parseSequenceExpression(
      Array.from({ length: copies }, () => chain).join(":"));
    const started = performance.now();
    const reporter = {
      stage: (stage: string, state: string, detail: string) => {
        if (state === "done" || /block 1\//u.test(detail)) {
          console.log(`[${((performance.now() - started) / 1000).toFixed(0)} s] ${stage} ${detail}`);
        }
      },
      status: () => undefined,
      // The profile arrives here, one line per labelled dispatch.
      log: (text: string) => console.log(text),
      modelProgress: () => undefined,
      recycle: () => undefined,
    };
    await inference.runInference({
      manifestUrl: new URL(
        "/qualification-assets/model-multimer/manifest.json", location.href).href,
      input: {
        a3m: `>query\n${parsed.sequence}\n`, sequence: parsed.sequence,
        depth: 1, multimer: parsed.multimer, chains: parsed.chains,
      },
      maxMsaSequences: 1, maxExtraSequences: 1, recycles: 0,
      randomSeed: 0, compactPolicy: false,
      // One extra-MSA block and one main block, from the first recycle.
      profile: { recycle: 0, extraMsaBlock: 0, mainEvoformerBlock: 0 },
    }, reporter);
  }, { root, chain: CHAIN, copies });

  // Aggregate the per-dispatch lines into what a block is actually made of.
  const totals = new Map<string, { milliseconds: number; count: number }>();
  for (const line of profileLines) {
    const entry = /^(?:browser: )?\s{2}(\S+)\s+([\d.]+)ms$/u.exec(line);
    if (entry === null) continue;
    const label = entry[1]!.replace(/[.:-]?\d+$/u, "");
    const previous = totals.get(label) ?? { milliseconds: 0, count: 0 };
    totals.set(label, {
      milliseconds: previous.milliseconds + Number(entry[2]), count: previous.count + 1,
    });
  }
  const ranked = [...totals].sort((left, right) => right[1].milliseconds - left[1].milliseconds);
  const sum = ranked.reduce((total, [, value]) => total + value.milliseconds, 0);
  const report = ranked.map(([label, value]) =>
    `  ${value.milliseconds.toFixed(3).padStart(10)} ms  ${(100 * value.milliseconds / sum)
      .toFixed(1).padStart(5)}%  x${String(value.count).padEnd(4)} ${label}`);
  console.log(`\nBLOCK PROFILE (${copies} chains)\ntotal ${sum.toFixed(1)} ms over `
    + `${ranked.length} labels\n${report.join("\n")}\n`);
  expect(ranked.length, "the profile produced labelled dispatches").toBeGreaterThan(0);
});
