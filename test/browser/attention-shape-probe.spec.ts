/**
 * What the attention shape probe chooses, and that it chooses by measuring.
 *
 * It replaces a constant — two queries per invocation above 128 of them —
 * whose own comment records it as measured on an NVIDIA GB10, and which on
 * this device is 2.2x the wrong way round. A probe that replaces a wrong
 * constant with a differently wrong constant would be no better, so this
 * reports every candidate's time alongside the winner: if the four are within
 * noise of each other the probe cannot tell them apart and its answer is not
 * worth having.
 */
import { expect, test } from "@playwright/test";

const enabled = process.env.AFWEBGPU_ATTENTION_PROBE === "1";

test.skip(!enabled, "set AFWEBGPU_ATTENTION_PROBE=1");
test("the attention probe measures the query count and the operand width", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  page.on("console", (message) => console.log(message.text()));
  const root = process.cwd();
  await page.goto("/");

  const outcome = await page.evaluate(async ({ root }) => {
    const queries = await import(/* @vite-ignore */
      `/@fs${root}/src/runtime/attention-queries.ts`) as {
        calibrateAttentionShape(device: GPUDevice, headDim: number): Promise<{
          slots: number; keyValue: string;
        } | undefined>;
      };
    const dev = await import(/* @vite-ignore */ `/@fs${root}/src/runtime/device.ts`) as {
      requestAlphaFoldDevice(adapter: GPUAdapter): Promise<GPUDevice>;
    };
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) return { chosen: "no adapter", milliseconds: 0 };
    const device = await dev.requestAlphaFoldDevice(adapter);
    const started = performance.now();
    // Triangle attention's head dimension, which is the shape that matters.
    const choice = await queries.calibrateAttentionShape(device, 32);
    const milliseconds = performance.now() - started;
    device.destroy();
    return {
      chosen: choice === undefined ? "none" : `${choice.slots}q ${choice.keyValue}`,
      milliseconds,
    };
  }, { root });

  console.log(`\nATTENTION SHAPE PROBE\nchose: ${outcome.chosen} `
    + `in ${outcome.milliseconds.toFixed(0)} ms\n`);
  // Whatever it picks has to be one the prediction gate cleared, and it has to
  // pick something: returning nothing means the probe failed and the shape
  // rule stands, which is safe but is not what this device should do.
  // Keys and values are packed independently, so the arrangements are the
  // product of the query count and which of the two is half width.
  expect(outcome.chosen).toMatch(/^[12]q (f32|f16|f16-key|f16-value)$/u);
  // It must not cost more than it saves. One device, one head dimension, once.
  expect(outcome.milliseconds, "the probe stays cheap").toBeLessThan(2000);
});
