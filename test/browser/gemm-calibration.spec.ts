/**
 * Which projection kernel is fastest on this machine.
 *
 * One hand-tiled GEMM serves every dense projection in the model, and the tile
 * it uses was chosen on a Linux workstation. Which tile wins depends on the
 * shape and on the GPU, and half-precision arithmetic is available on Apple
 * and not here, so the numbers that matter can only be taken in the browser
 * that will run the model. This measures every candidate at the shapes the
 * model actually runs and prints a table to paste back.
 */
import { test } from "@playwright/test";
import { CANDIDATES, SHAPES } from "../../tools/gemm-candidates.js";

test.skip(process.env.AFWEBGPU_GEMM_CALIBRATION !== "1", "set AFWEBGPU_GEMM_CALIBRATION=1");

test("measures every projection kernel", async ({ page }) => {
  test.setTimeout(20 * 60_000);
  await page.goto("/");
  const report = await page.evaluate(async ({ candidates, shapes }) => {
    const lines: string[] = [];
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) return ["no WebGPU adapter"];
    const wanted = ["shader-f16", "subgroups", "subgroup-size-control",
      "chromium-experimental-subgroup-matrix"];
    const features = wanted.filter((name) => adapter.features.has(name as GPUFeatureName));
    const device = await adapter.requestDevice({ requiredFeatures: features as GPUFeatureName[] });
    lines.push(`features: ${[...adapter.features].sort().join(", ")}`);
    lines.push(`granted: ${features.join(", ") || "none"}`);
    const info = adapter.info as GPUAdapterInfo | undefined;
    lines.push(`adapter: ${info?.vendor ?? "?"} ${info?.architecture ?? ""} ${info?.description ?? ""}`.trim());
    // The shapes and component types the hardware's matrix units accept. A
    // tensor-core kernel has to be written against these, and they differ by
    // vendor: this Linux NVIDIA adapter offers integer ones only, so a float
    // kernel for Apple cannot be designed, let alone compiled, without them.
    const configs = (adapter.info as unknown as {
      subgroupMatrixConfigs?: readonly Record<string, unknown>[];
    }).subgroupMatrixConfigs ?? [];
    lines.push(`subgroup matrix configs: ${configs.length === 0 ? "none" : ""}`);
    for (const config of configs) {
      lines.push(`  ${["componentType", "resultComponentType", "M", "N", "K"]
        .map((key) => `${key}=${String(config[key])}`).join(" ")}`);
    }

    const random = (count: number): Float32Array => {
      let state = 0x1234567;
      return Float32Array.from({ length: count }, () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x100000000 - 0.5;
      });
    };
    for (const shape of shapes) {
      lines.push(`== ${shape.name} ==`);
      const source = device.createBuffer({ size: shape.rows * shape.inner * 4, usage: 128 | 8 });
      device.queue.writeBuffer(source, 0, random(shape.rows * shape.inner));
      const weightCount = shape.inner * shape.columns + shape.columns;
      const weights = device.createBuffer({ size: weightCount * 4, usage: 128 | 8 });
      device.queue.writeBuffer(weights, 0, random(weightCount));
      const params = device.createBuffer({ size: 32, usage: 64 | 8 });
      device.queue.writeBuffer(params, 0, new Uint32Array([
        shape.rows, shape.inner, shape.columns, shape.inner * shape.columns, 0, 0, 0, 0,
      ]));
      const output = device.createBuffer({ size: shape.rows * shape.columns * 4, usage: 128 });
      const flops = 2 * shape.rows * shape.inner * shape.columns;
      let best = { name: "", milliseconds: Number.POSITIVE_INFINITY };
      let baseline = 0;
      for (const candidate of candidates) {
        if (candidate.requiresF16 === true && !device.features.has("shader-f16" as GPUFeatureName)) continue;
        device.pushErrorScope("validation");
        const pipeline = device.createComputePipeline({
          label: candidate.name, layout: "auto",
          compute: { module: device.createShaderModule({ code: candidate.shader }), entryPoint: "main" },
        });
        const failure = await device.popErrorScope();
        if (failure !== null) { lines.push(`  ${candidate.name} rejected: ${failure.message.split("\n")[0]}`); continue; }
        const group = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [source, weights, params, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const x = Math.ceil(shape.columns / candidate.tileColumns);
        const y = Math.ceil(shape.rows / candidate.tileRows);
        // A browser clamps performance.now to about 0.1 ms and a submission
        // costs a millisecond or two to come back, both of which swamp a
        // kernel of a few hundred microseconds. One command buffer therefore
        // carries many dispatches, and the batch is what gets timed.
        const batch = async (iterations: number): Promise<number> => {
          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline); pass.setBindGroup(0, group);
          for (let iteration = 0; iteration < iterations; iteration += 1) pass.dispatchWorkgroups(x, y, 1);
          pass.end();
          const started = performance.now();
          device.queue.submit([encoder.finish()]);
          await device.queue.onSubmittedWorkDone();
          return (performance.now() - started) / iterations;
        };
        const rough = await batch(4);
        const iterations = Math.max(4, Math.min(2000, Math.ceil(60 / Math.max(rough, 0.01))));
        let milliseconds = Number.POSITIVE_INFINITY;
        for (let repeat = 0; repeat < 3; repeat += 1) milliseconds = Math.min(milliseconds, await batch(iterations));
        if (candidate.name.startsWith("current")) baseline = milliseconds;
        if (milliseconds < best.milliseconds) best = { name: candidate.name, milliseconds };
        lines.push(`  ${candidate.name.padEnd(22)} ${milliseconds.toFixed(3)} ms  `
          + `${(flops / milliseconds / 1e9).toFixed(2)} TFLOP/s`);
      }
      lines.push(`  -> fastest ${best.name}`
        + (baseline > 0 ? ` (${(baseline / best.milliseconds).toFixed(2)}x over current)` : ""));
      for (const buffer of [source, weights, params, output]) buffer.destroy();
    }
    device.destroy();
    return lines;
  }, { candidates: CANDIDATES.map((c) => ({ ...c })), shapes: SHAPES.map((s) => ({ ...s })) });
  console.log(`\nGEMM CALIBRATION\n${report.join("\n")}\n`);
});
