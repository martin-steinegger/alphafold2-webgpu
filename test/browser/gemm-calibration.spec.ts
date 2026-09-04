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

    const matrixOffered = (candidate: typeof candidates[number]): boolean => {
      const wants = candidate.requiresSubgroupMatrix;
      if (wants === undefined) return true;
      return configs.some((config) => String(config.componentType) === wants.componentType
        && Number(config.M) === wants.size && Number(config.N) === wants.size && Number(config.K) === wants.size);
    };
    /** Raw f16 bytes back to f32, for a kernel whose result is half precision. */
    const fromHalves = (values: Uint16Array): Float32Array => Float32Array.from(values, (bits) => {
      const sign = (bits & 0x8000) !== 0 ? -1 : 1;
      const exponent = (bits >> 10) & 0x1f;
      const mantissa = bits & 0x3ff;
      if (exponent === 0) return sign * mantissa * 2 ** -24;
      if (exponent === 31) return mantissa === 0 ? sign * Infinity : Number.NaN;
      return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
    });
    /** f32 values as raw f16 bytes, for the matrix kernels that read them. */
    const halves = (values: Float32Array): Uint16Array => {
      const out = new Uint16Array(values.length);
      const view = new DataView(new ArrayBuffer(4));
      for (let index = 0; index < values.length; index += 1) {
        view.setFloat32(0, values[index]!);
        const bits = view.getUint32(0);
        const sign = (bits >>> 16) & 0x8000;
        const exponent = ((bits >>> 23) & 0xff) - 127 + 15;
        const mantissa = (bits >>> 13) & 0x3ff;
        out[index] = exponent <= 0 ? sign : exponent >= 31 ? sign | 0x7c00 : sign | (exponent << 10) | mantissa;
      }
      return out;
    };
    const random = (count: number): Float32Array => {
      let state = 0x1234567;
      return Float32Array.from({ length: count }, () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x100000000 - 0.5;
      });
    };
    // Accuracy before speed: half precision accumulates a K-long reduction in
    // f16, and a kernel that is fast and wrong is worth nothing. One small
    // aligned shape is checked against a reference computed here.
    const check = { rows: 64, inner: 256, columns: 64 };
    const checkSource = random(check.rows * check.inner);
    const checkWeights = random(check.inner * check.columns + check.columns);
    checkWeights.fill(0, check.inner * check.columns);
    const reference = new Float32Array(check.rows * check.columns);
    for (let row = 0; row < check.rows; row += 1) {
      for (let column = 0; column < check.columns; column += 1) {
        let total = 0;
        for (let k = 0; k < check.inner; k += 1) {
          total += checkSource[row * check.inner + k]! * checkWeights[k * check.columns + column]!;
        }
        reference[row * check.columns + column] = total;
      }
    }
    // A kernel that computes the wrong thing must not be able to win on
    // speed, so what fails here is marked and excluded from the ranking.
    const wrong = new Set<string>();
    lines.push(`== accuracy, M=${check.rows} K=${check.inner} N=${check.columns} (no bias) ==`);
    for (const candidate of candidates) {
      if (candidate.requiresF16 === true && !device.features.has("shader-f16" as GPUFeatureName)) continue;
      if (candidate.requiresSubgroupMatrix !== undefined && !matrixOffered(candidate)) continue;
      const halfSource = candidate.sourceFormat === "f16";
      const halfWeights = candidate.weightFormat === "f16";
      const source = device.createBuffer({
        size: check.rows * check.inner * (halfSource ? 2 : 4), usage: 128 | 8 });
      const weights = device.createBuffer({
        size: (check.inner * check.columns + check.columns) * (halfWeights ? 2 : 4), usage: 128 | 8 });
      device.queue.writeBuffer(source, 0, halfSource ? halves(checkSource) : checkSource);
      device.queue.writeBuffer(weights, 0, halfWeights ? halves(checkWeights) : checkWeights);
      const params = device.createBuffer({ size: 32, usage: 64 | 8 });
      device.queue.writeBuffer(params, 0, new Uint32Array([
        check.rows, check.inner, check.columns, 0, check.inner * check.columns, 0, 0, 0]));
      const outputBytes = check.rows * check.columns * (candidate.outputFormat === "f16" ? 2 : 4);
      const output = device.createBuffer({ size: outputBytes, usage: 128 | 4 });
      const readback = device.createBuffer({ size: outputBytes, usage: 1 | 8 });
      device.pushErrorScope("validation");
      const pipeline = device.createComputePipeline({
        label: candidate.name, layout: "auto",
        compute: { module: device.createShaderModule({ code: candidate.shader }), entryPoint: "main" },
      });
      const failure = await device.popErrorScope();
      if (failure !== null) {
        wrong.add(candidate.name);
        lines.push(`  ${candidate.name} rejected: ${failure.message.split("\n")[0]}`);
      } else {
        const group = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [source, weights, params, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(check.columns / candidate.tileColumns),
          Math.ceil(check.rows / candidate.tileRows), 1);
        pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(1);
        const raw = readback.getMappedRange().slice(0);
        const values = candidate.outputFormat === "f16" ? fromHalves(new Uint16Array(raw)) : new Float32Array(raw);
        readback.unmap();
        let worst = 0;
        let scale = 0;
        for (let index = 0; index < values.length; index += 1) {
          worst = Math.max(worst, Math.abs(values[index]! - reference[index]!));
          scale = Math.max(scale, Math.abs(reference[index]!));
        }
        const relative = 100 * worst / scale;
        if (relative > 1) wrong.add(candidate.name);
        lines.push(`  ${candidate.name.padEnd(22)} worst error ${worst.toExponential(2)} `
          + `(${relative.toFixed(3)}% of the largest value)${relative > 1 ? "   WRONG" : ""}`);
      }
      for (const buffer of [source, weights, params, output, readback]) buffer.destroy();
    }

    for (const shape of shapes) {
      lines.push(`== ${shape.name} ==`);
      const aligned = (candidate: typeof candidates[number]): boolean => candidate.alignment === undefined
        || (shape.rows % candidate.alignment === 0 && shape.inner % candidate.alignment === 0
          && shape.columns % candidate.alignment === 0);
      const sourceValues = random(shape.rows * shape.inner);
      const weightCount = shape.inner * shape.columns + shape.columns;
      const weightValues = random(weightCount);
      const source = device.createBuffer({ size: shape.rows * shape.inner * 4, usage: 128 | 8 });
      device.queue.writeBuffer(source, 0, sourceValues);
      const weights = device.createBuffer({ size: weightCount * 4, usage: 128 | 8 });
      device.queue.writeBuffer(weights, 0, weightValues);
      // A kernel that declares half-precision bindings gets half-precision
      // data: feeding it f32 bytes measures a kernel reading nonsense.
      const sourceHalf = device.createBuffer({ size: shape.rows * shape.inner * 2, usage: 128 | 8 });
      device.queue.writeBuffer(sourceHalf, 0, halves(sourceValues));
      const weightsHalf = device.createBuffer({ size: weightCount * 2, usage: 128 | 8 });
      device.queue.writeBuffer(weightsHalf, 0, halves(weightValues));
      const params = device.createBuffer({ size: 32, usage: 64 | 8 });
      device.queue.writeBuffer(params, 0, new Uint32Array([
        shape.rows, shape.inner, shape.columns, 0, shape.inner * shape.columns, 1, 0, 0,
      ]));
      const output = device.createBuffer({ size: shape.rows * shape.columns * 4, usage: 128 });
      const outputHalf = device.createBuffer({ size: shape.rows * shape.columns * 2, usage: 128 });
      const flops = 2 * shape.rows * shape.inner * shape.columns;
      let best = { name: "", milliseconds: Number.POSITIVE_INFINITY };
      let baseline = 0;
      for (const candidate of candidates) {
        if (candidate.requiresF16 === true && !device.features.has("shader-f16" as GPUFeatureName)) continue;
        if (candidate.requiresSubgroupMatrix !== undefined && !matrixOffered(candidate)) continue;
        if (!aligned(candidate)) {
          lines.push(`  ${candidate.name} skipped: needs M, N and K divisible by ${candidate.alignment}`);
          continue;
        }
        device.pushErrorScope("validation");
        const pipeline = device.createComputePipeline({
          label: candidate.name, layout: "auto",
          compute: { module: device.createShaderModule({ code: candidate.shader }), entryPoint: "main" },
        });
        const failure = await device.popErrorScope();
        if (failure !== null) { lines.push(`  ${candidate.name} rejected: ${failure.message.split("\n")[0]}`); continue; }
        const group = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [candidate.sourceFormat === "f16" ? sourceHalf : source,
            candidate.weightFormat === "f16" ? weightsHalf : weights, params,
            candidate.outputFormat === "f16" ? outputHalf : output]
            .map((buffer, binding) => ({ binding, resource: { buffer } })),
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
        const failed = wrong.has(candidate.name);
        if (candidate.name === "production") baseline = milliseconds;
        if (!failed && milliseconds < best.milliseconds) best = { name: candidate.name, milliseconds };
        lines.push(`  ${candidate.name.padEnd(22)} ${milliseconds.toFixed(3)} ms  `
          + `${(flops / milliseconds / 1e9).toFixed(2)} TFLOP/s`
          + (failed ? "   (failed accuracy, not ranked)" : ""));
      }
      lines.push(`  -> fastest correct: ${best.name || "none"}`
        + (baseline > 0 && best.milliseconds < Number.POSITIVE_INFINITY
          ? ` (${(baseline / best.milliseconds).toFixed(2)}x over production)` : ""));
      for (const buffer of [source, weights, sourceHalf, weightsHalf, params, output, outputHalf]) {
        buffer.destroy();
      }
    }
    device.destroy();
    return lines;
  }, { candidates: CANDIDATES.map((c) => ({ ...c })), shapes: SHAPES.map((s) => ({ ...s })) });
  console.log(`\nGEMM CALIBRATION\n${report.join("\n")}\n`);
});
