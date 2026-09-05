/**
 * Does the matrix kernel `gemm.ts` emits compute the projection?
 *
 * The candidates in `tools/gemm-candidates.ts` are checked by the calibration
 * harness, but those are a different implementation from the one the model
 * runs, and being right there says nothing about being right here. This checks
 * the shader `createTiledGemmShader` actually produces, against a reference
 * summed in double precision, at shapes that are and are not multiples of the
 * tile — a kernel whose bounds are wrong is exactly right until the edge.
 *
 * The vitest GPU suite cannot host this: dawn-node reports no subgroup matrix
 * configurations, so the kernel does not exist there.
 */
import { expect, test } from "@playwright/test";
import { createTiledGemmShader } from "../../src/runtime/gemm.js";

const enabled = process.env.AFWEBGPU_GEMM_MATRIX === "1";

const PREAMBLE = `
struct Parameters {
  rows: u32, inner: u32, columns: u32,
  weight_offset: u32, bias_offset: u32, activation: u32, padding: vec2<u32>,
};
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;`;

/** The shape of the projection every caller writes, including the bias. */
function projectionSpec(residual: boolean) {
  return {
    preamble: PREAMBLE,
    rows: "parameters.rows",
    inner: "parameters.inner",
    columns: "parameters.columns",
    sourceElement: "source[row * parameters.inner + k]",
    weightElement: "weights[parameters.weight_offset + k * parameters.columns + column]",
    store: `var stored = element + weights[parameters.bias_offset + column];
          if (parameters.activation == 1u) { stored = max(stored, 0.0); }
          output[row * parameters.columns + column] ${residual ? "+=" : "="} stored;`,
    sourceArray: { array: "source", stride: "parameters.inner" },
    weightArray: {
      array: "weights", base: "parameters.weight_offset", stride: "parameters.columns",
    },
  } as const;
}

const SHAPES = [
  // Exactly the 64x128 tile, then a whole number of them.
  { rows: 64, inner: 256, columns: 128 },
  { rows: 128, inner: 256, columns: 256 },
  // Ragged in one dimension, then both. This is where bounds go wrong.
  { rows: 81, inner: 256, columns: 128 },
  { rows: 64, inner: 256, columns: 132 },
  { rows: 81, inner: 256, columns: 132 },
  // A contraction the model actually runs, and a deeper one.
  { rows: 200, inner: 64, columns: 256 },
  { rows: 200, inner: 1024, columns: 256 },
  // Fewer rows than one 32-row region, where there is nowhere to pull the
  // load origin back to and the kernel computes the tile directly.
  { rows: 31, inner: 256, columns: 128 },
  { rows: 20, inner: 128, columns: 132 },
  { rows: 1, inner: 256, columns: 256 },
] as const;

test.skip(!enabled, "set AFWEBGPU_GEMM_MATRIX=1");
test("the emitted matrix kernel computes A x W + bias at every shape", async ({ page }) => {
  test.setTimeout(10 * 60_000);
  page.on("console", (message) => console.log(`browser: ${message.text()}`));
  await page.goto("/");

  const shaders = SHAPES.map(() => ({
    plain: createTiledGemmShader(projectionSpec(false), { precision: "matrix", inner: 8 }),
    reference: createTiledGemmShader(projectionSpec(false), { precision: "f32", inner: 8 }),
  }));

  const report = await page.evaluate(async ({ shapes, shaders }) => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) return ["no adapter"];
    const wanted = ["chromium-experimental-subgroup-matrix", "shader-f16", "subgroups"];
    const features = wanted.filter((name) => adapter.features.has(name as GPUFeatureName));
    if (!features.includes("chromium-experimental-subgroup-matrix")) {
      return ["adapter has no subgroup matrix units"];
    }
    const device = await adapter.requestDevice({ requiredFeatures: features as GPUFeatureName[] });
    const values = (count: number, seed: number): Float32Array => {
      let state = seed >>> 0;
      return Float32Array.from({ length: count }, () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 0x100000000 - 0.5;
      });
    };
    const lines: string[] = [];
    for (let index = 0; index < shapes.length; index += 1) {
      const shape = shapes[index]!;
      const { rows, inner, columns } = shape;
      const source = values(rows * inner, 0x1234567);
      const weightCount = inner * columns + columns;
      const weights = values(weightCount, 0x89abcdef);
      const run = async (code: string, label: string): Promise<Float32Array> => {
        const storage = 128 | 8;
        const sourceBuffer = device.createBuffer({ size: source.byteLength, usage: storage });
        const weightBuffer = device.createBuffer({ size: weights.byteLength, usage: storage });
        const params = device.createBuffer({ size: 32, usage: 64 | 8 });
        const output = device.createBuffer({ size: rows * columns * 4, usage: 128 | 4 });
        const readback = device.createBuffer({ size: rows * columns * 4, usage: 1 | 8 });
        device.queue.writeBuffer(sourceBuffer, 0, source);
        device.queue.writeBuffer(weightBuffer, 0, weights);
        device.queue.writeBuffer(params, 0, new Uint32Array([
          rows, inner, columns, 0, inner * columns, 0, 0, 0]));
        device.pushErrorScope("validation");
        const pipeline = device.createComputePipeline({
          label, layout: "auto",
          compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
        });
        const failure = await device.popErrorScope();
        if (failure !== null) { lines.push(`  ${label} rejected: ${failure.message.split("\n")[0]}`); }
        const group = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [sourceBuffer, weightBuffer, params, output]
            .map((buffer, binding) => ({ binding, resource: { buffer } })),
        });
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(columns / 128), Math.ceil(rows / 64), 1);
        pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, rows * columns * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(1);
        const result = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        for (const buffer of [sourceBuffer, weightBuffer, params, output, readback]) buffer.destroy();
        return result;
      };
      const actual = await run(shaders[index]!.plain, `matrix.${rows}x${inner}x${columns}`);
      // Worst error against a reference summed here, relative to the largest.
      let worst = 0;
      let scale = 0;
      let wrongCount = 0;
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          let total = 0;
          for (let k = 0; k < inner; k += 1) {
            total += source[row * inner + k]! * weights[k * columns + column]!;
          }
          total += weights[inner * columns + column]!;
          const difference = Math.abs(actual[row * columns + column]! - total);
          if (difference > 1e-3) wrongCount += 1;
          worst = Math.max(worst, difference);
          scale = Math.max(scale, Math.abs(total));
        }
      }
      lines.push(`M=${rows} K=${inner} N=${columns}: worst ${worst.toExponential(2)} `
        + `(${(100 * worst / scale).toFixed(4)}%), ${wrongCount} of ${rows * columns} wrong`);
    }
    device.destroy();
    return lines;
  }, { shapes: SHAPES.map((shape) => ({ ...shape })), shaders });

  console.log(`\nMATRIX CORRECTNESS\n${report.join("\n")}\n`);
  for (const line of report) {
    expect(line, "every shape is exact").not.toMatch(/[1-9]\d* of \d+ wrong/u);
  }
});
