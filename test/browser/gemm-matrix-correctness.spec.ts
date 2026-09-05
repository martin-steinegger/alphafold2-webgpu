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

/**
 * The projection as a caller writes it when its output is packed half
 * precision: four columns a time, two to a word. The widest contraction in
 * the model takes this path, so the matrix kernel has to serve it.
 */
function packedSpec() {
  return {
    ...projectionSpec(false),
    preamble: PREAMBLE.replace("var<storage, read_write> output: array<f32>;",
      "var<storage, read_write> output: array<u32>;"),
    store: "output[row * parameters.columns + column] = u32(element);",
    storeVector: `let base = row * parameters.columns + column;
${[0, 2].map((pair) => `          if (column + ${pair + 1}u < parameters.columns) {
            let stored = vec2<f32>(values[${pair}] + weights[parameters.bias_offset + column + ${pair}u],
              values[${pair + 1}] + weights[parameters.bias_offset + column + ${pair + 1}u]);
            output[(base + ${pair}u) >> 1u] = pack2x16float(stored);
          }`).join("\n")}`,
  } as const;
}

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

  const matrix = { precision: "matrix", inner: 8 } as const;
  const shaders = {
    plain: createTiledGemmShader(projectionSpec(false), matrix),
    packed: createTiledGemmShader(packedSpec(), matrix),
  };

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
      const run = async (code: string, label: string, packed: boolean): Promise<Float32Array> => {
        const storage = 128 | 8;
        const sourceBuffer = device.createBuffer({ size: source.byteLength, usage: storage });
        const weightBuffer = device.createBuffer({ size: weights.byteLength, usage: storage });
        const params = device.createBuffer({ size: 32, usage: 64 | 8 });
        const bytes = packed ? Math.ceil(rows * columns / 2) * 4 : rows * columns * 4;
        const output = device.createBuffer({ size: bytes, usage: 128 | 4 });
        const readback = device.createBuffer({ size: bytes, usage: 1 | 8 });
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
        encoder.copyBufferToBuffer(output, 0, readback, 0, bytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(1);
        const raw = readback.getMappedRange().slice(0);
        // A packed output holds two half-precision values per word.
        const half = (bits: number): number => {
          const sign = (bits & 0x8000) !== 0 ? -1 : 1;
          const exponent = (bits >> 10) & 0x1f;
          const mantissa = bits & 0x3ff;
          if (exponent === 0) return sign * mantissa * 2 ** -24;
          if (exponent === 31) return mantissa === 0 ? sign * Infinity : Number.NaN;
          return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
        };
        const result = packed
          ? Float32Array.from({ length: rows * columns }, (_unused, index) => {
            const word = new Uint32Array(raw)[index >> 1]!;
            return half((index & 1) === 0 ? word & 0xffff : word >>> 16);
          })
          : new Float32Array(raw);
        readback.unmap();
        for (const buffer of [sourceBuffer, weightBuffer, params, output, readback]) buffer.destroy();
        return result;
      };
      const actual = await run(shaders.plain, `matrix.${rows}x${inner}x${columns}`, false);
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
      // The same shape with a packed half-precision output, which rounds to
      // about three digits, so it is checked against that rather than exactly.
      const packedActual = await run(shaders.packed, `packed.${rows}x${inner}x${columns}`, true);
      let packedWrong = 0;
      let packedWorst = 0;
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          let total = 0;
          for (let k = 0; k < inner; k += 1) {
            total += source[row * inner + k]! * weights[k * columns + column]!;
          }
          total += weights[inner * columns + column]!;
          const difference = Math.abs(packedActual[row * columns + column]! - total);
          if (difference > 2e-2 * Math.max(1, Math.abs(total))) packedWrong += 1;
          packedWorst = Math.max(packedWorst, difference);
        }
      }
      lines.push(`  packed output: worst ${packedWorst.toExponential(2)}, `
        + `${packedWrong} of ${rows * columns} wrong`);
    }
    device.destroy();
    return lines;
  }, { shapes: SHAPES.map((shape) => ({ ...shape })), shaders });

  console.log(`\nMATRIX CORRECTNESS\n${report.join("\n")}\n`);
  for (const line of report) {
    expect(line, "every shape is exact").not.toMatch(/[1-9]\d* of \d+ wrong/u);
  }
});
