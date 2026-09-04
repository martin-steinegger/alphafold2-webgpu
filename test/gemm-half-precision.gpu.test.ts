/**
 * Differential test for the half-precision projection.
 *
 * The shared GEMM can compute in f16 on a device that offers `shader-f16`,
 * and the model reduces over as many as 1024 terms, so the accumulator is
 * where half precision is most exposed. Every variant is checked against a
 * reference summed here in double precision, at both contraction depths the
 * model runs and through both epilogue paths, because the whole reason the
 * f16 kernel needs no caller change is that the accumulator arrives at an
 * epilogue as `vec4<f32>` — an invariant only a compiled shader can confirm.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  createTiledGemmShader, gemmGrid, GEMM_TILE_ROWS, type GemmVariant,
} from "../src/runtime/gemm.js";
import { gemmVariantName } from "../src/runtime/gemm-selection.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

/**
 * Every arrangement the generator can emit, including pure `f16`, which the
 * model does not ship. The kernel still has to compute the right answer at
 * these depths; what disqualified it was a contraction over a deep MSA, which
 * is a property of the model's magnitudes rather than of the kernel.
 */
const VARIANTS: readonly GemmVariant[] = ([8, 16] as const).flatMap((inner) =>
  (["f32", "f16", "f16-mixed", "f16-chunked"] as const).map((precision) => ({ precision, inner })));

/**
 * What each arrangement may be wrong by, from the measured worst case at that
 * depth. These bound the kernel, not the model: what half precision costs a
 * prediction is settled by `test/browser/gemm-differential.spec.ts`.
 */
function tolerance(variant: GemmVariant, inner: number): number {
  if (variant.precision === "f32") return 1e-4;
  // Pure f16 reduces the whole of K in half precision, so its error grows
  // with the square root of K; the other two reduce in f32 and do not.
  if (variant.precision === "f16") return inner > 512 ? 1.5e-2 : 5e-3;
  return 2e-3;
}

/** Values with a spread the model's activations have, and reproducible. */
function values(count: number, seed: number): Float32Array {
  let state = seed >>> 0;
  return Float32Array.from({ length: count }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000 - 0.5;
  });
}

/** A x W + bias, summed independently of any shader. */
function reference(
  source: Float32Array, weights: Float32Array, bias: Float32Array,
  rows: number, inner: number, columns: number,
): Float64Array {
  const output = new Float64Array(rows * columns);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      let total = 0;
      for (let k = 0; k < inner; k += 1) {
        total += source[row * inner + k]! * weights[k * columns + column]!;
      }
      output[row * columns + column] = total + bias[column]!;
    }
  }
  return output;
}

const PREAMBLE = `
struct Parameters { rows: u32, inner: u32, columns: u32, padding: u32 };
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;`;

/** Bias in the per-element store, the path most projections take. */
function scalarStoreShader(variant: GemmVariant): string {
  return createTiledGemmShader({
    preamble: PREAMBLE,
    rows: "parameters.rows", inner: "parameters.inner", columns: "parameters.columns",
    sourceElement: "source[row * parameters.inner + k]",
    weightElement: "weights[k * parameters.columns + column]",
    store: `output[row * parameters.columns + column] = element
          + weights[parameters.inner * parameters.columns + column];`,
  }, variant);
}

/**
 * The same result through a whole-tile epilogue, which reads `acc{n}` itself.
 *
 * This is the contract that lets half precision stay invisible to callers: an
 * epilogue written against `vec4<f32>` has to keep compiling and keep
 * computing the same thing when the k loop accumulates in f16.
 */
function epilogueShader(variant: GemmVariant): string {
  const rowsPerThread = 8;
  return createTiledGemmShader({
    preamble: PREAMBLE,
    rows: "parameters.rows", inner: "parameters.inner", columns: "parameters.columns",
    sourceElement: "source[row * parameters.inner + k]",
    weightElement: "weights[k * parameters.columns + column]",
    // Required by the interface and replaced by the epilogue below.
    store: "output[row * parameters.columns + column] = element;",
    epilogue: Array.from({ length: rowsPerThread }, (_, index) => `
  {
    let row = tile_row_origin + row_thread * ${rowsPerThread}u + ${index}u;
    if (row < gemm_rows) {
      for (var lane = 0u; lane < 4u; lane += 1u) {
        let column = column_origin + column_thread * 4u + lane;
        if (column < gemm_columns) {
          output[row * gemm_columns + column] = acc${index}[lane]
            + weights[parameters.inner * parameters.columns + column];
        }
      }
    }
  }`).join("\n"),
    stageElements: 4,
  }, variant);
}

describe.skipIf(!enabled)("half-precision projection", () => {
  let gpu: GPU;
  let device: GPUDevice;
  let half = false;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.AFWEBGPU_ADAPTER;
    gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    half = adapter.features.has("shader-f16");
    device = await adapter.requestDevice({
      requiredFeatures: half ? (["shader-f16"] as GPUFeatureName[]) : [],
    });
  });

  afterAll(() => { device?.destroy(); });

  /** Runs one shader and returns what it wrote. */
  async function run(
    code: string, label: string, rows: number, inner: number, columns: number,
    source: Float32Array, weights: Float32Array, bias: Float32Array,
  ): Promise<Float32Array> {
    const weightsAndBias = new Float32Array(weights.length + bias.length);
    weightsAndBias.set(weights);
    weightsAndBias.set(bias, weights.length);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const sourceBuffer = device.createBuffer({ label, size: source.byteLength, usage: storage });
    const weightBuffer = device.createBuffer({
      label, size: weightsAndBias.byteLength, usage: storage });
    const parameters = device.createBuffer({
      label, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({
      label, size: rows * columns * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({
      label, size: rows * columns * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(sourceBuffer, 0, source);
    device.queue.writeBuffer(weightBuffer, 0, weightsAndBias);
    device.queue.writeBuffer(parameters, 0, new Uint32Array([rows, inner, columns, 0]));
    device.pushErrorScope("validation");
    const pipeline = device.createComputePipeline({
      label, layout: "auto",
      compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
    });
    const failure = await device.popErrorScope();
    expect(failure, `${label} compiles`).toBeNull();
    const encoder = device.createCommandEncoder({ label });
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [sourceBuffer, weightBuffer, parameters, output]
        .map((buffer, binding) => ({ binding, resource: { buffer } })),
    }));
    const [x, y] = gemmGrid(rows, columns);
    pass.dispatchWorkgroups(x, y, 1);
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, rows * columns * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    for (const buffer of [sourceBuffer, weightBuffer, parameters, output, readback]) {
      buffer.destroy();
    }
    return result;
  }

  /** Worst absolute error against the reference, relative to its largest value. */
  function worstRelativeError(actual: Float32Array, expected: Float64Array): number {
    let worst = 0;
    let scale = 0;
    for (let index = 0; index < expected.length; index += 1) {
      worst = Math.max(worst, Math.abs(actual[index]! - expected[index]!));
      scale = Math.max(scale, Math.abs(expected[index]!));
    }
    return worst / scale;
  }

  // Two depths, because a half-precision accumulator's error grows with the
  // square root of the number of terms: a check at 256 says nothing about the
  // transitions, which reduce over 1024.
  for (const inner of [256, 1024]) {
    // Not a round multiple of the tile, so the bounds checks are exercised.
    const rows = GEMM_TILE_ROWS + 17;
    const columns = 132;
    for (const variant of VARIANTS) {
      const name = gemmVariantName(variant);
      it(`computes A x W + bias with ${name} at K=${inner}`, async (context) => {
        // Skipped rather than silently passed: a vacuous green here would be
        // indistinguishable from a verified half-precision kernel.
        if (variant.precision !== "f32" && !half) context.skip();
        const source = values(rows * inner, 0x1234567);
        const weights = values(inner * columns, 0x89abcdef);
        const bias = values(columns, 0x2468ace);
        const expected = reference(source, weights, bias, rows, inner, columns);
        const actual = await run(scalarStoreShader(variant), `store.${name}.k${inner}`,
          rows, inner, columns, source, weights, bias);
        // f32 is exact to rounding; f16 rounds every operand and partial sum,
        // and 1% at K=1024 is what the measured worst case leaves room for.
        // These bound the kernel, not the model: what half precision costs a
        // prediction was settled end to end.
        expect(worstRelativeError(actual, expected)).toBeLessThan(tolerance(variant, inner));
      });

      it(`reaches an epilogue as vec4<f32> with ${name} at K=${inner}`, async (context) => {
        if (variant.precision !== "f32" && !half) context.skip();
        const source = values(rows * inner, 0x13579bd);
        const weights = values(inner * columns, 0x2468ace);
        const bias = values(columns, 0x1234567);
        const expected = reference(source, weights, bias, rows, inner, columns);
        const actual = await run(epilogueShader(variant), `epilogue.${name}.k${inner}`,
          rows, inner, columns, source, weights, bias);
        expect(worstRelativeError(actual, expected)).toBeLessThan(tolerance(variant, inner));
      });
    }

    it(`agrees between f16 and f32 to the model's packed-storage error at K=${inner}`,
      async (context) => {
        if (!half) context.skip();
        const rows = GEMM_TILE_ROWS + 17;
        const columns = 132;
        const source = values(rows * inner, 0x55aa55a);
        const weights = values(inner * columns, 0x0f0f0f0);
        const bias = values(columns, 0x3141592);
        const exact = await run(scalarStoreShader({ precision: "f32", inner: 8 }),
          `agree.f32.k${inner}`, rows, inner, columns, source, weights, bias);
        const approximate = await run(scalarStoreShader({ precision: "f16", inner: 8 }),
          `agree.f16.k${inner}`, rows, inner, columns, source, weights, bias);
        let worst = 0;
        let scale = 0;
        for (let index = 0; index < exact.length; index += 1) {
          worst = Math.max(worst, Math.abs(exact[index]! - approximate[index]!));
          scale = Math.max(scale, Math.abs(exact[index]!));
        }
        expect(worst / scale).toBeLessThan(inner > 512 ? 1.5e-2 : 5e-3);
      });
  }
});
