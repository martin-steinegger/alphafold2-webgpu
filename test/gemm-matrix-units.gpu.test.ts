/**
 * The projection over the hardware matrix units, against a CPU reference.
 *
 * The units are reached through a different kernel from every other variant —
 * one subgroup per workgroup, tiles addressed by subgroupMatrixLoad rather
 * than staged by hand — and on a device whose units take f16 operands it is
 * different again, staging both operands through workgroup memory because the
 * load reinterprets nothing. Neither path is exercised by the hand-tiled
 * differential, so it is checked here on whatever configuration the device
 * actually reports.
 *
 * AFWEBGPU_DAWN_FEATURES passes Dawn toggles through, which is what reaches
 * the units on a Vulkan device: Dawn hides both shader-f16 and the matrix
 * extension behind toggles on Nvidia.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createTiledGemmShader, gemmGrid } from "../src/runtime/gemm.js";
import {
  gemmVariantName, selectMatrixShape, type SubgroupMatrixConfig,
} from "../src/runtime/gemm-selection.js";
import { testGpu } from "./support/gpu-instance.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

/** The same deterministic values on every device, so a run is reproducible. */
function values(count: number, seed: number): Float32Array {
  let state = seed >>> 0;
  return Float32Array.from({ length: count }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000 - 0.5;
  });
}

describe.skipIf(!enabled)("projection over the hardware matrix units", () => {
  // The Dawn instance lives exactly as long as the object create returns, so
  // it is held for the suite rather than left to be collected once beforeAll
  // ends, which tore the device down under the test and aborted the worker.
  let gpu: GPU | undefined;
  let device: GPUDevice | undefined;
  let configs: readonly SubgroupMatrixConfig[] = [];

  beforeAll(async () => {
    // The same flags the model asks for: the matrix extension is
    // experimental, so Dawn hides it without allow_unsafe_apis.
    gpu = testGpu();
    const adapter = await gpu!.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    // The kernels pin their subgroup width, so the feature that lets them is
    // not optional for this suite.
    const wanted = ["chromium-experimental-subgroup-matrix", "subgroups",
      "subgroup-size-control", "shader-f16"] as const;
    const requiredFeatures = wanted.filter(
      (feature) => adapter.features.has(feature as GPUFeatureName),
    ) as GPUFeatureName[];
    if (!requiredFeatures.includes("chromium-experimental-subgroup-matrix" as GPUFeatureName)) {
      return;
    }
    configs = ((adapter.info as unknown as {
      subgroupMatrixConfigs?: readonly SubgroupMatrixConfig[];
    }).subgroupMatrixConfigs) ?? [];
    device = await adapter.requestDevice({
      requiredFeatures,
      // The f16 kernel stages both operands and the result, past the baseline.
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      },
    });
  });

  afterAll(() => { device?.destroy(); });

  // The staged kernel walks an operand along whichever index the caller says
  // runs contiguously, and both the fetch and the write into workgroup memory
  // have to agree about it. When they did not, every shape here still passed —
  // they all use the default layout — and the model came back at 26 pLDDT.
  for (const layout of [
    { name: "default layout", source: undefined, weight: undefined },
    { name: "row-contiguous source", source: "row" as const, weight: undefined },
    { name: "k-contiguous weight", source: undefined, weight: "k" as const },
    { name: "both transposed", source: "row" as const, weight: "k" as const },
  ]) {
    it(`reproduces a reference projection with a ${layout.name}`, async (context) => {
      if (device === undefined) {
        context.skip("this adapter has no usable subgroup matrix configuration");
        return;
      }
      const shape = selectMatrixShape(device, configs);
      if (shape === undefined) {
        context.skip("this adapter reports no configuration the kernel can walk");
        return;
      }
      const rows = 130;
      const inner = 96;
      const columns = 132;
      const source = values(rows * inner, 0x2244668);
      const weights = values(inner * columns, 0x1133557);
      // The operands are written transposed when the layout says they are, so
      // the expression the kernel reads matches what the hint claims.
      const sourceIndex = layout.source === "row" ? "k * 130u + row" : "row * 96u + k";
      const weightIndex = layout.weight === "k" ? "column * 96u + k" : "k * 132u + column";
      const code = createTiledGemmShader({
        preamble: `
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;`,
        rows: `${rows}u`, inner: `${inner}u`, columns: `${columns}u`,
        sourceElement: `source[${sourceIndex}]`,
        weightElement: `weights[${weightIndex}]`,
        store: `output[row * ${columns}u + column] = element;`,
        ...(layout.source === undefined ? {} : { sourceContiguous: layout.source }),
        ...(layout.weight === undefined ? {} : { weightContiguous: layout.weight }),
      }, { precision: "matrix", inner: 8, matrix: shape });

      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const created: GPUBuffer[] = [];
      const make = (data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer => {
        const buffer = device!.createBuffer({ size: data.byteLength, usage });
        device!.queue.writeBuffer(buffer, 0, data);
        created.push(buffer);
        return buffer;
      };
      try {
        const output = device.createBuffer({ size: rows * columns * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readback = device.createBuffer({ size: rows * columns * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        created.push(output, readback);
        const bound = [make(source, storage), make(weights, storage), output];
        device.pushErrorScope("validation");
        const pipeline = device.createComputePipeline({ layout: "auto",
          compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
        expect(await device.popErrorScope(), "compiles").toBeNull();
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bound.map((buffer, binding) => ({ binding, resource: { buffer } })),
        }));
        const [x, y] = gemmGrid(rows, columns);
        pass.dispatchWorkgroups(x, y, 1);
        pass.end();
        encoder.copyBufferToBuffer(output, 0, readback, 0, rows * columns * 4);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const actual = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        let worst = 0;
        let scale = 0;
        for (let row = 0; row < rows; row += 1) {
          for (let column = 0; column < columns; column += 1) {
            let total = 0;
            for (let k = 0; k < inner; k += 1) {
              const left = layout.source === "row" ? source[k * rows + row]! : source[row * inner + k]!;
              const right = layout.weight === "k" ? weights[column * inner + k]! : weights[k * columns + column]!;
              total += left * right;
            }
            worst = Math.max(worst, Math.abs(actual[row * columns + column]! - total));
            scale = Math.max(scale, Math.abs(total));
          }
        }
        expect(scale).toBeGreaterThan(0);
        expect(worst / scale, `${layout.name} reproduces the reference`).toBeLessThan(4e-3);
      } finally {
        for (const buffer of created) buffer.destroy();
      }
    }, 60_000);
  }

  it("reproduces a reference projection on the reported configuration", async (context) => {
    if (device === undefined) {
      context.skip("this adapter does not expose the subgroup matrix units");
      return;
    }
    const shape = selectMatrixShape(device, configs);
    if (shape === undefined) {
      context.skip("this adapter reports no configuration the kernel can walk");
      return;
    }
    // Rows past one 32-row region and not a multiple of it, so the pull-back
    // that keeps the last tile inside the source is exercised rather than
    // merely compiled; an inner dimension that is not a multiple of a 16-wide
    // unit checks the staged tail is zeroed rather than read from the next row.
    const rows = 81;
    const inner = 200;
    const columns = 128;
    const source = values(rows * inner, 0x1234567);
    const weights = values(inner * columns, 0x7654321);
    const code = createTiledGemmShader({
      preamble: `
struct Parameters { rows: u32, inner: u32, columns: u32, padding: u32 };
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<uniform> parameters: Parameters;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;`,
      rows: "parameters.rows",
      inner: "parameters.inner",
      columns: "parameters.columns",
      sourceElement: "source[row * parameters.inner + k]",
      weightElement: "weights[k * parameters.columns + column]",
      store: "output[row * parameters.columns + column] = element;",
      sourceArray: { array: "source", stride: "parameters.inner" },
      weightArray: { array: "weights", stride: "parameters.columns" },
    }, { precision: "matrix", inner: 8, matrix: shape });
    expect(code, "uses the units").toContain("subgroupMatrixMultiplyAccumulate");
    if (shape.componentType === "f16") {
      expect(code, "stages f16 operands").toContain("var<workgroup> gemm_matrix_a");
    }

    const label = gemmVariantName({ precision: "matrix", inner: 8, matrix: shape });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const sourceBuffer = device.createBuffer({ label, size: source.byteLength, usage: storage });
    const weightBuffer = device.createBuffer({ label, size: weights.byteLength, usage: storage });
    const parameters = device.createBuffer({
      label, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const output = device.createBuffer({
      label, size: rows * columns * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({
      label, size: rows * columns * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    try {
      device.queue.writeBuffer(sourceBuffer, 0, source);
      device.queue.writeBuffer(weightBuffer, 0, weights);
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
      const actual = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();

      let worst = 0;
      let scale = 0;
      for (let row = 0; row < rows; row += 1) {
        for (let column = 0; column < columns; column += 1) {
          let total = 0;
          for (let k = 0; k < inner; k += 1) {
            total += source[row * inner + k]! * weights[k * columns + column]!;
          }
          worst = Math.max(worst, Math.abs(actual[row * columns + column]! - total));
          scale = Math.max(scale, Math.abs(total));
        }
      }
      // f32 units accumulate the reference exactly; f16 operands round both
      // sides of every product, which is the same trade the half-precision
      // arrangements already make and is bounded the same way.
      const tolerance = shape.componentType === "f32" ? 1e-5 : 2e-3;
      expect(scale, "the reference is not degenerate").toBeGreaterThan(0);
      expect(worst / scale, `${label} reproduces the reference`).toBeLessThan(tolerance);
    } finally {
      for (const buffer of [sourceBuffer, weightBuffer, parameters, output, readback]) {
        buffer.destroy();
      }
    }
  }, 60_000);
});
