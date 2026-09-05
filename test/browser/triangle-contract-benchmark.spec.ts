/**
 * Why the triangle contraction runs at a fraction of the projection's rate.
 *
 * `triangle.outgoing.contract` and its incoming twin are 14% of an Evoformer
 * block at 708 residues and rising with length — 6.7% at 236, 9.3% at 472 —
 * and the profile puts them at about 500 GFLOP/s where the projection kernel,
 * from the same generator on the same device, reaches 2,800.
 *
 * The suspicion is the access pattern rather than the arithmetic. The
 * contraction computes `out[i][j] = sum_k a[i][k] b[j][k]`, so its second
 * operand is addressed `column * L + k`: the staging loop walks columns at a
 * fixed k, and consecutive columns are a whole row apart. Nothing about that
 * coalesces.
 *
 * This times the same shape and the same volume with the operand addressed
 * both ways. Row-major is not the computation the model wants — it is the
 * control that says whether the pattern is what costs, before anything is
 * built to fix it.
 */
import { expect, test } from "@playwright/test";
import { createTiledGemmShader, gemmGrid } from "../../src/runtime/gemm.js";

const enabled = process.env.AFWEBGPU_TRIANGLE_BENCH === "1";

/** The contraction's shape: a block of pair rows against every residue. */
const LENGTH = Number(process.env.AFWEBGPU_TRIANGLE_LENGTH ?? "708");
const BLOCK_ROWS = Number(process.env.AFWEBGPU_TRIANGLE_ROWS ?? "64");
const CHANNELS = Number(process.env.AFWEBGPU_TRIANGLE_CHANNELS ?? "8");

function contractShader(transposedWeight: boolean, length: number, blockRows: number): string {
  const whole = length * length;
  return createTiledGemmShader({
    preamble: `
@group(0) @binding(0) var<storage, read> blocked: array<f32>;
@group(0) @binding(1) var<storage, read> whole: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;`,
    rows: `${blockRows}u`,
    inner: `${length}u`,
    columns: `${length}u`,
    sourceElement: `blocked[group.z * ${blockRows * length}u + row * ${length}u + k]`,
    // The shipped addressing, and the same volume read the other way.
    weightElement: transposedWeight
      ? `whole[group.z * ${whole}u + column * ${length}u + k]`
      : `whole[group.z * ${whole}u + k * ${length}u + column]`,
    store: `output[group.z * ${blockRows * length}u + row * ${length}u + column] = element;`,
  });
}

test.skip(!enabled, "set AFWEBGPU_TRIANGLE_BENCH=1");
test("times the triangle contraction against its operand's access pattern", async ({ page }) => {
  test.setTimeout(30 * 60_000);
  page.on("console", (message) => console.log(message.text()));
  await page.goto("/");

  const candidates = [
    { name: "shipped (b[j][k])", code: contractShader(true, LENGTH, BLOCK_ROWS) },
    { name: "row-major (b[k][j])", code: contractShader(false, LENGTH, BLOCK_ROWS) },
  ];

  const report = await page.evaluate(async ({ candidates, length, blockRows, channels, grid }) => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) return ["no adapter"];
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(
          adapter.limits.maxStorageBufferBindingSize, 1024 * 1024 * 1024),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024),
      },
    });
    const lines: string[] = [];
    device.addEventListener("uncapturederror", (event) => {
      lines.push(`uncaptured: ${String((event as GPUUncapturedErrorEvent).error.message).split("\n")[0]}`);
    });
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const blocked = device.createBuffer({ size: channels * blockRows * length * 4, usage: storage });
    const whole = device.createBuffer({ size: channels * length * length * 4, usage: storage });
    const output = device.createBuffer({
      size: channels * blockRows * length * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({
      size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // Non-zero operands, so a kernel that never ran cannot look fast.
    const seed = new Float32Array(1 << 16);
    for (let index = 0; index < seed.length; index += 1) {
      seed[index] = ((index * 2654435761) % 1000) / 1000 - 0.5;
    }
    for (let offset = 0; offset + seed.byteLength <= channels * blockRows * length * 4;
      offset += seed.byteLength) device.queue.writeBuffer(blocked, offset, seed);
    for (let offset = 0; offset + seed.byteLength <= channels * length * length * 4;
      offset += seed.byteLength) device.queue.writeBuffer(whole, offset, seed);

    const timings: Record<string, number> = {};
    for (const candidate of candidates) {
      device.pushErrorScope("validation");
      const pipeline = device.createComputePipeline({
        label: candidate.name, layout: "auto",
        compute: { module: device.createShaderModule({ code: candidate.code }), entryPoint: "main" },
      });
      const failure = await device.popErrorScope();
      if (failure !== null) {
        lines.push(`${candidate.name}: rejected ${failure.message.split("\n")[0]}`);
        continue;
      }
      const group = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [blocked, whole, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const run = async (): Promise<number> => {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(grid[0], grid[1], channels);
        pass.end();
        const started = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        return performance.now() - started;
      };
      await run();
      let best = Number.POSITIVE_INFINITY;
      for (let repeat = 0; repeat < 3; repeat += 1) best = Math.min(best, await run());
      timings[candidate.name] = best;
      {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(output, 0, readback, 0, 256);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const written = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        lines.push(`  ${candidate.name}: ${written.filter((v) => v !== 0).length} of 64 outputs non-zero`);
      }
      const flops = 2 * channels * blockRows * length * length;
      const shipped = timings["shipped (b[j][k])"];
      lines.push(`${candidate.name.padEnd(20)} ${best.toFixed(1).padStart(7)} ms  `
        + `${(flops / (best / 1000) / 1e9).toFixed(0).padStart(5)} GFLOP/s  `
        + `${(shipped === undefined ? 1 : shipped / best).toFixed(2)}x`);
    }
    device.destroy();
    return lines;
  }, {
    candidates, length: LENGTH, blockRows: BLOCK_ROWS, channels: CHANNELS,
    grid: gemmGrid(BLOCK_ROWS, LENGTH),
  });

  console.log(`\nTRIANGLE CONTRACTION (L=${LENGTH} rows=${BLOCK_ROWS} channels=${CHANNELS})\n`
    + `${report.join("\n")}\n`);
  expect(report.length).toBeGreaterThan(0);
});
