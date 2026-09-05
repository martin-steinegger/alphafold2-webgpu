/**
 * What the register flash kernel is actually waiting for.
 *
 * At 708 residues, triangle attention is 57% of an Evoformer block and rising
 * with length; it is the largest single cost in a long prediction. It achieves
 * about 500 GFLOP/s where the projection kernel on the same device reaches
 * 2,800, so it is not compute-bound and making the arithmetic half precision
 * would buy nothing on its own. The question is which memory access it is
 * waiting on, and there are two candidates.
 *
 * Every lane in a workgroup reads the same key and value for a given key
 * index, so those are sixteen vector loads shared by sixty-four lanes. But the
 * pair bias is indexed `(head * queries + query) * queries + key`, so for one
 * key index the lanes read addresses `queries * 4` bytes apart — sixty-four
 * separate cache lines per key, none of them coalesced. Transposing that
 * layout to `(head * queries + key) * queries + query` makes the same reads
 * contiguous.
 *
 * This measures the kernel with the operands it really has, at a real shape,
 * against three changes: half-precision keys and values, the transposed bias,
 * and both. It measures only; correctness is a separate question and a
 * separate test.
 */
import { expect, test } from "@playwright/test";
import { createAttentionRegisterFlashShader } from "../../src/evoformer/attention.js";

const enabled = process.env.AFWEBGPU_ATTENTION_BENCH === "1";

/** Triangle attention's shape, with the batch cut so the buffers fit. */
const QUERIES = Number(process.env.AFWEBGPU_BENCH_QUERIES ?? "708");
const BATCH = Number(process.env.AFWEBGPU_BENCH_BATCH ?? "64");
const HEADS = 4;
const HEAD_DIM = 32;

/**
 * The shipped kernel, and the same kernel with one thing changed at a time.
 *
 * These are textual edits of the generated shader rather than new generators,
 * because the point is to find out which change is worth building properly,
 * not to build all three.
 */
function variants(): readonly {
  readonly name: string; readonly code: string; readonly halfKv: boolean; readonly slots: number;
}[] {
  const base = createAttentionRegisterFlashShader(HEAD_DIM, 1);
  // Keys and values as packed half words: half the bytes, unpacked on load.
  const halfKv = (code: string): string => code
    .replace("@group(0) @binding(1) var<storage, read> key: array<vec4<f32>>;",
      "@group(0) @binding(1) var<storage, read> key: array<vec2<u32>>;")
    .replace("@group(0) @binding(2) var<storage, read> value: array<vec4<f32>>;",
      "@group(0) @binding(2) var<storage, read> value: array<vec2<u32>>;")
    .replace(/let kv(\d+) = key\[k_base \+ (\d+)u\];/gu,
      "let kw$1 = key[k_base + $2u];\n    let kv$1 = vec4<f32>("
      + "unpack2x16float(kw$1.x), unpack2x16float(kw$1.y));")
    .replace(/let vv(\d+) = value\[k_base \+ (\d+)u\];/gu,
      "let vw$1 = value[k_base + $2u];\n    let vv$1 = vec4<f32>("
      + "unpack2x16float(vw$1.x), unpack2x16float(vw$1.y));");
  // The bias indexed so that neighbouring lanes read neighbouring words.
  const transposedBias = (code: string): string => code.replace(
    /pair_bias\[\(head \* p\.queries \+ select\(0u, q_index_(\d+), live_\1\)\) \* p\.queries \+ k_index\]/gu,
    "pair_bias[(head * p.queries + k_index) * p.queries + select(0u, q_index_$1, live_$1)]");
  // More queries per invocation amortises each key and value load across more
  // of them. The kernel already supports one, two and four; the calibration
  // only ever offers one, so whether the others are faster has never been
  // measured on this device.
  const out: { name: string; code: string; halfKv: boolean; slots: number }[] = [];
  for (const slots of [1, 2, 4]) {
    const kernel = createAttentionRegisterFlashShader(HEAD_DIM, slots);
    out.push({ name: `${slots}q f32 kv`, code: kernel, halfKv: false, slots });
    out.push({ name: `${slots}q f16 kv`, code: halfKv(kernel), halfKv: true, slots });
  }
  out.push({ name: "1q transposed bias", code: transposedBias(base), halfKv: false, slots: 1 });
  return out;
}

test.skip(!enabled, "set AFWEBGPU_ATTENTION_BENCH=1");
test("times the register flash kernel against its two suspected bottlenecks", async ({ page }) => {
  test.setTimeout(30 * 60_000);
  page.on("console", (message) => console.log(message.text()));
  await page.goto("/");

  const report = await page.evaluate(async ({ candidates, queries, batch, heads, headDim }) => {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) return ["no adapter"];
    const device = await adapter.requestDevice({
      requiredLimits: { maxStorageBufferBindingSize: Math.min(
        adapter.limits.maxStorageBufferBindingSize, 1024 * 1024 * 1024),
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1024 * 1024 * 1024) },
    });
    const elements = batch * queries * heads * headDim;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const make = (bytes: number): GPUBuffer => device.createBuffer({ size: bytes, usage: storage });
    const full = make(elements * 4);
    const half = make(elements * 2);
    const gate = make(elements * 4);
    const output = device.createBuffer({
      size: elements * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const mask = make(batch * queries * 4);
    const bias = make(heads * queries * queries * 4);
    // Eighteen u32 and a vec2<u32> of padding: eighty bytes, not sixty-four.
    const parameters = device.createBuffer({
      size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // The struct is batch, queries, channels, heads, head_dim, transpose,
    // has_pair_bias, then the weight offsets, then batch_offset, batch_total.
    const fields = new Uint32Array(20);
    fields[0] = batch; fields[1] = queries; fields[2] = heads * headDim; fields[3] = heads;
    fields[4] = headDim; fields[5] = 0; fields[6] = 1;
    fields[16] = 0; fields[17] = batch;
    device.queue.writeBuffer(parameters, 0, fields);
    const lines: string[] = [];
    const timings: Record<string, number> = {};
    const readback = device.createBuffer({
      size: 256, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    device.addEventListener("uncapturederror", (event) => {
      lines.push(`uncaptured: ${String((event as GPUUncapturedErrorEvent).error.message).split("\n")[0]}`);
    });
    // Non-zero operands, or a result of zero says nothing about whether it ran.
    const seed = new Float32Array(1 << 16);
    for (let index = 0; index < seed.length; index += 1) {
      seed[index] = ((index * 2654435761) % 1000) / 1000 - 0.5;
    }
    for (let offset = 0; offset + seed.byteLength <= elements * 4; offset += seed.byteLength) {
      device.queue.writeBuffer(full, offset, seed);
      device.queue.writeBuffer(gate, offset, seed);
      if (offset + seed.byteLength <= elements * 2) device.queue.writeBuffer(half, offset, seed);
    }
    device.queue.writeBuffer(mask, 0, new Float32Array(batch * queries).fill(1));
    for (let offset = 0; offset + seed.byteLength <= heads * queries * queries * 4;
      offset += seed.byteLength) device.queue.writeBuffer(bias, offset, seed);
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
        entries: [full, candidate.halfKv ? half : full, candidate.halfKv ? half : full,
          gate, mask, bias, parameters, output]
          .map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const run = async (): Promise<number> => {
        device.pushErrorScope("validation");
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(queries / (64 * candidate.slots)), batch, heads);
        pass.end();
        const started = performance.now();
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        const elapsed = performance.now() - started;
        const dispatchFailure = await device.popErrorScope();
        if (dispatchFailure !== null) {
          lines.push(`${candidate.name}: dispatch ${dispatchFailure.message.split("\n")[0]}`);
        }
        return elapsed;
      };
      await run();
      // Did it write anything? A kernel that returns immediately looks fast.
      {
        const encoder = device.createCommandEncoder();
        encoder.copyBufferToBuffer(output, 0, readback, 0, 256);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(GPUMapMode.READ);
        const written = new Float32Array(readback.getMappedRange().slice(0));
        readback.unmap();
        const nonZero = written.filter((value) => value !== 0).length;
        lines.push(`  ${candidate.name}: ${nonZero} of 64 sampled outputs are non-zero`);
      }
      let best = Number.POSITIVE_INFINITY;
      for (let repeat = 0; repeat < 3; repeat += 1) best = Math.min(best, await run());
      timings[candidate.name] = best;
      const flops = 2 * 2 * batch * queries * queries * heads * headDim;
      lines.push(`${candidate.name.padEnd(16)} ${best.toFixed(1).padStart(8)} ms  `
        + `${(flops / (best / 1000) / 1e9).toFixed(0).padStart(5)} GFLOP/s  `
        + `${(timings["1q f32 kv"] === undefined ? 1 : timings["1q f32 kv"] / best).toFixed(2)}x`);
    }
    device.destroy();
    return lines;
  }, { candidates: variants().map((v) => ({ ...v })), queries: QUERIES, batch: BATCH, heads: HEADS, headDim: HEAD_DIM });

  console.log(`\nATTENTION MICROBENCHMARK (batch=${BATCH} queries=${QUERIES} `
    + `heads=${HEADS} headDim=${HEAD_DIM})\n${report.join("\n")}\n`);
  expect(report.length).toBeGreaterThan(0);
});
