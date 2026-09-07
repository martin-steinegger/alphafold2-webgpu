/**
 * The matrix-unit flash attention against the register kernel it competes with.
 *
 * The register kernel is the reference here rather than a captured tensor: it
 * is what the model ships and what every other differential in this suite has
 * already held to AlphaFold's own intermediates, so reproducing it is the
 * claim that matters. The tolerance is half precision, which is what the
 * matrix units consume — the accumulator stays f32, so the error is the
 * operands' rounding and not a drifting reduction.
 *
 * Lengths that are not multiples of the tiles are the point of the unaligned
 * cases: the kernel takes a bounds-checked path only for the last pass of keys
 * and the last block of queries, and that path is the one a whole-length test
 * never reaches.
 */
import { beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  createAttentionRegisterFlashShader, selectAttentionFlashKernel, supportsAttentionMatrix,
} from "../src/evoformer/attention.js";
import { recordSubgroupMatrixConfigs } from "../src/runtime/subgroups.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const HEAD_DIM = 32;
const HEADS = 4;

/** The same deterministic values on every device, so a run is reproducible. */
function values(count: number, seed: number): Float32Array {
  let state = seed >>> 0;
  return Float32Array.from({ length: count }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 0x100000000 - 0.5;
  });
}

describe.skipIf(!enabled)("flash attention over the matrix units", () => {
  // Dawn's instance lives as long as the object `create` returns, so it is
  // held for the suite rather than collected when `beforeAll` ends.
  let gpu: ReturnType<typeof create> | undefined;
  let device: GPUDevice | undefined;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.AFWEBGPU_ADAPTER;
    const toggles = process.env.AFWEBGPU_DAWN_FEATURES;
    gpu = create([
      ...(adapterName === undefined ? [] : [`adapter=${adapterName}`]),
      ...(toggles === undefined || toggles === "" ? [] : [`enable-dawn-features=${toggles}`]),
    ]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    const wanted = ["subgroups", "subgroup-size-control", "shader-f16",
      "chromium-experimental-subgroup-matrix"] as const;
    const requiredFeatures = wanted.filter(
      (feature) => adapter.features.has(feature as GPUFeatureName),
    ) as GPUFeatureName[];
    const candidate = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    recordSubgroupMatrixConfigs(candidate, adapter);
    device = supportsAttentionMatrix(candidate, HEAD_DIM) ? candidate : undefined;
  });

  for (const [batch, queries] of [[8, 128], [8, 130], [6, 97]] as const) {
    it(`reproduces the register kernel at ${batch}x${queries}`, async (context) => {
      if (device === undefined) {
        context.skip("this adapter has no usable subgroup matrix configuration");
        return;
      }
      const elements = batch * queries * HEADS * HEAD_DIM;
      const query = values(elements, 0x1234567);
      const key = values(elements, 0x7654321);
      const value = values(elements, 0x2468ace);
      const gate = values(elements, 0x1357bdf);
      const mask = Float32Array.from({ length: batch * queries },
        // A masked key in every row, so the softmax has to exclude it.
        (_, index) => (index % 17 === 0 ? 0 : 1));
      const bias = values(HEADS * queries * queries, 0xfedcba9);
      const parameters = new Uint32Array(20);
      parameters.set([batch, queries, HEADS * HEAD_DIM, HEADS, HEAD_DIM, 0, 1], 0);
      parameters[17] = batch;

      const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
      const created: GPUBuffer[] = [];
      const make = (data: Float32Array | Uint32Array, usage: GPUBufferUsageFlags): GPUBuffer => {
        const buffer = device!.createBuffer({ size: data.byteLength, usage });
        device!.queue.writeBuffer(buffer, 0, data);
        created.push(buffer);
        return buffer;
      };
      try {
        const bound = [make(query, storage), make(key, storage), make(value, storage),
          make(gate, storage), make(mask, storage), make(bias, storage),
          make(parameters, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)];
        const output = device.createBuffer({ size: elements * 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        const readback = device.createBuffer({ size: elements * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        created.push(output, readback);

        const run = async (
          label: string, code: string, tile: number, batchFirst = false,
        ): Promise<Float32Array> => {
          // Cleared between kernels. Sharing it, a kernel that writes only part
          // of the output leaves the previous kernel's answer in the rest, and
          // the comparison passes on the strength of it: dispatched in the
          // wrong order the matrix kernel covered half the batch and still
          // matched, because the other half was the register kernel's own
          // output read back again.
          device!.queue.writeBuffer(output, 0, new Float32Array(elements));
          device!.pushErrorScope("validation");
          const pipeline = device!.createComputePipeline({ label, layout: "auto",
            compute: { module: device!.createShaderModule({ label, code }), entryPoint: "main" } });
          const failure = await device!.popErrorScope();
          expect(failure, `${label} compiles`).toBeNull();
          const encoder = device!.createCommandEncoder({ label });
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, device!.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [...bound, output].map((buffer, binding) => ({ binding, resource: { buffer } })),
          }));
          const blocks = Math.ceil(queries / tile);
          pass.dispatchWorkgroups(batchFirst ? batch : blocks, batchFirst ? blocks : batch, HEADS);
          pass.end();
          encoder.copyBufferToBuffer(output, 0, readback, 0, elements * 4);
          device!.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ);
          const held = new Float32Array(readback.getMappedRange().slice(0));
          readback.unmap();
          return held;
        };

        const expected = await run("register",
          createAttentionRegisterFlashShader(HEAD_DIM, 1, "f32"), 64);
        const kernel = selectAttentionFlashKernel(device, HEAD_DIM, "matrix");
        const actual = await run("matrix", kernel.shader, kernel.queryTile,
          kernel.batchFirst === true);

        let worst = 0;
        let scale = 0;
        for (let index = 0; index < expected.length; index += 1) {
          worst = Math.max(worst, Math.abs(expected[index]! - actual[index]!));
          scale = Math.max(scale, Math.abs(expected[index]!));
        }
        expect(scale, "the reference is not degenerate").toBeGreaterThan(0);
        // Half-precision operands against a single-precision reduction.
        expect(worst / scale, "matrix attention reproduces the register kernel")
          .toBeLessThan(4e-3);
      } finally {
        for (const buffer of created) buffer.destroy();
      }
    }, 60_000);
  }
});
