import { create, globals } from "webgpu";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { forceGemmVariant } from "../src/runtime/gemm-selection.js";
import { GEMM_VARIANT_F32 } from "../src/runtime/gemm.js";
import { requestWgpuAdapter } from "../src/runtime/wgpu/adapter.js";
import { createDeterministicTriangleInput } from "../src/testing/deterministic-input.js";
import { TriangleMultiplicationOutgoingGpu } from "../src/triangle/webgpu.js";
import type { Precision } from "../src/triangle/types.js";

function option(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  return process.argv.find((argument) => argument.startsWith(prefix))?.slice(prefix.length) ?? fallback;
}

const lengths = option("lengths", "128,256,512").split(",").map(Number);
const cZ = Number(option("cz", "128"));
const cHidden = Number(option("hidden", "128"));
const precision = option("precision", "f16") as Precision;
// One run a length includes compiling that length's pipelines, which is tens
// of milliseconds against a kernel of tens more. Best of several, after a
// discarded first, times the kernel.
const repeats = Number(option("repeats", "4"));
if (!lengths.every((length) => Number.isSafeInteger(length) && length > 0)) {
  throw new Error("--lengths must be a comma-separated list of positive integers");
}
if (precision !== "f16" && precision !== "f32") throw new Error("--precision must be f16 or f32");

Object.assign(globalThis, globals);
const requestedAdapter = option("adapter", "");
// dawnInstanceFlags, not a bare instance: without its toggles the adapter
// reports no matrix units and every number here is off the wrong kernel.
// AFWEBGPU_BACKEND=wgpu measures the same kernels through the wgpu addon.
const gpu = process.env.AFWEBGPU_BACKEND === "wgpu" ? undefined
  : create(dawnInstanceFlags({ unclamped: true,
    ...(requestedAdapter === "" ? {} : { adapter: requestedAdapter }) }));
const adapter = gpu === undefined
  ? requestWgpuAdapter()
  : await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter is available");
const adapterName = adapter.info.description || adapter.info.device || adapter.info.vendor || "unknown";
console.log(`adapter=${adapterName}`);
if (precision === "f16" && !adapter.features.has("shader-f16")) {
  throw new Error("the selected adapter does not expose shader-f16; retry with --precision=f32");
}
// The whole adapter's feature set and workgroup limits, because the matrix
// kernels are gated on both and this is measuring which kernel wins. The
// buffer limit is asked for from the largest pair this run will build.
const largest = Math.max(...lengths) ** 2 * cZ * 4;
const device = await requestAlphaFoldDevice(adapter,
  { maxBufferSize: largest, maxStorageBufferBindingSize: largest });
// AFWEBGPU_GEMM pins the projection variant, so a comparison of anything else
// is not really a comparison of which projection the calibration picked.
const gemmPin = process.env.AFWEBGPU_GEMM;
if (gemmPin === "f32") forceGemmVariant(GEMM_VARIANT_F32);
const runner = new TriangleMultiplicationOutgoingGpu(device);

console.log(`precision=${precision} c_z=${cZ} c_hidden=${cHidden}`);
console.log("L\tbest_ms\tpeak_gpu_mib");
for (const length of lengths) {
  const input = createDeterministicTriangleInput({ length, cZ, cHidden }, 1000 + length);
  let best = Number.POSITIVE_INFINITY;
  let peakBytes = 0;
  for (let round = 0; round <= repeats; round += 1) {
    const result = await runner.run(input, { precision });
    if (round > 0) best = Math.min(best, result.elapsedMilliseconds);
    peakBytes = Math.max(peakBytes, result.memory.peakBytes);
  }
  console.log(`${length}\t${best.toFixed(3)}\t${(peakBytes / 2 ** 20).toFixed(2)}`);
}
device.destroy();
