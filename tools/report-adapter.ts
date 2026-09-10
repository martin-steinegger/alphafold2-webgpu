/**
 * Prints the adapter a run would get, and the features it offers.
 *
 * CI has no GPU on Linux and a virtualised one on macOS, so which adapter
 * answered decides what the tests after it mean. A run that quietly fell back
 * to a software device, or quietly failed to, should say so in its own log
 * rather than be inferred from a timing.
 *
 * It also opens a device and reports which kernel each operator settled on. A
 * gate that refuses a faster kernel says nothing on its own, and one wrong
 * feature name once turned off every matrix kernel on a card that has them.
 *
 * AFWEBGPU_BACKEND=wgpu reports the wgpu addon instead of Dawn.
 *
 * Usage: tsx tools/report-adapter.ts
 */
import { create, globals } from "webgpu";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { formatKernelReport, kernelSelectionReport, refusedFastPaths } from "../src/runtime/kernel-report.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { requestWgpuAdapter } from "../src/runtime/wgpu/adapter.js";
Object.assign(globalThis, globals);

// Held, not left as a temporary: dawn.node pumps its event loop from this
// object, and a collectable instance takes the device with it.
const gpu = process.env.AFWEBGPU_BACKEND === "wgpu"
  ? undefined : create(dawnInstanceFlags({ unclamped: true }));
const adapter = gpu === undefined
  ? requestWgpuAdapter()
  : await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) {
  console.error("no WebGPU adapter");
  process.exit(1);
}
const info = (adapter as unknown as { info?: Record<string, string> }).info ?? {};
for (const field of ["vendor", "architecture", "device", "description"]) {
  console.log(`${field.padEnd(13)} ${info[field] ?? "(unknown)"}`);
}
console.log(`features      ${[...adapter.features].sort().join(", ") || "(none)"}`);
// The featurisation kernels need these; the matrix units are a bonus and their
// absence is expected on a software adapter.
for (const feature of ["shader-f16", "subgroups", "subgroup-size-control"]) {
  console.log(`${feature.padEnd(13)} ${adapter.features.has(feature as GPUFeatureName) ? "yes" : "NO"}`);
}

const device = await requestAlphaFoldDevice(adapter);
const rows = await kernelSelectionReport(device);
console.log("");
console.log(formatKernelReport(rows));
const refused = refusedFastPaths(rows);
if (refused.length > 0) {
  console.log("");
  console.log(`${refused.length} faster kernel(s) refused on an adapter that reports the hardware.`);
}
device.destroy();
