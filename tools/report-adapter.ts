/**
 * Prints the adapter a run would get, and the features it offers.
 *
 * CI has no GPU on Linux and a virtualised one on macOS, so which adapter
 * answered decides what the tests after it mean. A run that quietly fell back
 * to a software device, or quietly failed to, should say so in its own log
 * rather than be inferred from a timing.
 *
 * Usage: tsx tools/report-adapter.ts
 */
import { create, globals } from "webgpu";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
Object.assign(globalThis, globals);

const gpu = create(dawnInstanceFlags({ unclamped: true }));
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
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
