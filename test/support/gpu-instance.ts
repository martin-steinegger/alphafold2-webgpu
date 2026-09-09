/**
 * The one place a test reaches for WebGPU.
 *
 * Dawn's toggles, or the adapter reports no shader-f16 and no matrix units and
 * a suite gates itself off the feature it exists to check. Held at module
 * scope, or dawn.node faults in ProcessEvents once the instance is collected.
 * Globals installed first.
 */
import { create, globals } from "webgpu";
import { dawnInstanceFlags, type DawnInstanceOptions } from "../../src/runtime/dawn.js";

const instances = new Map<string, GPU>();

/** The shared instance for these flags. AFWEBGPU_ADAPTER picks one of several. */
export function testGpu(options: DawnInstanceOptions = {}): GPU {
  const adapterName = process.env.AFWEBGPU_ADAPTER;
  const flags = [...dawnInstanceFlags(options),
    ...(adapterName === undefined ? [] : [`adapter=${adapterName}`])];
  const key = flags.join(" ");
  let instance = instances.get(key);
  if (instance === undefined) {
    Object.assign(globalThis, globals);
    instance = create(flags);
    instances.set(key, instance);
  }
  return instance;
}

export async function testAdapter(options: DawnInstanceOptions = {}): Promise<GPUAdapter> {
  const adapter = await testGpu(options).requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("no WebGPU adapter is available");
  return adapter;
}
