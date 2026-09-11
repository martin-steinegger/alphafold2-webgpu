/**
 * Records every compute pipeline the page builds in Safari, for a replay
 * through WebKit's own WGSL compiler (tools/safari/replay.sh).
 *
 * Safari compiles WGSL with rules Dawn and naga do not have, and it reports a
 * rejection only as "Compute library failed creation". So this runs the page's
 * monomer configuration on Dawn with the adapter reporting only what Safari
 * reports, and writes each shader module and each pipeline made from it, with
 * its override constants, in the order the page makes them. Dawn accepting
 * them says nothing about Safari; the replay is what does.
 *
 * What it covers:
 * - the page's own storages, transitions and row caps (508 and 1024);
 * - the calibration the device runs when it is made, since the recorder is
 *   installed before that;
 * - one projection variant a run. Safari calibrates its own and the choice is
 *   not known here, so CI records each one a device without matrix units can
 *   take;
 * - a second, shorter fold on the same device before the real one, as a tab
 *   that predicts again reuses its device and its shader modules.
 *
 * Usage: tsx tools/record-safari-pipelines.ts <file.a3m> <output directory>
 *          [--gemm <precision>:<inner>]
 * AFWEBGPU_MANIFEST names the model bundle (default .models/model/manifest.json).
 * Output: <hash>.wgsl and <hash>.pipelines for each module, where each line of
 * a .pipelines file is "<entry point> NAME=VALUE ...".
 */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { create, globals } from "webgpu";
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { parseA3m } from "../src/input/a3m.js";
import { iterateA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";
import { setGpuMemoryBudget } from "../src/runtime/allocator.js";
import { forceGemmVariant } from "../src/runtime/gemm-selection.js";
import type { GemmVariant } from "../src/runtime/gemm.js";
import { selectGpu } from "./native-device.js";
Object.assign(globalThis, globals);

// What Safari 26 on an Apple GPU reports, read from the page's own log.
const SAFARI_FEATURES = new Set(["core-features-and-limits", "shader-f16", "timestamp-query"]);
const SAFARI_LIMITS: Readonly<Record<string, number>> = {
  maxComputeInvocationsPerWorkgroup: 1024, maxComputeWorkgroupStorageSize: 32768,
  maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456,
  maxStorageBuffersPerShaderStage: 44,
};
// The page's defaults, and the memory budget it gives an Apple GPU.
const MSA_ROWS = 508;
const EXTRA_ROWS = 1024;
const MEMORY_BUDGET = 2867 * 1024 ** 2;
const STORAGE = { triangleWholeStorage: "f16", msaStorage: "f16", pairStorage: "f16" } as const;

const [file, output] = process.argv.slice(2).filter((arg, index, all) =>
  !arg.startsWith("--") && all[index - 1] !== "--gemm");
if (file === undefined || output === undefined) {
  throw new Error("usage: record-safari-pipelines.ts <file.a3m> <output directory> [--gemm <precision>:<inner>]");
}
const gemmArgument = process.argv[process.argv.indexOf("--gemm") + 1];
const gemm: GemmVariant | undefined = process.argv.includes("--gemm") && gemmArgument !== undefined
  ? { precision: gemmArgument.split(":")[0] as GemmVariant["precision"],
    inner: Number(gemmArgument.split(":")[1]) as GemmVariant["inner"] }
  : undefined;
mkdirSync(output, { recursive: true });

/** Writes each module, and each pipeline made from it, as the device makes them. */
function record(device: GPUDevice): void {
  const hashes = new WeakMap<GPUShaderModule, string>();
  const createShaderModule = device.createShaderModule.bind(device);
  device.createShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
    const hash = createHash("sha1").update(descriptor.code).digest("hex").slice(0, 12);
    const label = (descriptor.label ?? "unlabelled").replace(/[^A-Za-z0-9._:-]/g, "_");
    writeFileSync(`${output}/${hash}.wgsl`, `// ${label}\n${descriptor.code}`);
    const module = createShaderModule(descriptor);
    hashes.set(module, hash);
    return module;
  };
  const line = (descriptor: GPUComputePipelineDescriptor): void => {
    const hash = hashes.get(descriptor.compute.module);
    if (hash === undefined) return;
    const constants = Object.entries(descriptor.compute.constants ?? {}).map(([name, value]) => `${name}=${value}`);
    appendFileSync(`${output}/${hash}.pipelines`,
      `${[descriptor.compute.entryPoint ?? "main", ...constants].join(" ")}\n`);
  };
  const createAsync = device.createComputePipelineAsync.bind(device);
  device.createComputePipelineAsync = (descriptor: GPUComputePipelineDescriptor) => {
    line(descriptor);
    return createAsync(descriptor);
  };
  const createSync = device.createComputePipeline.bind(device);
  device.createComputePipeline = (descriptor: GPUComputePipelineDescriptor) => {
    line(descriptor);
    return createSync(descriptor);
  };
}

/** The real adapter, reporting only Safari's features and at most its limits. */
function safariAdapter(real: GPUAdapter): GPUAdapter {
  const features = new Set([...real.features].filter((name) => SAFARI_FEATURES.has(name)));
  const cap = (name: string, value: number): number =>
    name in SAFARI_LIMITS ? Math.min(value, SAFARI_LIMITS[name]!) : value;
  const limits = new Proxy(real.limits, {
    get: (target, key) => {
      const value = Reflect.get(target, key) as number;
      return typeof key === "string" ? cap(key, value) : value;
    },
  });
  // No subgroup matrix configurations, as Safari reports none.
  const info = { vendor: "apple", architecture: "", device: "", description: "apple GPU" };
  return new Proxy(real, {
    get: (target, key) => {
      if (key === "features") return features;
      if (key === "limits") return limits;
      if (key === "info") return info;
      if (key === "requestDevice") {
        return async (descriptor: GPUDeviceDescriptor = {}) => {
          const device = await target.requestDevice({
            ...descriptor,
            requiredFeatures: [...(descriptor.requiredFeatures ?? [])].filter((name) => features.has(name)),
            requiredLimits: Object.fromEntries(Object.entries(descriptor.requiredLimits ?? {})
              .map(([name, value]) => [name, cap(name, value as number)])),
          });
          // Before requestAlphaFoldDevice calibrates on it, so that is recorded too.
          record(device);
          return device;
        };
      }
      const value = Reflect.get(target, key);
      return typeof value === "function" ? value.bind(target) : value;
    },
  });
}

const a3m = readFileSync(file, "utf8");
const { length, depth, query } = parseA3m(a3m);
const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(
  process.env.AFWEBGPU_MANIFEST ?? ".models/model/manifest.json"));
const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables] =
  await Promise.all([model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(),
    model.mainStackWeights(), model.structureWeights(), model.confidenceWeights(),
    model.geometryTables(), model.queryOnlyFeatureTables()]);

// Honours CUDA_VISIBLE_DEVICES on a host with several cards; see selectGpu.
selectGpu();
// Held, not left as a temporary: dawn.node pumps its event loop from this object.
const gpu = create(dawnInstanceFlags({ unclamped: true }));
const real = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (real === null) throw new Error("no WebGPU adapter");
const adapter = safariAdapter(real);
const clustered = Math.min(MSA_ROWS, depth);
const extra = Math.max(1, Math.min(EXTRA_ROWS, Math.max(0, depth - clustered)));
const plan = planMonomerDevice(adapter, length, clustered, extra, MEMORY_BUDGET, true, STORAGE);
const device = await requestAlphaFoldDevice(adapter, plan.requirements);
setGpuMemoryBudget(device, MEMORY_BUDGET);
if (gemm !== undefined) forceGemmVariant(gemm);
console.error(`recording to ${output}: ${[...device.features].sort().join(",")}, `
  + `workgroup storage ${device.limits.maxComputeWorkgroupStorageSize}, projection `
  + `${gemm === undefined ? "as calibrated" : `${gemm.precision}:${gemm.inner}`}`);

// A shorter single sequence first, then the alignment itself.
const shorter = `>shorter\n${query.slice(0, Math.max(16, Math.floor(query.length * 2 / 3)))}\n`;
for (const input of [shorter, a3m]) {
  const folded = await new AlphaFoldMonomerGpu(device, {
    compactTransitions: plan.transitionMode === "chunked", ...STORAGE,
  }).predict(iterateA3mFeatures(device, input, featureTables,
    { recycles: 0, randomSeed: 0, maxMsaSequences: MSA_ROWS, maxExtraSequences: EXTRA_ROWS }),
  { embedding, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry, template });
  console.error(`folded ${parseA3m(input).length} residues: pLDDT ${folded.recycles.at(-1)?.confidence.meanPlddt.toFixed(2)}`);
}
device.destroy();
process.exit(0);
