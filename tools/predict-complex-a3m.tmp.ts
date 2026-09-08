/** Runs Multimer-v3 on a local ColabFold serialized complex A3M. */
import { readFileSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { AlphaFoldMultimerGpu } from "../src/model/multimer.js";
import { EXACT_STORAGE } from "../src/model/monomer.js";
import { iterateMultimerA3mFeatures } from "../src/input/multimer-features.js";
import { parseColabFoldComplexA3m } from "../src/input/colabfold-complex-a3m.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";
import { calibrateGemmVariant, gemmVariantName } from "../src/runtime/gemm-selection.js";
import { presetAttentionFlashKernel } from "../src/evoformer/attention-calibration.js";
import type { AttentionFlashVariant } from "../src/evoformer/attention.js";
import { forceAttentionKeyValueStorage, forceAttentionQueriesPerThread, type AttentionKeyValueStorage }
  from "../src/evoformer/attention.js";
import { calibrateAttentionShape } from "../src/runtime/attention-queries.js";
Object.assign(globalThis, globals);

const manifest = process.argv[2];
const file = process.argv[3];
if (manifest === undefined || file === undefined) {
  throw new Error("usage: predict-complex-a3m.tmp.ts <manifest.json> <complex.a3m> [recycles] [msaRows] [extraRows]");
}
const recycles = Number(process.argv[4] ?? "3");
const msaRows = Number(process.argv[5] ?? "252");
const extraRows = Number(process.argv[6] ?? "1152");

const complex = parseColabFoldComplexA3m(readFileSync(file, "utf8"));
if (complex === undefined) throw new Error("not a ColabFold serialized complex A3M");
const chains = complex.chains;
const length = chains.reduce((sum, chain) => sum + chain.length, 0);
console.error(`entities ${complex.uniqueSequences.map((s) => s.length).join(",")}`
  + ` x ${complex.cardinalities.join(",")} -> ${chains.length} chains, ${length} residues,`
  + ` ${complex.depth} assembled rows`);

const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(manifest));
const [embedding, multimerTemplate, extraStack, mainStack, structure, confidence, geometry, featureTables, paeBreaks] =
  await Promise.all([
    model.multimerEmbeddingWeights(), model.multimerTemplateWeights(), model.extraStackWeights(),
    model.mainStackWeights(), model.multimerStructureWeights(), model.confidenceWeights(), model.geometryTables(),
    model.queryOnlyFeatureTables(), model.tensor("confidencePaeBreaks"),
  ]);
// AFWEBGPU_DAWN_FEATURES passes Dawn toggles, e.g.
// vulkan_enable_f16_on_nvidia to expose shader-f16 on an NVIDIA Vulkan device.
const dawnFeatures = process.env.AFWEBGPU_DAWN_FEATURES;
const dawnDisabled = process.env.AFWEBGPU_DAWN_DISABLE;
const gpu = create([
  ...(dawnFeatures === undefined || dawnFeatures === "" ? [] : [`enable-dawn-features=${dawnFeatures}`]),
  ...(dawnDisabled === undefined || dawnDisabled === "" ? [] : [`disable-dawn-features=${dawnDisabled}`]),
]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
// AFWEBGPU_SCRATCH_SCALE trades memory for far fewer dispatches, for a run
// that owns a large accelerator rather than a browser's share of one.
const scratchScale = Number(process.env.AFWEBGPU_SCRATCH_SCALE ?? "1");
const plan = planMonomerDevice(adapter, length, 1, 1, undefined, false,
  { multimer: true, templateRows: multimerTemplate.templateRows,
    scratchBudgetScale: scratchScale,
    ...(process.env.AFWEBGPU_EXACT === "1" ? EXACT_STORAGE : {}) });
console.error(`scratch budget scale: ${scratchScale}x`);
console.error(`estimated peak ${(plan.memory.estimatedPeakBytes / 1024 ** 2).toFixed(0)} MiB`);
if (process.env.AFWEBGPU_PLAN_ONLY === "1") process.exit(0);

console.error(`adapter shader-f16=${adapter.features.has("shader-f16")}`
  + ` subgroups=${adapter.features.has("subgroups")}`);
const device = await requestAlphaFoldDevice(adapter, plan.requirements);
// requestAlphaFoldDevice already calibrated; the result is cached per device.
const chosenGemm = await calibrateGemmVariant(device);
console.error(`gemm variant: ${gemmVariantName(chosenGemm)}`);
// AFWEBGPU_ATTENTION pins the flash kernel, so the tiled variants the
// calibration never offers can be measured on the real shapes.
// AFWEBGPU_KV pins the packed key/value width the register kernel reads.
const kv = process.env.AFWEBGPU_KV;
if (kv !== undefined && kv !== "") forceAttentionKeyValueStorage(kv as AttentionKeyValueStorage);
for (const headDim of [8, 32]) {
  try {
    const shape = await calibrateAttentionShape(device, headDim);
    console.error(`attention shape headDim=${headDim}: slots=${shape?.slots} keyValue=${shape?.keyValue}`);
  } catch (error) { console.error(`attention shape headDim=${headDim}: ${(error as Error).message}`); }
}
// AFWEBGPU_SLOTS pins queries per invocation for the register kernels.
const slotsEnv = process.env.AFWEBGPU_SLOTS;
if (slotsEnv !== undefined && slotsEnv !== "") forceAttentionQueriesPerThread(Number(slotsEnv));
const attentionVariant = process.env.AFWEBGPU_ATTENTION;
if (attentionVariant !== undefined && attentionVariant !== "") {
  const pinned: number[] = [];
  for (const headDim of [8, 16, 32, 64]) {
    try {
      presetAttentionFlashKernel(device, headDim, attentionVariant as AttentionFlashVariant);
      pinned.push(headDim);
    } catch { /* this head width has no such kernel; it keeps its own choice. */ }
  }
  console.error(`attention kernel: ${attentionVariant} for head dims ${pinned.join(",") || "none"}`);
}
let largestBufferBytes = 0;
const createBuffer = device.createBuffer.bind(device);
(device as { createBuffer: GPUDevice["createBuffer"] }).createBuffer = (descriptor: GPUBufferDescriptor) => {
  largestBufferBytes = Math.max(largestBufferBytes, descriptor.size);
  return createBuffer(descriptor);
};
try {
  const features = iterateMultimerA3mFeatures(chains, complex.a3m, complex.mask, featureTables, {
    recycles: recycles - 1, randomSeed: 0, maxMsaSequences: msaRows, maxExtraSequences: extraRows,
  });
  const storage = process.env.AFWEBGPU_EXACT === "1" ? { ...EXACT_STORAGE } : {};
  const profiling = process.env.AFWEBGPU_PROFILE === "1"
    ? { profile: true, profileRecycle: recycles - 1 } : {};
  const prediction = await new AlphaFoldMultimerGpu(device, { ...storage, ...profiling }).predict(features, {
    embedding, multimerTemplate, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, paeBreaks, (summary, recycle) => {
    console.error(`recycle=${recycle} pLDDT=${summary.confidence.meanPlddt.toFixed(1)}`
      + ` pTM=${summary.confidence.ptm.toFixed(3)}`
      + ` ipTM=${(summary.confidence as { iptm?: number }).iptm?.toFixed(3) ?? "-"}`
      + ` time=${(summary.elapsedMilliseconds / 1000).toFixed(2)} s`);
  });
  const pdbOut = process.env.AFWEBGPU_PDB_OUT;
  if (pdbOut !== undefined && pdbOut !== "") {
    const { predictionToPdb } = await import("../web/prediction-results.js");
    writeFileSync(pdbOut, predictionToPdb(chains.join(""), prediction.final.structure,
      prediction.final.confidence.plddt, chains.map((chain) => chain.length)));
    console.error(`wrote ${pdbOut}`);
  }
  const profile = (prediction.final as { gpuProfile?: {
    extraMsa: { block: number; method: string; wallMilliseconds: number;
      entries: readonly { label: string; nanoseconds: number }[] };
    mainEvoformer: { block: number; method: string; wallMilliseconds: number;
      entries: readonly { label: string; nanoseconds: number }[] };
  } }).gpuProfile;
  if (profile !== undefined) {
    for (const [name, block, count] of [
      ["extra-MSA", profile.extraMsa, extraStack.length],
      ["main-Evoformer", profile.mainEvoformer, mainStack.length],
    ] as const) {
      const gpuMs = block.entries.reduce((sum, e) => sum + e.nanoseconds, 0) / 1e6;
      console.error(`\n== ${name} block ${block.block} (${block.method}) ==`);
      console.error(`dispatches=${block.entries.length} gpu=${gpuMs.toFixed(3)}ms `
        + `wall=${block.wallMilliseconds.toFixed(3)}ms`);
      console.error(`stack projection: ${count} blocks -> gpu ${(gpuMs * count / 1000).toFixed(2)}s`);
      const byLabel = new Map<string, { total: number; count: number }>();
      for (const entry of block.entries) {
        const key = entry.label.replace(/[.-]?\d+$/, "");
        const current = byLabel.get(key) ?? { total: 0, count: 0 };
        byLabel.set(key, { total: current.total + entry.nanoseconds, count: current.count + 1 });
      }
      for (const [label, value] of [...byLabel].sort((a, b) => b[1].total - a[1].total)) {
        console.error(`  ${(value.total / 1e6).toFixed(3)}ms  x${value.count}  ${label}`
          + `   (stack ${(value.total * count / 1e9).toFixed(2)}s)`);
      }
    }
  }
  console.log(JSON.stringify({
    file, length, chains: chains.map((chain) => chain.length), depth: complex.depth,
    recycles, msaRows, extraRows, gemm: gemmVariantName(chosenGemm),
    shaderF16: adapter.features.has("shader-f16"),
    millisecondsPerRecycle: Math.round(prediction.elapsedMilliseconds / recycles),
    peakConcurrentMiB: Math.round(prediction.memory.peakBytes / 1024 ** 2),
    peakResidentMiB: Math.round(prediction.memory.combinedPeakResidentBytes / 1024 ** 2),
    meanPlddt: Number(prediction.final.confidence.meanPlddt.toFixed(2)),
    ptm: Number(prediction.final.confidence.ptm.toFixed(3)),
    iptm: Number(((prediction.final.confidence as { iptm?: number }).iptm ?? 0).toFixed(3)),
    largestBufferMiB: Math.round(largestBufferBytes / 1024 ** 2),
  }));
} finally {
  device.destroy();
}
