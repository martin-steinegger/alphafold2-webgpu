/**
 * Runs the monomer model on an A3M file with explicit row caps and prints the
 * per-recycle confidence and the allocator peaks, for quality-versus-memory
 * comparisons on real alignments.
 *
 * Usage: tsx tools/predict-a3m.ts <file.a3m> [msaRows] [extraRows] [recycles]
 */
import { EXACT_STORAGE } from "../src/model/monomer.js";
import { readFileSync, writeFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { dawnInstanceFlags, fitScratchBudgetScale } from "../src/runtime/dawn.js";
import { nativeMemoryBudgetBytes, selectGpu } from "./native-device.js";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { parseA3m } from "../src/input/a3m.js";
import { iterateA3mFeatures, type RecycleFeatureSource } from "../src/input/a3m-features.js";
import type { MonomerRecycleFeatures } from "../src/model/monomer.js";
import { prepareTemplate, withTemplate } from "../src/input/template.js";
import { predictionToPdb } from "../web/prediction-results.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";
import { requestWgpuAdapter } from "../src/runtime/wgpu/adapter.js";
Object.assign(globalThis, globals);
const file = process.argv[2];
if (file === undefined) throw new Error("usage: predict-a3m.ts <file.a3m> [msaRows] [extraRows] [recycles]");
const msaRows = Number(process.argv[3] ?? "508");
const extraRows = Number(process.argv[4] ?? "1024");
const recycles = Number(process.argv[5] ?? "4");
const a3m = readFileSync(file, "utf8");
// Use the production parser for sizing as well as feature construction. MMseqs2
// A3Ms begin with a #length\tchains metadata line, which is not a sequence.
const alignment = parseA3m(a3m);
const { length, depth } = alignment;
// AFWEBGPU_MANIFEST runs against a different bundle, such as the quantized one
// the site serves, rather than the local float32 fixture.
const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(
  process.env.AFWEBGPU_MANIFEST ?? "test/fixtures/evoformer/model1-query-59-stack/manifest.json",
));
const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables] = await Promise.all([
  model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(), model.mainStackWeights(),
  model.structureWeights(), model.confidenceWeights(), model.geometryTables(), model.queryOnlyFeatureTables(),
]);
let applyTemplate: ((s: RecycleFeatureSource<MonomerRecycleFeatures>)
  => RecycleFeatureSource<MonomerRecycleFeatures>) | undefined;
let features: RecycleFeatureSource<MonomerRecycleFeatures>;
// AFWEBGPU_TEMPLATE=<structure> folds with a custom template, the way the page
// does; AFWEBGPU_TEMPLATE_CHAIN picks a chain other than the first.
const templatePath = process.env.AFWEBGPU_TEMPLATE;
let templateReport: Record<string, unknown> | undefined;
if (templatePath !== undefined && templatePath !== "") {
  const prepared = prepareTemplate(readFileSync(templatePath, "utf8"), alignment.query, {
    ...(process.env.AFWEBGPU_TEMPLATE_CHAIN === undefined
      ? {} : { chainId: process.env.AFWEBGPU_TEMPLATE_CHAIN }),
  });
  applyTemplate = (source: RecycleFeatureSource<MonomerRecycleFeatures>) =>
    withTemplate(source, prepared.features);
  templateReport = {
    file: templatePath, chain: prepared.chain.id, residues: prepared.chain.sequence.length,
    covered: prepared.alignment.alignedResidues,
    coverage: Number((prepared.alignment.coverage * 100).toFixed(1)),
    identity: Number((prepared.alignment.identity * 100).toFixed(1)),
  };
}
// Chosen before the instance exists, because the Vulkan loader reads the
// selection when it makes one. Honours CUDA_VISIBLE_DEVICES; see selectGpu.
const selectedGpu = selectGpu();
if (selectedGpu !== undefined) console.error(`pinned to ${selectedGpu}`);
// AFWEBGPU_BACKEND=wgpu folds over the wgpu addon instead of Dawn. Both are
// asked for without the bounds clamp: the kernels do not rely on it, and it is
// worth 11% of a recycle. See dawnInstanceFlags.
// The instance is held, not left as a temporary: dawn.node pumps its event
// loop from this object, and one that becomes collectable takes the device
// with it part way through the fold.
const gpu = process.env.AFWEBGPU_BACKEND === "wgpu"
  ? undefined : create(dawnInstanceFlags({ unclamped: true }));
const adapter = gpu === undefined
  ? requestWgpuAdapter()
  : await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
const clustered = Math.min(msaRows, depth);
const extra = Math.max(1, Math.min(extraRows, Math.max(0, depth - clustered)));
// The model stores its activations packed. AFWEBGPU_EXACT=1 selects the f32
// storages the differential tests use, for comparison.
const memoryOptions = process.env.AFWEBGPU_EXACT === "1" ? { ...EXACT_STORAGE } : {};
const baseMemoryOptions = templateReport === undefined
  ? memoryOptions : { ...memoryOptions, template: true };
// The scratch windows are sized to the memory this host actually has rather
// than to the browser's conservative default, which costs a 1,650-residue
// dimer a third of its speed. WebGPU cannot say how much that is, so it is
// asked for out of band and each scale is costed against it.
const memoryBudget = nativeMemoryBudgetBytes();
const scratchBudgetScale = memoryBudget === undefined ? 1 : fitScratchBudgetScale(
  (scale) => planMonomerDevice(adapter, length, clustered, extra, undefined, false,
    { ...baseMemoryOptions, scratchBudgetScale: scale }).memory.estimatedPeakBytes,
  memoryBudget);
const plan = planMonomerDevice(adapter, length, clustered, extra, undefined, false,
  { ...baseMemoryOptions, scratchBudgetScale });
console.error(`device memory budget ${memoryBudget === undefined ? "unknown"
  : `${(memoryBudget / 1024 ** 3).toFixed(1)} GiB`}, scratch budget ${scratchBudgetScale}x`);
const device = await requestAlphaFoldDevice(adapter, plan.requirements);
// AFWEBGPU_OPM=f16|f32 pins how the outer product mean stores the two
// projections its contraction reads. Packed halves the bytes that contraction
// fetches; the default packs wherever the units round them to f16 anyway.
const opmOperands = process.env.AFWEBGPU_OPM;
if (opmOperands === "f16" || opmOperands === "f32") {
  const { forceOuterProductMeanOperands } = await import("../src/evoformer/outer-product-mean.js");
  forceOuterProductMeanOperands(opmOperands);
  console.error(`outer product mean operands pinned to ${opmOperands}`);
}
// AFWEBGPU_GEMM=f32 pins the projection away from the matrix units, so a fold
// can say what those units are worth to the model rather than to a probe.
if (process.env.AFWEBGPU_GEMM === "f32") {
  const { forceGemmVariant } = await import("../src/runtime/gemm-selection.js");
  const { GEMM_VARIANT_F32 } = await import("../src/runtime/gemm.js");
  forceGemmVariant(GEMM_VARIANT_F32);
  console.error("projection pinned to f32");
}
// AFWEBGPU_ATTENTION pins the flash kernel, so a fold can check what the
// calibration chose. Either one variant for every head width, or a comma-list
// of <headDim>:<variant>. Worth having: the probe measures sixteen batches of
// 256 queries, which is not the shape of every attention the model runs.
const pinnedAttention = process.env.AFWEBGPU_ATTENTION;
if (pinnedAttention !== undefined && pinnedAttention !== "") {
  const { presetAttentionFlashKernel } = await import("../src/evoformer/attention-calibration.js");
  for (const entry of pinnedAttention.split(",")) {
    const [head, variant] = entry.includes(":")
      ? [Number(entry.split(":")[0]), entry.split(":")[1]!] : [undefined, entry];
    for (const headDim of head === undefined ? [8, 16, 32, 64] : [head]) {
      // A head width with no such kernel keeps whatever it measured.
      try { presetAttentionFlashKernel(device, headDim, variant as never); } catch { /* none */ }
    }
  }
  console.error(`attention pinned to ${pinnedAttention}`);
}
features = iterateA3mFeatures(device, a3m, featureTables, {
  recycles: recycles - 1, maxMsaSequences: msaRows, maxExtraSequences: extraRows, randomSeed: 0,
});
if (applyTemplate !== undefined) features = applyTemplate(features);
try {
  // AFWEBGPU_PROFILE=1 times every dispatch of one extra-MSA and one main
  // block, which is the only view that says where the main stack goes.
  const profiling = process.env.AFWEBGPU_PROFILE === "1";
  const prediction = await new AlphaFoldMonomerGpu(device, {
    ...memoryOptions,
    ...(profiling ? { profile: true, profileRecycle: recycles - 1 } : {}),
  }).predict(features, {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, await model.tensor("confidencePaeBreaks"));
  const profile = prediction.final.gpuProfile;
  if (profile !== undefined) {
    for (const [name, block] of
      [["extra-MSA", profile.extraMsa], ["main-Evoformer", profile.mainEvoformer]] as const) {
      const gpuMilliseconds = block.entries.reduce((sum, entry) => sum + entry.nanoseconds, 0) / 1e6;
      console.error(`\n== ${name} block ${block.block} ==  ${block.entries.length} dispatches, `
        + `gpu ${gpuMilliseconds.toFixed(3)} ms, wall ${block.wallMilliseconds.toFixed(3)} ms`);
      const byLabel = new Map<string, { total: number; count: number }>();
      for (const entry of block.entries) {
        const key = entry.label.replace(/[.-]?\d+$/, "");
        const held = byLabel.get(key) ?? { total: 0, count: 0 };
        byLabel.set(key, { total: held.total + entry.nanoseconds, count: held.count + 1 });
      }
      for (const [label, value] of [...byLabel].sort((left, right) => right[1].total - left[1].total)) {
        const share = (value.total / 1e6) / gpuMilliseconds * 100;
        console.error(`  ${(value.total / 1e6).toFixed(3)} ms  ${share.toFixed(1).padStart(5)}%  `
          + `x${String(value.count).padStart(3)}  ${label}`);
      }
    }
  }
  // AFWEBGPU_PROFILE_JSON=<file> saves the same timings for tools/kernel-tflops.ts,
  // which turns them into arithmetic a second and holds them against Pallas.
  const profileJson = process.env.AFWEBGPU_PROFILE_JSON;
  if (profile !== undefined && profileJson !== undefined && profileJson !== "") {
    writeFileSync(profileJson, `${JSON.stringify({
      file, length, msaSequences: clustered, extraSequences: extra,
      extraMsa: profile.extraMsa, mainEvoformer: profile.mainEvoformer,
    }, undefined, 2)}\n`);
    console.error(`profile written to ${profileJson}`);
  }
  // AFWEBGPU_PDB=<file> writes the structure, for comparing against a template.
  const pdbPath = process.env.AFWEBGPU_PDB;
  if (pdbPath !== undefined && pdbPath !== "") {
    writeFileSync(pdbPath, predictionToPdb(
      alignment.query, prediction.final.structure, prediction.final.confidence.plddt,
    ));
  }
  console.log(JSON.stringify({
    file, length, depth, msaRows: clustered, extraRows: extra, recycles,
    ...(templateReport === undefined ? {} : { template: templateReport }),
    millisecondsPerRecycle: Math.round(prediction.elapsedMilliseconds / recycles),
    estimatedPeakMiB: Math.round(plan.memory.estimatedPeakBytes / 1024 ** 2),
    peakConcurrentMiB: Math.round(prediction.memory.peakBytes / 1024 ** 2),
    peakResidentMiB: Math.round(prediction.memory.combinedPeakResidentBytes / 1024 ** 2),
    meanPlddt: Number(prediction.final.confidence.meanPlddt.toFixed(2)),
    ptm: Number(prediction.final.confidence.ptm.toFixed(3)),
    recyclePlddt: prediction.recycles.map((result) => Number(result.confidence.meanPlddt.toFixed(2))),
  }));
  if (process.env.AFWEBGPU_MEMORY === "1") {
    for (const share of prediction.memory.peakComposition ?? []) {
      console.log(`  ${(share.bytes / 1024 ** 2).toFixed(1).padStart(8)} MiB  x ${share.count}  ${share.label}`);
    }
  }
} finally {
  device.destroy();
}
