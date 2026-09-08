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
import { dawnInstanceFlags } from "../src/runtime/dawn.js";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { parseA3m } from "../src/input/a3m.js";
import { makeA3mFeatures, type RecycleFeatureSource } from "../src/input/a3m-features.js";
import type { MonomerRecycleFeatures } from "../src/model/monomer.js";
import { prepareTemplate, withTemplate } from "../src/input/template.js";
import { predictionToPdb } from "../web/prediction-results.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";
import { calibrateGemmVariant, gemmVariantName } from "../src/runtime/gemm-selection.js";
import { forceGemmVariant } from "../src/runtime/gemm-selection.js";
import { presetAttentionFlashKernel } from "../src/evoformer/attention-calibration.js";
import {
  forceAttentionKeyValueStorage, forceAttentionQueriesPerThread,
  type AttentionFlashVariant, type AttentionKeyValueStorage,
} from "../src/evoformer/attention.js";
Object.assign(globalThis, globals);
const file = process.argv[2];
if (file === undefined) throw new Error("usage: predict-a3m.ts <file.a3m> [msaRows] [extraRows] [recycles]");
const msaRows = Number(process.argv[3] ?? "508");
const extraRows = Number(process.argv[4] ?? "1024");
const recycles = Number(process.argv[5] ?? "4");
const a3m = readFileSync(file, "utf8");
// Use the production parser for sizing as well as feature construction. MMseqs2
// A3Ms begin with a `#length\tchains` metadata line, which is not a sequence.
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
let features: RecycleFeatureSource<MonomerRecycleFeatures> = makeA3mFeatures(a3m, featureTables, {
  recycles: recycles - 1, maxMsaSequences: msaRows, maxExtraSequences: extraRows, randomSeed: 0,
});
// AFWEBGPU_TEMPLATE=<structure> folds with a custom template, the way the page
// does; AFWEBGPU_TEMPLATE_CHAIN picks a chain other than the first.
const templatePath = process.env.AFWEBGPU_TEMPLATE;
let templateReport: Record<string, unknown> | undefined;
if (templatePath !== undefined && templatePath !== "") {
  const prepared = prepareTemplate(readFileSync(templatePath, "utf8"), alignment.query, {
    ...(process.env.AFWEBGPU_TEMPLATE_CHAIN === undefined
      ? {} : { chainId: process.env.AFWEBGPU_TEMPLATE_CHAIN }),
  });
  features = withTemplate(features, prepared.features);
  templateReport = {
    file: templatePath, chain: prepared.chain.id, residues: prepared.chain.sequence.length,
    covered: prepared.alignment.alignedResidues,
    coverage: Number((prepared.alignment.coverage * 100).toFixed(1)),
    identity: Number((prepared.alignment.identity * 100).toFixed(1)),
  };
}
// The same native flags every shipped node entry point sets: without them the
// adapter reports no matrix units at all and the probe silently measures a
// device that has them.
const gpu = create(dawnInstanceFlags({ unclamped: true }));
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
const clustered = Math.min(msaRows, depth);
const extra = Math.max(1, Math.min(extraRows, Math.max(0, depth - clustered)));
// The model stores its activations packed. AFWEBGPU_EXACT=1 selects the f32
// storages the differential tests use, for comparison.
const memoryOptions = process.env.AFWEBGPU_EXACT === "1" ? { ...EXACT_STORAGE } : {};
// AFWEBGPU_SCRATCH_SCALE widens the bounded scratch windows for a run that
// owns a large accelerator rather than a browser's share of one.
const scratchBudgetScale = Number(process.env.AFWEBGPU_SCRATCH_SCALE ?? "1");
const plan = planMonomerDevice(adapter, length, clustered, extra, undefined, false,
  { ...(templateReport === undefined ? memoryOptions : { ...memoryOptions, template: true }),
    scratchBudgetScale });
// AFWEBGPU_GEMM pins the projection variant, so a comparison of anything else
// is not moved by the calibration picking differently between runs.
const gemmPin = process.env.AFWEBGPU_GEMM;
if (gemmPin === "f16-chunked") forceGemmVariant({ precision: "f16-chunked", inner: 8 });
if (gemmPin === "f32") forceGemmVariant({ precision: "f32", inner: 8 });
const device = await requestAlphaFoldDevice(adapter, plan.requirements);
console.error(`f16=${adapter.features.has("shader-f16")} scratch=${scratchBudgetScale}x`
  + ` gemm=${gemmVariantName(await calibrateGemmVariant(device))}`);
// AFWEBGPU_ATTENTION pins the flash kernel; AFWEBGPU_KV its key/value width.
const attentionVariant = process.env.AFWEBGPU_ATTENTION;
if (attentionVariant !== undefined && attentionVariant !== "") {
  for (const headDim of [8, 16, 32, 64]) {
    try { presetAttentionFlashKernel(device, headDim, attentionVariant as AttentionFlashVariant); }
    catch { /* no such kernel at this head width */ }
  }
}
// AFWEBGPU_SLOTS pins queries per invocation, which the probe measures on a
// synthetic shape rather than on the ones the model runs.
const slotsEnv = process.env.AFWEBGPU_SLOTS;
if (slotsEnv !== undefined && slotsEnv !== "") forceAttentionQueriesPerThread(Number(slotsEnv));
for (const headDim of [8, 32]) {
  try {
    const chosen = await (await import("../src/evoformer/attention-calibration.js"))
      .calibrateAttentionFlashKernel(device, headDim);
    console.error(`flash kernel headDim=${headDim}: ${chosen.variant}`);
  } catch (error) { console.error(`flash headDim=${headDim}: ${(error as Error).message}`); }
}
if (process.env.AFWEBGPU_GEMM_TABLE === "1") {
  const { measureGemmVariants } = await import("../src/runtime/gemm-selection.js");
  const { subgroupMatrixConfigs } = await import("../src/runtime/subgroups.js");
  const measured = await measureGemmVariants(device, subgroupMatrixConfigs(device));
  for (const entry of [...measured].sort((a, b) => a.milliseconds - b.milliseconds)) {
    console.error(`  ${gemmVariantName(entry.variant).padEnd(28)} `
      + `${entry.milliseconds.toFixed(3).padStart(8)} ms  `
      + `perShape [${entry.perShape.map((value) => value.toFixed(3)).join(", ")}]  `
      + `relErr ${entry.relativeError.toExponential(2)}`);
  }
}
const kvWidth = process.env.AFWEBGPU_KV;
if (kvWidth !== undefined && kvWidth !== "") {
  forceAttentionKeyValueStorage(kvWidth as AttentionKeyValueStorage);
}
try {
  const profiling = process.env.AFWEBGPU_PROFILE === "1"
    ? { profile: true, profileRecycle: recycles - 1 } : {};
  const prediction = await new AlphaFoldMonomerGpu(device, {
    ...memoryOptions, ...profiling,
  }).predict(features, {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, await model.tensor("confidencePaeBreaks"), (summary, recycle) => {
    // Per recycle, because the first absorbs the clock ramp and the pipeline
    // compiles; only the later ones are the steady state worth comparing.
    console.error(`recycle=${recycle} ${(summary.elapsedMilliseconds / 1000).toFixed(2)} s`);
  });
  const profile = (prediction.final as unknown as { gpuProfile?: Record<string, {
    block: number; method: string; wallMilliseconds: number;
    entries: readonly { label: string; nanoseconds: number }[];
  }> }).gpuProfile;
  if (profile !== undefined) {
    for (const [name, count] of [["extraMsa", extraStack.length],
      ["mainEvoformer", mainStack.length]] as const) {
      const block = profile[name]!;
      const gpuMs = block.entries.reduce((sum, e) => sum + e.nanoseconds, 0) / 1e6;
      console.error(`\n== ${name} block ${block.block} (${block.method}) ==`);
      console.error(`dispatches=${block.entries.length} gpu=${gpuMs.toFixed(3)}ms `
        + `wall=${block.wallMilliseconds.toFixed(3)}ms  stack=${(gpuMs * count / 1000).toFixed(2)}s`);
      const byLabel = new Map<string, { total: number; count: number }>();
      for (const entry of block.entries) {
        const key = entry.label.replace(/[.-]?\d+$/, "");
        const current = byLabel.get(key) ?? { total: 0, count: 0 };
        byLabel.set(key, { total: current.total + entry.nanoseconds, count: current.count + 1 });
      }
      for (const [label, value] of [...byLabel].sort((a, b) => b[1].total - a[1].total).slice(0, 18)) {
        console.error(`  ${(value.total / 1e6).toFixed(3)}ms x${value.count}  ${label}`
          + `  (stack ${(value.total * count / 1e9).toFixed(2)}s)`);
      }
    }
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
