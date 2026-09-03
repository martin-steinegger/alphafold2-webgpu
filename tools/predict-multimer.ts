/**
 * Runs Multimer-v3 on colon-separated chains without an alignment and prints
 * the per-recycle confidence and allocator peaks.
 *
 * Usage: tsx tools/predict-multimer.ts <manifest.json> <CHAIN_A:CHAIN_B[:...]> [recycles]
 */
import { create, globals } from "webgpu";
import { AlphaFoldMultimerGpu } from "../src/model/multimer.js";
import { EXACT_STORAGE } from "../src/model/monomer.js";
import {
  iterateMultimerA3mFeatures, iterateMultimerQueryOnlyFeatures,
} from "../src/input/multimer-features.js";
import { generateMmseqs2ComplexMsa } from "../src/input/mmseqs2-api.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";
Object.assign(globalThis, globals);
const manifest = process.argv[2];
const chains = process.argv[3]?.split(":").map((chain) => chain.trim().toUpperCase()).filter((chain) => chain !== "");
if (manifest === undefined || chains === undefined || chains.length === 0) {
  throw new Error("usage: predict-multimer.ts <manifest.json> <CHAIN_A:CHAIN_B[:...]> [recycles]");
}
const recycles = Number(process.argv[4] ?? "1");
const length = chains.reduce((sum, chain) => sum + chain.length, 0);
const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(manifest));
const [embedding, multimerTemplate, extraStack, mainStack, structure, confidence, geometry, featureTables, paeBreaks] =
  await Promise.all([
    model.multimerEmbeddingWeights(), model.multimerTemplateWeights(), model.extraStackWeights(),
    model.mainStackWeights(), model.multimerStructureWeights(), model.confidenceWeights(), model.geometryTables(),
    model.queryOnlyFeatureTables(), model.tensor("confidencePaeBreaks"),
  ]);
const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
const plan = planMonomerDevice(adapter, length, 1, 1, undefined, false,
  { multimer: true, templateRows: multimerTemplate.templateRows,
    ...(process.env.AFWEBGPU_EXACT === "1" ? EXACT_STORAGE : {}) });
console.error(`chains ${chains.map((chain) => chain.length).join("+")} = ${length} residues; estimated peak `
  + `${(plan.memory.estimatedPeakBytes / 1024 ** 2).toFixed(0)} MiB`);
const device = await requestAlphaFoldDevice(adapter, plan.requirements);
try {
  // AFWEBGPU_COMPLEX_MSA=1 searches the public ColabFold server for paired and
  // unpaired alignments, which is the only way to judge a complex's confidence.
  let features = iterateMultimerQueryOnlyFeatures(chains, featureTables, { recycles: recycles - 1, randomSeed: 0 });
  if (process.env.AFWEBGPU_COMPLEX_MSA === "1") {
    const search = await generateMmseqs2ComplexMsa(chains, {
      onProgress: (progress) => console.error(`MMseqs2 ${progress.search ?? ""} ${progress.phase}`),
    });
    console.error(`complex alignment: ${search.depth} rows`);
    features = iterateMultimerA3mFeatures(chains, search.a3m, search.mask, featureTables, {
      recycles: recycles - 1, randomSeed: 0, maxMsaSequences: 252, maxExtraSequences: 1152,
    });
  }
  const storage = process.env.AFWEBGPU_EXACT === "1" ? { ...EXACT_STORAGE } : {};
const prediction = await new AlphaFoldMultimerGpu(device, storage).predict(features, {
    embedding, multimerTemplate, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, paeBreaks, (summary, recycle) => {
    console.error(`recycle=${recycle} pLDDT=${summary.confidence.meanPlddt.toFixed(1)} pTM=${summary.confidence.ptm.toFixed(3)}`
      + ` ipTM=${(summary.confidence as { iptm?: number }).iptm?.toFixed(3) ?? "-"} time=${(summary.elapsedMilliseconds / 1000).toFixed(2)} s`);
  });
  console.log(JSON.stringify({
    length, chains: chains.map((chain) => chain.length), recycles,
    millisecondsPerRecycle: Math.round(prediction.elapsedMilliseconds / recycles),
    peakConcurrentMiB: Math.round(prediction.memory.peakBytes / 1024 ** 2),
    peakResidentMiB: Math.round(prediction.memory.combinedPeakResidentBytes / 1024 ** 2),
    meanPlddt: Number(prediction.final.confidence.meanPlddt.toFixed(2)),
    ptm: Number(prediction.final.confidence.ptm.toFixed(3)),
    iptm: (prediction.final.confidence as { iptm?: number }).iptm,
  }));
} finally {
  device.destroy();
}
