/**
 * Runs the monomer model on an A3M file with explicit row caps and prints the
 * per-recycle confidence and the allocator peaks, for quality-versus-memory
 * comparisons on real alignments.
 *
 * Usage: tsx tools/predict-a3m.ts <file.a3m> [msaRows] [extraRows] [recycles]
 */
import { EXACT_STORAGE } from "../src/model/monomer.js";
import { readFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { parseA3m } from "../src/input/a3m.js";
import { makeA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";
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
const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(
  "test/fixtures/evoformer/model1-query-59-stack/manifest.json",
));
const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables] = await Promise.all([
  model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(), model.mainStackWeights(),
  model.structureWeights(), model.confidenceWeights(), model.geometryTables(), model.queryOnlyFeatureTables(),
]);
const features = makeA3mFeatures(a3m, featureTables, {
  recycles: recycles - 1, maxMsaSequences: msaRows, maxExtraSequences: extraRows, randomSeed: 0,
});
const gpu = create([]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
const clustered = Math.min(msaRows, depth);
const extra = Math.max(1, Math.min(extraRows, Math.max(0, depth - clustered)));
// The model stores its activations packed. AFWEBGPU_EXACT=1 selects the f32
// storages the differential tests use, for comparison.
const memoryOptions = process.env.AFWEBGPU_EXACT === "1" ? { ...EXACT_STORAGE } : {};
const plan = planMonomerDevice(adapter, length, clustered, extra, undefined, false, memoryOptions);
const device = await requestAlphaFoldDevice(adapter, plan.requirements);
try {
  const prediction = await new AlphaFoldMonomerGpu(device, {
    ...memoryOptions,
  }).predict(features, {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, await model.tensor("confidencePaeBreaks"));
  console.log(JSON.stringify({
    file, length, depth, msaRows: clustered, extraRows: extra, recycles,
    millisecondsPerRecycle: Math.round(prediction.elapsedMilliseconds / recycles),
    peakConcurrentMiB: Math.round(prediction.memory.peakBytes / 1024 ** 2),
    peakResidentMiB: Math.round(prediction.memory.combinedPeakResidentBytes / 1024 ** 2),
    meanPlddt: Number(prediction.final.confidence.meanPlddt.toFixed(2)),
    ptm: Number(prediction.final.confidence.ptm.toFixed(3)),
    recyclePlddt: prediction.recycles.map((result) => Number(result.confidence.meanPlddt.toFixed(2))),
  }));
} finally {
  device.destroy();
}
