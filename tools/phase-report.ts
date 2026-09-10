/**
 * Folds an A3M and prints where the wall clock went, end to end.
 *
 * The rows sum to the total by construction, so a phase nobody named shows up
 * as unaccounted rather than being absorbed by its neighbour. That is the
 * whole point: featurisation built each recycle's features from inside the
 * recycle loop, so its seconds read as GPU time until they had a row.
 *
 * Usage: tsx tools/phase-report.ts <file.a3m> [msaRows] [extraRows] [recycles] [repeats]
 */
import { readFileSync } from "node:fs";
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { dawnInstanceFlags, fitScratchBudgetScale, scratchBudgetScalesFor } from "../src/runtime/dawn.js";
import { iterateA3mFeatures } from "../src/input/a3m-features.js";
import { nativeMemoryBudgetBytes, selectGpu, useFileCalibrationStore } from "./native-device.js";
import { parseA3m } from "../src/input/a3m.js";
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";
import { requestWgpuAdapter } from "../src/runtime/wgpu/adapter.js";
import {
  PhaseLedger, endPhase, formatPhaseReport, markPhase, setPhaseLedger,
} from "../src/runtime/phase-ledger.js";

// performance.now() counts from process start on node, so origin zero puts the
// imports in the ledger instead of before it.
const ledger = new PhaseLedger(0, "module load");
setPhaseLedger(ledger);

Object.assign(globalThis, globals);
selectGpu();
const file = process.argv[2];
if (file === undefined) throw new Error("usage: phase-report.ts <file.a3m> [msaRows] [extraRows] [recycles] [repeats]");
const msaRows = Number(process.argv[3] ?? "508");
const extraRows = Number(process.argv[4] ?? "1024");
const recycles = Number(process.argv[5] ?? "3");
// Repeating the same prediction in one process separates what a fold costs
// cold from what it costs warm, which is what batch inference actually pays.
const repeats = Number(process.argv[6] ?? "1");
const manifest = process.env.AFWEBGPU_MANIFEST ?? ".models/model/manifest.json";
// A cold start measures the compiles; the default reuses the stored answer, as
// a second visit would. AFWEBGPU_COLD=1 asks the question again.
if (process.env.AFWEBGPU_COLD !== "1") useFileCalibrationStore(manifest);

markPhase("read a3m");
const a3m = readFileSync(file, "utf8");

markPhase("parse a3m");
const alignment = parseA3m(a3m);
const { length, depth } = alignment;

markPhase("load weights");
const model = AlphaFoldFixture.fromStore(await FileTensorStore.open(manifest));
const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables] =
  await Promise.all([
    model.embeddingWeights(), model.templateWeights(), model.extraStackWeights(),
    model.mainStackWeights(), model.structureWeights(), model.confidenceWeights(),
    model.geometryTables(), model.queryOnlyFeatureTables(),
  ]);

markPhase("create instance");
// AFWEBGPU_BACKEND=wgpu profiles the same fold over the wgpu addon. The
// instance is held rather than left as a temporary: dawn.node pumps its event
// loop from this object, and one that becomes collectable takes the device
// with it part way through the fold.
const gpu = process.env.AFWEBGPU_BACKEND === "wgpu"
  ? undefined : create(dawnInstanceFlags({ unclamped: true }));
markPhase("request adapter");
const adapter = gpu === undefined
  ? requestWgpuAdapter()
  : await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter");
markPhase("plan device");
const clustered = Math.max(1, Math.min(msaRows, depth));
const extra = Math.max(1, Math.min(extraRows, Math.max(0, depth - clustered)));
const budget = nativeMemoryBudgetBytes();
const scratchBudgetScale = budget === undefined ? 1 : fitScratchBudgetScale(
  (scale) => planMonomerDevice(adapter, length, clustered, extra, undefined, false,
    { scratchBudgetScale: scale }).memory.estimatedPeakBytes,
  budget, scratchBudgetScalesFor(length));
const plan = planMonomerDevice(adapter, length, clustered, extra, undefined, false,
  { scratchBudgetScale });
markPhase("request device");
const device = await requestAlphaFoldDevice(adapter, plan.requirements);

markPhase("build model");
const monomer = new AlphaFoldMonomerGpu(device, {});
const paeBreaks = await model.tensor("confidencePaeBreaks");

endPhase();
const foldMilliseconds: number[] = [];
let prediction!: Awaited<ReturnType<typeof monomer.predict>>;
for (let fold = 0; fold < repeats; fold += 1) {
  const start = performance.now();
  prediction = await monomer.predict(
    iterateA3mFeatures(device, alignment, featureTables, {
      recycles: recycles - 1, maxMsaSequences: msaRows, maxExtraSequences: extraRows, randomSeed: 0,
    }),
    { embedding, template, extraStack, mainStack, structure,
      lddt: confidence.lddt, pae: confidence.pae, geometry },
    paeBreaks);
  foldMilliseconds.push(performance.now() - start);
  console.log(`fold ${fold + 1}/${repeats}: ${(foldMilliseconds[fold]! / 1000).toFixed(2)} s,`
    + ` pLDDT ${prediction.final.confidence.meanPlddt.toFixed(2)},`
    + ` pTM ${prediction.final.confidence.ptm.toFixed(3)}`);
}

markPhase("report");
const report = ledger.report();
console.log(`${length} residues, ${depth} rows, ${clustered} clustered, ${extra} extra,`
  + ` ${recycles} recycles, ${repeats} fold(s), scratch ${scratchBudgetScale}x`);
console.log(`pLDDT ${prediction.final.confidence.meanPlddt.toFixed(2)},`
  + ` pTM ${prediction.final.confidence.ptm.toFixed(3)}`);
console.log(formatPhaseReport(report));
const summed = report.rows.reduce((sum, row) => sum + row.milliseconds, 0);
console.log(`rows sum to ${(summed / 1000).toFixed(2)} s of ${(report.totalMilliseconds / 1000).toFixed(2)} s`);
// Without this the process does not exit: dawn.node keeps pumping
// InstanceBase::ProcessEvents from the event loop and deadlocks on the
// instance mutex once its worker threads have gone. Every other tool here
// already destroys its device.
device.destroy();
