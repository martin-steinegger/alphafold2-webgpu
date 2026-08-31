/**
 * Runs the monomer model at an arbitrary sequence length and MSA depth.
 *
 * AlphaFold's weights do not depend on the sequence length, so the released
 * parameters can be driven with a synthetic alignment. The fixture benchmark is
 * fixed at 59 residues, where the MSA axis dominates; this one reaches the
 * lengths where the pair axis does.
 *
 * Usage: tsx tools/benchmark-shape.ts [length] [msaRows] [extraRows] [recycles]
 */
import { create, globals } from "webgpu";
import { AlphaFoldMonomerGpu } from "../src/model/monomer.js";
import { makeA3mFeatures } from "../src/input/a3m-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

Object.assign(globalThis, globals);

const length = Number(process.argv[2] ?? "128");
const msaRows = Number(process.argv[3] ?? "128");
const extraRows = Number(process.argv[4] ?? "512");
const recycles = Number(process.argv[5] ?? "1");
const AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY";

// A deterministic alignment: the query plus mutated copies of it.
let state = 0x1234567;
const random = (): number => {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  return state / 0x100000000;
};
const query = Array.from({ length }, () => AMINO_ACIDS[Math.floor(random() * 20)]!).join("");
const rows = [`>query\n${query}`];
for (let row = 1; row < msaRows + extraRows; row += 1) {
  const mutated = Array.from(query, (residue) => (random() < 0.25
    ? (random() < 0.4 ? "-" : AMINO_ACIDS[Math.floor(random() * 20)]!) : residue)).join("");
  rows.push(`>seq${row}\n${mutated}`);
}
const a3m = `${rows.join("\n")}\n`;

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
const device = await requestAlphaFoldDevice(adapter);
const tally = new Map<string, { bytes: number; count: number }>();
if (process.env.AFWEBGPU_MEMORY === "1") {
  const original = device.createBuffer.bind(device);
  (device as { createBuffer: GPUDevice["createBuffer"] }).createBuffer = (descriptor: GPUBufferDescriptor) => {
    const label = (descriptor.label ?? "unlabelled").replace(/-?\d+$/, "");
    const entry = tally.get(label) ?? { bytes: 0, count: 0 };
    entry.bytes += descriptor.size; entry.count += 1;
    tally.set(label, entry);
    return original(descriptor);
  };
}
try {
  const profile = process.env.AFWEBGPU_PROFILE === "1";
  const prediction = await new AlphaFoldMonomerGpu(device, profile ? { profile: true } : {}).predict(features, {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, await model.tensor("confidencePaeBreaks"));
  console.log(JSON.stringify({
    shape: { length, msaRows, extraRows, recycles },
    elapsedMilliseconds: Math.round(prediction.elapsedMilliseconds),
    millisecondsPerRecycle: Math.round(prediction.elapsedMilliseconds / recycles),
    peakResidentMiB: Math.round(prediction.memory.peakResidentBytes / 1024 ** 2),
    peakConcurrentMiB: Math.round(prediction.memory.peakBytes / 1024 ** 2),
    meanPlddt: Number(prediction.final.confidence.meanPlddt.toFixed(3)),
  }));
  if (tally.size > 0) {
    const total = [...tally.values()].reduce((sum, entry) => sum + entry.bytes, 0);
    console.error(`\nGPU buffers created (total ${(total / 1024 ** 2).toFixed(0)} MiB):`);
    for (const [label, entry] of [...tally].sort((a, b) => b[1].bytes - a[1].bytes).slice(0, 14)) {
      console.error(`  ${(entry.bytes / 1024 ** 2).toFixed(1).padStart(8)} MiB  x${String(entry.count).padStart(3)}  ${label}`);
    }
  }
  const gpuProfile = prediction.final.gpuProfile;
  if (gpuProfile !== undefined) {
    for (const [name, block, blocks] of [
      ["extra-MSA", gpuProfile.extraMsa, extraStack.length],
      ["main-Evoformer", gpuProfile.mainEvoformer, mainStack.length],
    ] as const) {
      const total = block.entries.reduce((sum, entry) => sum + entry.nanoseconds, 0) / 1e6;
      console.error(`\n== ${name} block ${block.block}: ${total.toFixed(2)} ms x ${blocks} blocks `
        + `= ${(total * blocks / 1000).toFixed(2)} s ==`);
      for (const entry of [...block.entries].sort((a, b) => b.nanoseconds - a.nanoseconds).slice(0, 12)) {
        console.error(`  ${(entry.nanoseconds / 1e6).toFixed(3)} ms  ${entry.label}`);
      }
    }
  }
} finally { device.destroy(); }
