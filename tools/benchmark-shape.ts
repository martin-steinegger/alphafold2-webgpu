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
import { planMonomerDevice, requestAlphaFoldDevice } from "../src/runtime/device.js";

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
const memoryOptions = {
  ...(process.env.AFWEBGPU_TRIANGLE_F16 === "1" ? { triangleWholeStorage: "f16" as const } : {}),
  ...(process.env.AFWEBGPU_MSA_F16 === "1" ? { msaStorage: "f16" as const } : {}),
};
// Match what the page does: size the device to the shape, not to the base tier.
const plan = planMonomerDevice(adapter, length, msaRows, extraRows, undefined, false, memoryOptions);
const device = await requestAlphaFoldDevice(adapter, plan.requirements);
console.error(`device limits: buffer ${(plan.requirements.maxBufferSize / 1024 ** 2).toFixed(0)} MiB, `
  + `storage binding ${(plan.requirements.maxStorageBufferBindingSize / 1024 ** 2).toFixed(0)} MiB, `
  + `estimated peak ${(plan.memory.estimatedPeakBytes / 1024 ** 2).toFixed(0)} MiB, mode ${plan.transitionMode}`);
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
  const profileMode = process.env.AFWEBGPU_PROFILE ?? "";
  const profile = profileMode !== "";
  const poolMib = Number(process.env.AFWEBGPU_POOL_MIB ?? "");
  const prediction = await new AlphaFoldMonomerGpu(device, {
    ...(profile ? { profile: true } : {}),
    ...memoryOptions,
    ...(poolMib > 0 ? { maxPooledBytes: poolMib * 1024 ** 2 } : {}),
  }).predict(features, {
    embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry,
  }, await model.tensor("confidencePaeBreaks"));
  console.log(JSON.stringify({
    shape: { length, msaRows, extraRows, recycles },
    elapsedMilliseconds: Math.round(prediction.elapsedMilliseconds),
    millisecondsPerRecycle: Math.round(prediction.elapsedMilliseconds / recycles),
    peakResidentMiB: Math.round(prediction.memory.combinedPeakResidentBytes / 1024 ** 2),
    trunkPeakResidentMiB: Math.round(prediction.memory.mainPeakResidentBytes / 1024 ** 2),
    structureCorePeakMiB: Math.round(prediction.memory.structureCorePeakResidentBytes / 1024 ** 2),
    confidencePeakMiB: Math.round(prediction.memory.confidencePeakResidentBytes / 1024 ** 2),
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
      // Group windows and blocks of the same operation back together.
      if (profileMode === "raw") {
        // Per-dispatch labels with only the block index stripped, for kernel-level attribution.
        const raw = new Map<string, { milliseconds: number; count: number }>();
        for (const entry of block.entries) {
          const key = entry.label.replace(/-\d+$/, "");
          const current = raw.get(key) ?? { milliseconds: 0, count: 0 };
          current.milliseconds += entry.nanoseconds / 1e6; current.count += 1;
          raw.set(key, current);
        }
        for (const [label, value] of [...raw].sort((a, b) => b[1].milliseconds - a[1].milliseconds).slice(0, 40)) {
          console.error(`  ${value.milliseconds.toFixed(2).padStart(8)} ms  x${String(value.count).padStart(3)}  ${label}`);
        }
        continue;
      }
      const grouped = new Map<string, { milliseconds: number; count: number }>();
      for (const entry of block.entries) {
        const key = entry.label.replace(/-\d+$/, "").replace(/\.[a-z-]+$/, (suffix) => suffix);
        const operation = key.split(".").slice(0, 2).join(".");
        const current = grouped.get(operation) ?? { milliseconds: 0, count: 0 };
        current.milliseconds += entry.nanoseconds / 1e6;
        current.count += 1;
        grouped.set(operation, current);
      }
      for (const [operation, value] of [...grouped].sort((a, b) => b[1].milliseconds - a[1].milliseconds)) {
        const share = (value.milliseconds / total * 100).toFixed(0);
        console.error(`  ${value.milliseconds.toFixed(2).padStart(6)} ms  ${share.padStart(3)}%  `
          + `x${String(value.count).padStart(3)}  ${operation}`);
      }
    }
  }
} finally { device.destroy(); }
