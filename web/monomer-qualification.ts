import { makeA3mFeatures } from "../src/input/a3m-features.js";
import {
  AlphaFoldMonomerGpu, type MonomerGpuOptions, type MonomerModelWeights,
} from "../src/model/monomer.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const QUERY = "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK";

export interface MonomerQualificationPrediction {
  readonly recycles: readonly { readonly meanPlddt: number; readonly ptm: number }[];
  readonly plddt: readonly number[];
  readonly pae: readonly number[];
  readonly atom37: readonly number[];
}

export interface MonomerQualificationResult {
  readonly bundleId: string;
  readonly queryOnly: MonomerQualificationPrediction;
  readonly deepMsa: MonomerQualificationPrediction;
}

async function monomerWeights(fixture: AlphaFoldFixture): Promise<MonomerModelWeights> {
  const embedding = await fixture.embeddingWeights();
  const template = await fixture.templateWeights();
  const extraStack = await fixture.extraStackWeights();
  const mainStack = await fixture.mainStackWeights();
  const structure = await fixture.structureWeights();
  const confidence = await fixture.confidenceWeights();
  const geometry = await fixture.geometryTables();
  return { embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry };
}

function summarize(prediction: Awaited<ReturnType<AlphaFoldMonomerGpu["predict"]>>): MonomerQualificationPrediction {
  return {
    recycles: prediction.recycles.map((result) => ({
      meanPlddt: result.confidence.meanPlddt, ptm: result.confidence.ptm,
    })),
    plddt: Array.from(prediction.final.confidence.plddt),
    pae: Array.from(prediction.final.confidence.predictedAlignedError),
    atom37: Array.from(prediction.final.structure.atom37),
  };
}

/** Runs the fixed query-only and deep-MSA monomer qualification cases in Chrome WebGPU. */
export async function qualifyMonomer(
  modelManifestUrl: string,
  deepA3m: string,
  storageOptions: Pick<MonomerGpuOptions, "triangleWholeStorage" | "msaStorage"> = {},
): Promise<MonomerQualificationResult> {
  const store = await HttpTensorStore.open(modelManifestUrl);
  const fixture = AlphaFoldFixture.fromStore(store);
  const weights = await monomerWeights(fixture);
  const tables = await fixture.queryOnlyFeatureTables();
  const breaks = await fixture.tensor("confidencePaeBreaks");
  if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("no WebGPU adapter");
  const device = await requestAlphaFoldDevice(adapter);
  try {
    const runner = new AlphaFoldMonomerGpu(device, { compactTransitions: true, ...storageOptions });
    const queryOnly = await runner.predict(
      makeA3mFeatures(`>query\n${QUERY}\n`, tables, {
        recycles: 3, randomSeed: 0, maxMsaSequences: 1, maxExtraSequences: 1,
      }), weights, breaks,
    );
    const deepMsa = await runner.predict(
      makeA3mFeatures(deepA3m, tables, {
        recycles: 0, randomSeed: 0, maxMsaSequences: 508, maxExtraSequences: 1024,
      }), weights, breaks,
    );
    return {
      bundleId: store.manifest.bundle?.id ?? "unversioned",
      queryOnly: summarize(queryOnly), deepMsa: summarize(deepMsa),
    };
  } finally {
    device.destroy();
  }
}
