/**
 * The prediction pipeline without the page: adapter, budget, model, features
 * and inference, reporting progress through callbacks. It runs identically on
 * the main thread and inside a dedicated worker, which is where the page runs
 * it when the browser exposes WebGPU there, so the document stays responsive
 * for the minutes a long chain takes.
 */
import {
  AlphaFoldMonomerGpu, type MonomerPrediction, type MonomerProgress, type MonomerRecycleSummary,
} from "../src/model/monomer.js";
import {
  AlphaFoldMultimerGpu, type MultimerPrediction, type MultimerRecycleSummary,
} from "../src/model/multimer.js";
import { iterateA3mFeatures } from "../src/input/a3m-features.js";
import {
  iterateMultimerA3mFeatures, iterateMultimerQueryOnlyFeatures, type MultimerRecycleFeatures,
} from "../src/input/multimer-features.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { HttpTensorStore, type TensorDownloadProgress } from "../src/reference/http-tensor-store.js";
import {
  planMonomerDevice, requestAlphaFoldDevice, suggestMonomerRows, type AlphaFoldDeviceRequirements,
} from "../src/runtime/device.js";
import { setGpuMemoryBudget } from "../src/runtime/allocator.js";
import { remainingPhrase, remainingTrunkSeconds } from "./progress.js";
import { adapterDisplayName } from "./webgpu-preflight.js";

export const inferenceStages = ["device", "msa", "model", "features", "inference", "results"] as const;
export type InferenceStage = typeof inferenceStages[number];
export type InferenceStageState = "active" | "done" | "error";

/** The alignment and sequence the page prepared; plain data so it crosses to a worker. */
export interface InferenceInput {
  readonly a3m: string; readonly sequence: string; readonly depth: number;
  readonly multimer: boolean; readonly chains?: readonly string[];
  readonly alignmentMask?: Float32Array;
}

export interface InferenceJob {
  /** Absolute manifest URL: a worker resolves relative URLs against its own script. */
  readonly manifestUrl: string;
  readonly input: InferenceInput;
  readonly maxMsaSequences: number;
  readonly maxExtraSequences: number;
  readonly recycles: number;
  readonly randomSeed: number;
  /** Force the bounded-transition memory policy. */
  readonly compactPolicy: boolean;
  readonly profile?: {
    readonly recycle: number; readonly extraMsaBlock: number; readonly mainEvoformerBlock: number;
  };
}

export interface InferenceReporter {
  stage(stage: InferenceStage, state: InferenceStageState, detail: string): void;
  status(text: string): void;
  log(text: string): void;
  modelProgress(progress: TensorDownloadProgress): void;
  recycle(summary: MonomerRecycleSummary | MultimerRecycleSummary, recycle: number): void;
}

export interface InferenceOutcome {
  readonly prediction: MonomerPrediction | MultimerPrediction;
  readonly modelLoadMilliseconds: number;
  readonly adapterName: string;
}

const formatSeconds = (milliseconds: number): string => `${(milliseconds / 1000).toFixed(2)} s`;
const formatMib = (bytes: number): string => `${(bytes / 1024 ** 2).toFixed(0)} MiB`;

export function browserPlatform(): string {
  const data = (navigator as Navigator & { readonly userAgentData?: { readonly platform?: string } }).userAgentData;
  return data?.platform ?? navigator.platform ?? "unknown";
}

function isAppleUnifiedMemory(adapter: GPUAdapter): boolean {
  const identity = `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.device} ${adapter.info.description}`
    .toLowerCase();
  return identity.includes("apple") || /mac/i.test(browserPlatform());
}

function unifiedMemoryBudget(appleUnifiedMemory: boolean): number | undefined {
  if (!appleUnifiedMemory) return undefined;
  const reported = (navigator as Navigator & { readonly deviceMemory?: number }).deviceMemory;
  // navigator.deviceMemory is Chromium-only, so on Firefox this budget used to
  // vanish and nothing stopped a prediction whose own estimate was several
  // gigabytes: it ran until WebGPU refused a buffer mid-way. Chromium caps the
  // figure at 8 anyway, so assuming it gives Firefox the budget every Chromium
  // user on Apple already gets rather than none.
  const deviceMemory = reported === undefined || !Number.isFinite(reported) || reported <= 0 ? 8 : reported;
  // Apple GPUs share system RAM, and Metal accepts allocations well past the
  // point where macOS starts paging, which freezes the machine rather than
  // failing the prediction. navigator.deviceMemory is privacy-rounded and capped
  // at 8 GB in Chromium, so this cannot see a larger machine; a third of what
  // it reports leaves room for Chrome, Dawn/Metal, the page and the rest of the
  // system. The same figure caps the allocator, so an underestimate becomes an
  // error rather than a freeze.
  return Math.floor(deviceMemory * 1024 ** 3 * 0.35);
}

async function loadModelWeights(manifestUrl: string, reporter: InferenceReporter) {
  const store = await HttpTensorStore.open(manifestUrl, (progress) => reporter.modelProgress(progress));
  const fixture = AlphaFoldFixture.fromStore(store);
  const manifestModel = store.manifest.model as { readonly type?: string; readonly number?: number } | undefined;
  const multimer = manifestModel?.type === "alphafold2_multimer_v3";
  if (multimer && manifestModel.number !== 1) {
    throw new Error("Only AlphaFold-Multimer-v3 model 1 manifests are supported");
  }
  const [embedding, extraStack, mainStack, structure, confidence, geometry, featureTables, paeBreaks] = await Promise.all([
    multimer ? fixture.multimerEmbeddingWeights() : fixture.embeddingWeights(),
    fixture.extraStackWeights(), fixture.mainStackWeights(),
    multimer ? fixture.multimerStructureWeights() : fixture.structureWeights(),
    fixture.confidenceWeights(), fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"),
  ] as const);
  const template = multimer ? undefined : await fixture.templateWeights();
  const multimerTemplate = multimer ? await fixture.multimerTemplateWeights() : undefined;
  return { multimer, embedding, template, multimerTemplate, extraStack, mainStack, structure,
    confidence, geometry, featureTables, paeBreaks };
}

type LoadedModelWeights = Awaited<ReturnType<typeof loadModelWeights>>;
let cachedModel: { readonly manifestUrl: string; readonly promise: Promise<LoadedModelWeights> } | undefined;

function modelWeights(manifestUrl: string, reporter: InferenceReporter): {
  readonly promise: Promise<LoadedModelWeights>; readonly cached: boolean;
} {
  if (cachedModel?.manifestUrl === manifestUrl) return { promise: cachedModel.promise, cached: true };
  const promise = loadModelWeights(manifestUrl, reporter);
  const entry = { manifestUrl, promise };
  cachedModel = entry;
  void promise.catch(() => { if (cachedModel === entry) cachedModel = undefined; });
  return { promise, cached: false };
}

/** Begin loading a model so it is ready, or in flight, when a prediction asks for it. */
export function prepareModel(manifestUrl: string, reporter: InferenceReporter): Promise<unknown> {
  return modelWeights(manifestUrl, reporter).promise;
}

/** Forget the in-memory model (and the persisted shards) so the next run downloads again. */
export async function clearInferenceCaches(): Promise<boolean> {
  cachedModel = undefined;
  return HttpTensorStore.clearPersistentCache();
}

let sharedPredictionDevice: GPUDevice | undefined;

/** Destroy the retained device, for example after an allocation failure. */
export function resetInferenceDevice(): void {
  sharedPredictionDevice?.destroy();
  sharedPredictionDevice = undefined;
}

async function predictionDevice(
  adapter: GPUAdapter, requirements: AlphaFoldDeviceRequirements, reporter: InferenceReporter,
): Promise<{ readonly device: GPUDevice; readonly cached: boolean }> {
  // A binding requirement past the adapter is met by windowing rather than by
  // a larger binding, so the cached device is compared against what the
  // adapter can actually give: otherwise every prediction rebuilds it.
  const wantedBinding = Math.min(
    requirements.maxStorageBufferBindingSize, adapter.limits.maxStorageBufferBindingSize);
  if (sharedPredictionDevice !== undefined
    && sharedPredictionDevice.limits.maxBufferSize >= requirements.maxBufferSize
    && sharedPredictionDevice.limits.maxStorageBufferBindingSize >= wantedBinding) {
    return { device: sharedPredictionDevice, cached: true };
  }
  resetInferenceDevice();
  const device = await requestAlphaFoldDevice(adapter, requirements);
  sharedPredictionDevice = device;
  device.addEventListener("uncapturederror", (event) => {
    const error = (event as GPUUncapturedErrorEvent).error;
    reporter.log(`WebGPU uncaptured error: ${error.message}`);
  });
  void device.lost.then((info) => {
    if (sharedPredictionDevice === device) sharedPredictionDevice = undefined;
    if (info.reason !== "destroyed") {
      reporter.log(`WebGPU device lost (${info.reason || "unknown"}): ${info.message || "no driver message"}.`);
    }
  });
  return { device, cached: false };
}

/** Whether an error message describes exhausted GPU memory, in which case the device is reset. */
export function isMemoryFailure(message: string): boolean {
  return /out.?of.?memory|allocation|device (?:was )?lost|memory budget/i.test(message);
}

export async function runInference(job: InferenceJob, reporter: InferenceReporter): Promise<InferenceOutcome> {
  reporter.stage("device", "active", "Requesting adapter");
  if (typeof navigator.gpu === "undefined") throw new Error("WebGPU is not available in this context");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("No WebGPU adapter is available");
  const adapterName = adapterDisplayName(adapter.info);
  const appleUnifiedMemory = isAppleUnifiedMemory(adapter);
  const compactMemoryPolicy = appleUnifiedMemory || job.compactPolicy;
  reporter.stage("device", "active", `${adapterName} · sizing buffers`);
  reporter.log(`GPU adapter: ${adapterName}`);
  reporter.log(`Browser platform: ${browserPlatform()}${compactMemoryPolicy ? " · compact memory policy" : ""}.`);
  reporter.log(`WebGPU features: ${[...adapter.features].sort().join(", ")}`);

  reporter.stage("model", "active", "Downloading one model");
  const modelStart = performance.now();
  const modelRequest = modelWeights(job.manifestUrl, reporter);
  if (modelRequest.cached) reporter.stage("model", "active", "Using in-memory model");
  const weights = await modelRequest.promise;
  const modelLoadMilliseconds = performance.now() - modelStart;
  const { input } = job;
  const { multimer: multimerModel, embedding, template, multimerTemplate, extraStack, mainStack, structure,
    confidence, geometry, featureTables, paeBreaks } = weights;
  if (input.multimer !== multimerModel) {
    throw new Error(input.multimer
      ? "Complex input requires an alphafold2_multimer_v3 model manifest"
      : "This is a Multimer-v3 manifest, but the current input is a monomer");
  }
  reporter.stage("model", "done", `${multimerModel ? "Multimer-v3" : "Model 1 PTM"} · `
    + `${modelRequest.cached ? "cached · " : ""}${formatSeconds(modelLoadMilliseconds)}`);
  reporter.log(`${modelRequest.cached ? "Reused" : "Loaded"} the reduced ${multimerModel ? "Multimer-v3" : "model-1"} `
    + `tensor bundle ${modelRequest.cached ? "from memory " : ""}in ${formatSeconds(modelLoadMilliseconds)}.`);

  reporter.stage("features", "active", "Parsing input"); reporter.status("Building AF2 features");
  const clusteredRows = Math.min(job.maxMsaSequences, input.depth);
  const extraRows = Math.max(1, Math.min(job.maxExtraSequences, Math.max(0, input.depth - clusteredRows)));
  // The model keeps its activations packed, monomer and Multimer alike; the
  // exact f32 storages exist only for the differential tests.
  const memoryOptions = {
    triangleWholeStorage: "f16" as const, msaStorage: "f16" as const, pairStorage: "f16" as const,
    ...(input.multimer ? { multimer: true, templateRows: multimerTemplate?.templateRows ?? 4 } : {}),
  };
  const memoryBudget = unifiedMemoryBudget(appleUnifiedMemory);
  const devicePlan = planMonomerDevice(
    adapter, input.sequence.length, clusteredRows, extraRows, memoryBudget, compactMemoryPolicy, memoryOptions,
  );
  reporter.log("Activations are stored as packed half precision: the MSA, the pair and the triangle projection.");
  reporter.log(`Estimated peak GPU allocations: ${formatMib(devicePlan.memory.estimatedPeakBytes)} `
    + `(${formatMib(devicePlan.memory.persistentBytes)} persistent, `
    + `${formatMib(devicePlan.memory.scratchBytes)} scratch, `
    + `${formatMib(devicePlan.memory.residentHeadroomBytes)} resident allowance).`);
  if (memoryBudget !== undefined) {
    reporter.log(`Apple unified-memory safety budget: ${formatMib(memoryBudget)}; bounded transitions and scratch pooling enabled.`);
    if (devicePlan.memory.estimatedPeakBytes > memoryBudget) {
      const suggestion = suggestMonomerRows(
        input.sequence.length, clusteredRows, extraRows, devicePlan.transitionMode, memoryBudget, memoryOptions,
      );
      // Past a certain length the pair and the triangle's projection are most
      // of the budget on their own, and the rows that still fit are too few to
      // fold with. Naming that beats suggesting a depth that would predict
      // badly, and single-sequence input is the honest way to spend what is
      // left.
      const degenerate = suggestion !== undefined
        && (suggestion.msaSequences < 32 || suggestion.extraSequences < 32);
      const advice = suggestion === undefined
        ? `This sequence is too long for the conservative safety budget: at ${input.sequence.length} `
          + "residues the pair representation alone is past it."
        : degenerate
          ? `At ${input.sequence.length} residues the pair representation is most of the budget, so only `
            + "a handful of alignment rows would fit. Predict from the single sequence instead, or "
            + "shorten the input."
          : `Set Clustered MSA rows to ${suggestion.msaSequences} and Extra MSA rows to `
            + `${suggestion.extraSequences} or lower.`;
      throw new RangeError(`Estimated peak GPU allocation ${formatMib(devicePlan.memory.estimatedPeakBytes)} `
        + `exceeds this Mac's ${formatMib(memoryBudget)} safety budget. ${advice}`);
    }
  }
  const deviceResult = await predictionDevice(adapter, devicePlan.requirements, reporter);
  const device = deviceResult.device;
  // An allocation past the budget fails with a message instead of paging the machine.
  setGpuMemoryBudget(device, memoryBudget);
  reporter.stage("device", "done", `${adapterName}${deviceResult.cached ? " · cached device" : ""}`);
  reporter.log(`GPU: ${adapterName}${deviceResult.cached ? " (reusing device and pipelines)" : ""}`);
  reporter.log(`WebGPU limits: invocations=${device.limits.maxComputeInvocationsPerWorkgroup} `
    + `workgroupStorage=${device.limits.maxComputeWorkgroupStorageSize} `
    + `storageBinding=${device.limits.maxStorageBufferBindingSize} buffer=${device.limits.maxBufferSize} `
    + `storageBuffersPerStage=${device.limits.maxStorageBuffersPerShaderStage}`);
  reporter.log(`Transition memory mode: ${devicePlan.transitionMode}.`);
  const featureOptions = {
    recycles: job.recycles, randomSeed: job.randomSeed,
    maxMsaSequences: job.maxMsaSequences, maxExtraSequences: job.maxExtraSequences,
  };
  const features = input.multimer
    ? input.alignmentMask === undefined
      ? iterateMultimerQueryOnlyFeatures(input.chains!, featureTables, featureOptions)
      : iterateMultimerA3mFeatures(input.chains!, input.a3m, input.alignmentMask, featureTables, featureOptions)
    : iterateA3mFeatures(input.a3m, featureTables, featureOptions);
  reporter.stage("features", "done", `${input.sequence.length} aa · ${input.depth} rows`);
  reporter.log(`Features: ${input.sequence.length} residues, ${input.multimer ? `${input.chains!.length} chains` : `A3M depth ${input.depth}`}.`);

  reporter.stage("inference", "active", `Recycle 0/${job.recycles}`); reporter.status("Running AlphaFold2 on WebGPU");
  const reportRecycle = (result: MonomerRecycleSummary | MultimerRecycleSummary, recycle: number): void => {
    reporter.stage("inference", "active", `Recycle ${recycle}/${job.recycles} · pLDDT ${result.confidence.meanPlddt.toFixed(1)}`);
    const multimerResult = result as MultimerRecycleSummary;
    reporter.log(`recycle=${recycle} pLDDT=${result.confidence.meanPlddt.toFixed(1)} `
      + `pTM=${result.confidence.ptm.toFixed(3)}${multimerResult.confidence.iptm === undefined
        ? "" : ` ipTM=${multimerResult.confidence.iptm.toFixed(3)}`} time=${formatSeconds(result.elapsedMilliseconds)} `
      + `${result.templateMilliseconds === undefined ? "" : `template=${result.templateMilliseconds.toFixed(0)}ms `}`
      + `trunkSubmissions=${result.trunkSubmissions.total}`);
    if (result.gpuProfile !== undefined) {
      for (const [stack, profile] of [
        ["extraMsa", result.gpuProfile.extraMsa],
        ["mainEvoformer", result.gpuProfile.mainEvoformer],
      ] as const) {
        const gpuMilliseconds = profile.entries.reduce((sum, entry) => sum + entry.nanoseconds, 0) / 1e6;
        reporter.log(`profile ${stack} block=${profile.block} method=${profile.method} `
          + `gpu=${gpuMilliseconds.toFixed(3)}ms wall=${profile.wallMilliseconds.toFixed(3)}ms`);
        for (const entry of profile.entries) reporter.log(`  ${entry.label} ${(entry.nanoseconds / 1e6).toFixed(3)}ms`);
      }
    }
    reporter.recycle(result, recycle);
  };
  // A recycle at 1500 residues is minutes of GPU work. Reporting each block as
  // the device finishes it keeps the stage line moving, so a long prediction
  // reads as slow rather than stalled.
  let lastProgress = 0;
  let blockStarted = performance.now();
  // Mean seconds a block of each stack has taken, which is what turns blocks
  // remaining into time remaining. The two differ by an order of magnitude.
  const blockSeconds = new Map<string, { total: number; count: number }>();
  const onProgress = (progress: MonomerProgress): void => {
    const now = performance.now();
    const measured = blockSeconds.get(progress.phase) ?? { total: 0, count: 0 };
    measured.total += (now - blockStarted) / 1000;
    measured.count += 1;
    blockSeconds.set(progress.phase, measured);
    blockStarted = now;
    if (progress.phase !== "structure" && progress.completed < progress.total
      && now - lastProgress < 1000) return;
    lastProgress = now;
    const where = progress.phase === "structure"
      ? "Structure module"
      : `${progress.phase === "extra-msa" ? "Extra MSA" : "Evoformer"} block `
        + `${progress.completed}/${progress.total}`;
    reporter.stage("inference", "active",
      `Recycle ${progress.recycle}/${job.recycles} · ${where}${remaining(progress)}`);
  };
  const remaining = (progress: MonomerProgress): string => {
    if (blockCounts === undefined) return "";
    return remainingPhrase(remainingTrunkSeconds(progress, {
      extraBlocks: blockCounts.extra, mainBlocks: blockCounts.main, recycles: job.recycles,
    }, {
      extraSeconds: mean(blockSeconds.get("extra-msa")),
      mainSeconds: mean(blockSeconds.get("evoformer")),
    }));
  };
  const mean = (measured: { total: number; count: number } | undefined): number | undefined =>
    measured === undefined ? undefined : measured.total / measured.count;
  let blockCounts: { readonly extra: number; readonly main: number } | undefined;
  const modelOptions = {
    onProgress,
    compactTransitions: devicePlan.transitionMode === "chunked",
    profile: job.profile !== undefined,
    profileRecycle: job.profile?.recycle ?? 0,
    profileExtraMsaBlock: job.profile?.extraMsaBlock ?? 0,
    profileMainEvoformerBlock: job.profile?.mainEvoformerBlock ?? 0,
    ...memoryOptions,
  } as const;
  const commonWeights = {
    embedding, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
  };
  blockCounts = { extra: extraStack.length, main: mainStack.length };
  // The first block's time is measured from here, not from when the callback
  // was built: loading the model and building features happen in between.
  blockStarted = performance.now();
  const prediction: MonomerPrediction | MultimerPrediction = input.multimer
    ? await new AlphaFoldMultimerGpu(device, modelOptions).predict(
      features as Iterable<MultimerRecycleFeatures> & { readonly length: number },
      { ...commonWeights, multimerTemplate: multimerTemplate! },
      paeBreaks, reportRecycle,
    )
    : await new AlphaFoldMonomerGpu(device, modelOptions).predict(features, {
      ...commonWeights, template: template!,
    }, paeBreaks, reportRecycle);
  reporter.stage("inference", "done", formatSeconds(prediction.elapsedMilliseconds));
  reporter.log(`Measured allocator peak: ${formatMib(prediction.memory.combinedPeakResidentBytes)} combined resident `
    + `(${formatMib(prediction.memory.mainPeakResidentBytes)} trunk).`);
  return { prediction, modelLoadMilliseconds, adapterName };
}
