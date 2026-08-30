import { AlphaFoldMonomerGpu, type MonomerPrediction, type MonomerRecycleResult } from "../src/model/monomer.js";
import { makeA3mFeatures } from "../src/input/a3m-features.js";
import { parseA3m } from "../src/input/a3m.js";
import { generateMmseqs2Msa, type Mmseqs2MsaProgress, type Mmseqs2MsaResult } from "../src/input/mmseqs2-api.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import type { TensorDownloadProgress } from "../src/reference/http-tensor-store.js";
import {
  planMonomerDevice, requestAlphaFoldDevice, suggestMonomerRows, type AlphaFoldDeviceRequirements,
} from "../src/runtime/device.js";
import { createDeterministicTriangleInput } from "../src/testing/deterministic-input.js";
import { triangleMultiplicationOutgoingReference } from "../src/triangle/cpu-reference.js";
import { errorMetrics, type Precision } from "../src/triangle/types.js";
import { TriangleMultiplicationOutgoingGpu } from "../src/triangle/webgpu.js";
import { confidenceJson, predictionToPdb, safeJobName } from "./prediction-results.js";
import { drawMsaCoverage } from "./msa-plot.js";

interface BrowserResult {
  readonly elapsedMilliseconds: number;
  readonly peakBytes: number;
  readonly meanAbsoluteError: number;
  readonly maxAbsoluteError: number;
}

interface Viewer3D {
  addModel(data: string, format: string): void;
  setStyle(selection: object, style: object): void;
  zoomTo(): void;
  render(): void;
  resize(): void;
}

interface ThreeDmolApi { createViewer(element: HTMLElement, options: object): Viewer3D; }

declare global {
  interface Window {
    __AFWEBGPU_RESULT__?: BrowserResult;
    __AFWEBGPU_PREDICTION__?: {
      meanPlddt: number; ptm: number; elapsedMilliseconds: number; modelLoadMilliseconds: number;
      recycles: readonly {
        elapsedMilliseconds: number; meanPlddt: number; ptm: number;
        trunkSubmissions: MonomerRecycleResult["trunkSubmissions"];
      }[];
    };
    $3Dmol?: ThreeDmolApi;
  }
}

const element = <T extends HTMLElement>(id: string): T => {
  const value = document.getElementById(id);
  if (value === null) throw new Error(`missing element #${id}`);
  return value as T;
};

const parameter = (name: string, fallback: string): string => new URLSearchParams(location.search).get(name) ?? fallback;
const normalizedSequence = (): string => element<HTMLTextAreaElement>("sequence").value.replace(/\s+/g, "").toUpperCase();
const formatSeconds = (milliseconds: number): string => `${(milliseconds / 1000).toFixed(2)} s`;
const formatMib = (bytes: number): string => `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
const stageOrder = ["device", "msa", "model", "features", "inference", "results"] as const;
type Stage = typeof stageOrder[number];
type StageState = "active" | "done" | "error";

let currentPdb = "";
let currentScores = "";
let generatedMsa: { readonly key: string; readonly result: Mmseqs2MsaResult } | undefined;
let lastMsaStatus = "";
let viewerLoader: Promise<ThreeDmolApi> | undefined;
let viewerResizeObserver: ResizeObserver | undefined;
let sharedPredictionDevice: GPUDevice | undefined;

function unifiedMemoryBudget(adapter: GPUAdapter): number | undefined {
  const identity = `${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.device} ${adapter.info.description}`
    .toLowerCase();
  if (!identity.includes("apple")) return undefined;
  const deviceMemory = (navigator as Navigator & { readonly deviceMemory?: number }).deviceMemory;
  if (deviceMemory === undefined || !Number.isFinite(deviceMemory) || deviceMemory <= 0) return undefined;
  // Apple GPUs share system RAM. navigator.deviceMemory is privacy-rounded and
  // capped in Chromium, so reserve 30% while relying on the estimator's own
  // calibrated 2.5x pooling/implementation safety allowance.
  return Math.floor(deviceMemory * 1024 ** 3 * 0.70);
}

async function loadModelWeights(manifestValue: string) {
  const store = await HttpTensorStore.open(manifestValue, updateModelProgress);
  const fixture = AlphaFoldFixture.fromStore(store);
  return Promise.all([
    fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraStackWeights(), fixture.mainStackWeights(),
    fixture.structureWeights(), fixture.confidenceWeights(), fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
    fixture.tensor("confidencePaeBreaks"),
  ] as const);
}

type LoadedModelWeights = Awaited<ReturnType<typeof loadModelWeights>>;
let cachedModel: { readonly manifestValue: string; readonly promise: Promise<LoadedModelWeights> } | undefined;

function modelWeights(manifestValue: string): {
  readonly promise: Promise<LoadedModelWeights>; readonly cached: boolean;
} {
  if (cachedModel?.manifestValue === manifestValue) return { promise: cachedModel.promise, cached: true };
  const promise = loadModelWeights(manifestValue);
  const entry = { manifestValue, promise };
  cachedModel = entry;
  void promise.catch(() => { if (cachedModel === entry) cachedModel = undefined; });
  return { promise, cached: false };
}

async function predictionDevice(adapter: GPUAdapter, requirements: AlphaFoldDeviceRequirements): Promise<{
  readonly device: GPUDevice; readonly cached: boolean;
}> {
  if (sharedPredictionDevice !== undefined
    && sharedPredictionDevice.limits.maxBufferSize >= requirements.maxBufferSize
    && sharedPredictionDevice.limits.maxStorageBufferBindingSize >= requirements.maxStorageBufferBindingSize) {
    return { device: sharedPredictionDevice, cached: true };
  }
  sharedPredictionDevice?.destroy();
  sharedPredictionDevice = undefined;
  const device = await requestAlphaFoldDevice(adapter, requirements);
  sharedPredictionDevice = device;
  void device.lost.then(() => { if (sharedPredictionDevice === device) sharedPredictionDevice = undefined; });
  return { device, cached: false };
}

function stage(stageName: Stage, state: StageState, detail: string): void {
  const item = document.querySelector<HTMLElement>(`[data-stage="${stageName}"]`);
  if (item === null) throw new Error(`missing stage ${stageName}`);
  item.dataset.state = state;
  const small = item.querySelector("small");
  if (small !== null) small.textContent = detail;
}

function resetStages(): void {
  for (const name of stageOrder) {
    const item = document.querySelector<HTMLElement>(`[data-stage="${name}"]`);
    if (item === null) continue;
    delete item.dataset.state;
    const small = item.querySelector("small");
    if (small !== null) small.textContent = "Waiting";
  }
  const progress = element<HTMLProgressElement>("model-progress");
  progress.value = 0;
  element<HTMLElement>("model-progress-wrap").hidden = true;
  element<HTMLElement>("model-progress-label").textContent = "0%";
}

function updateModelProgress(value: TensorDownloadProgress): void {
  const fraction = value.totalBytes === 0 ? 0 : value.loadedBytes / value.totalBytes;
  element<HTMLElement>("model-progress-wrap").hidden = false;
  element<HTMLProgressElement>("model-progress").value = fraction;
  const loaded = value.loadedBytes / 1024 / 1024;
  const total = value.totalBytes / 1024 / 1024;
  element<HTMLElement>("model-progress-label").textContent = `${Math.floor(fraction * 100)}%`;
  stage("model", "active", `${loaded.toFixed(0)} / ${total.toFixed(0)} MiB · ${value.loadedTensors}/${value.totalTensors}`);
}

function updateMsaProgress(value: Mmseqs2MsaProgress): void {
  const seconds = Math.round(value.elapsedMilliseconds / 1000);
  const labels: Readonly<Record<string, string>> = {
    submitting: "Submitting query", queued: "Queued", running: "Searching databases",
    downloading: "Downloading alignment", complete: "Alignment ready", retrying: "Server busy; retrying",
  };
  const label = labels[value.phase] ?? value.status;
  stage("msa", value.phase === "complete" ? "done" : "active", `${label} · ${seconds}s`);
  setPredictionStatus(label);
  const statusKey = `${value.phase}:${value.status}`;
  if (statusKey !== lastMsaStatus) {
    log(`MMseqs2: ${label}${value.ticket === undefined ? "" : ` · ticket ${value.ticket}`}.`);
    lastMsaStatus = statusKey;
  }
}

function setPredictionStatus(text: string, state = "running"): void {
  const status = element<HTMLDivElement>("prediction-status");
  status.dataset.state = state;
  status.textContent = text;
}

function log(text: string, append = true): void {
  const target = element<HTMLPreElement>("run-log");
  target.textContent = append ? `${target.textContent ?? ""}${target.textContent ? "\n" : ""}${text}` : text;
  target.scrollTop = target.scrollHeight;
}

function download(filename: string, contents: string, type: string): void {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement("a");
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function confidenceLabel(score: number): string {
  if (score >= 90) return "Very high confidence";
  if (score >= 70) return "Confident";
  if (score >= 50) return "Low confidence";
  return "Very low confidence";
}

function plddtColor(value: number): string {
  if (value >= 90) return "#187bd1";
  if (value >= 70) return "#56b9dc";
  if (value >= 50) return "#f2c94c";
  return "#ef6a62";
}

function drawPlddt(values: Float32Array): void {
  const canvas = element<HTMLCanvasElement>("plddt-plot");
  const context = canvas.getContext("2d");
  if (context === null) return;
  const { width, height } = canvas; const left = 46; const bottom = 30; const top = 18; const right = 12;
  context.clearRect(0, 0, width, height); context.fillStyle = "#ffffff"; context.fillRect(0, 0, width, height);
  context.strokeStyle = "#dddddd"; context.fillStyle = "#777777"; context.font = "12px Roboto Mono";
  for (const tick of [0, 50, 70, 90, 100]) {
    const y = top + (100 - tick) / 100 * (height - top - bottom);
    context.beginPath(); context.moveTo(left, y); context.lineTo(width - right, y); context.stroke();
    context.fillText(String(tick), 8, y + 4);
  }
  const barWidth = (width - left - right) / values.length;
  for (let index = 0; index < values.length; index += 1) {
    const value = Math.max(0, Math.min(100, values[index]!));
    const barHeight = value / 100 * (height - top - bottom);
    context.fillStyle = plddtColor(value);
    context.fillRect(left + index * barWidth, height - bottom - barHeight, Math.max(1, barWidth), barHeight);
  }
  context.fillStyle = "#777777"; context.fillText("Residue", width / 2 - 25, height - 7);
}

function paeColor(value: number, maximum: number): [number, number, number] {
  const fraction = Math.max(0, Math.min(1, value / Math.max(maximum, 1e-6)));
  if (fraction < .5) {
    const t = fraction * 2; return [Math.round(28 + t * 106), Math.round(91 + t * 126), Math.round(164 + t * 42)];
  }
  const t = (fraction - .5) * 2; return [Math.round(134 + t * 105), Math.round(217 - t * 150), Math.round(206 - t * 122)];
}

function drawPae(values: Float32Array, length: number, maximum: number): void {
  const canvas = element<HTMLCanvasElement>("pae-plot"); const context = canvas.getContext("2d");
  if (context === null) return;
  const image = context.createImageData(length, length);
  for (let index = 0; index < values.length; index += 1) {
    const [red, green, blue] = paeColor(values[index]!, maximum); const output = index * 4;
    image.data[output] = red; image.data[output + 1] = green; image.data[output + 2] = blue; image.data[output + 3] = 255;
  }
  const temporary = document.createElement("canvas"); temporary.width = length; temporary.height = length;
  temporary.getContext("2d")?.putImageData(image, 0, 0);
  context.imageSmoothingEnabled = false; context.clearRect(0, 0, canvas.width, canvas.height);
  context.drawImage(temporary, 0, 0, canvas.width, canvas.height);
}

function loadViewer(): Promise<ThreeDmolApi> {
  if (window.$3Dmol !== undefined) return Promise.resolve(window.$3Dmol);
  viewerLoader ??= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "https://3dmol.org/build/3Dmol.js"; script.async = true;
    script.onload = () => window.$3Dmol === undefined ? reject(new Error("3Dmol did not initialize")) : resolve(window.$3Dmol);
    script.onerror = () => reject(new Error("Could not load the optional 3D structure viewer"));
    document.head.append(script);
  });
  return viewerLoader;
}

async function showStructure(pdb: string): Promise<void> {
  const container = element<HTMLDivElement>("structure-viewer");
  viewerResizeObserver?.disconnect();
  container.replaceChildren();
  try {
    const api = await loadViewer();
    const viewer = api.createViewer(container, { backgroundColor: "#ffffff" });
    viewer.addModel(pdb, "pdb");
    viewer.setStyle({}, { cartoon: { colorscheme: { prop: "b", gradient: "roygb", min: 50, max: 90 } } });
    viewer.zoomTo(); viewer.render();
    viewerResizeObserver = new ResizeObserver(() => { viewer.resize(); viewer.render(); });
    viewerResizeObserver.observe(container);
  } catch (error) {
    const message = document.createElement("p");
    message.textContent = `${error instanceof Error ? error.message : String(error)}. The PDB download is still available.`;
    container.append(message);
  }
}

function showResults(prediction: MonomerPrediction, sequence: string, depth: number, jobName: string, a3m: string,
  modelLoadMilliseconds: number): void {
  const confidence = prediction.final.confidence;
  currentPdb = predictionToPdb(sequence, prediction.final.structure, confidence.plddt);
  currentScores = confidenceJson(sequence, confidence);
  element<HTMLElement>("results-section").hidden = false;
  element<HTMLElement>("mean-plddt").textContent = confidence.meanPlddt.toFixed(1);
  element<HTMLElement>("plddt-label").textContent = confidenceLabel(confidence.meanPlddt);
  element<HTMLElement>("ptm").textContent = confidence.ptm.toFixed(3);
  element<HTMLElement>("result-length").textContent = String(sequence.length);
  element<HTMLElement>("msa-depth").textContent = String(depth);
  element<HTMLElement>("total-time").textContent = `WebGPU inference ${formatSeconds(prediction.elapsedMilliseconds)}`;
  const rows = element<HTMLTableSectionElement>("recycle-results"); rows.replaceChildren();
  prediction.recycles.forEach((result, index) => {
    const row = document.createElement("tr");
    for (const value of [String(index), result.confidence.meanPlddt.toFixed(1), result.confidence.ptm.toFixed(3), formatSeconds(result.elapsedMilliseconds)]) {
      const cell = document.createElement("td"); cell.textContent = value; row.append(cell);
    }
    rows.append(row);
  });
  drawPlddt(confidence.plddt);
  drawPae(confidence.predictedAlignedError, sequence.length, confidence.maxPredictedAlignedError);
  const msa = drawMsaCoverage(element<HTMLCanvasElement>("msa-plot"), a3m);
  const meanCoverage = Array.from(msa.coverage).reduce((sum, value) => sum + value, 0) / msa.length;
  element<HTMLElement>("msa-plot-summary").textContent =
    `${msa.depth.toLocaleString()} sequences · mean ${meanCoverage.toFixed(0)} sequences/position`;
  void showStructure(currentPdb);
  element<HTMLButtonElement>("download-pdb").onclick = () => download(`${jobName}_unrelaxed_model_1.pdb`, currentPdb, "chemical/x-pdb");
  element<HTMLButtonElement>("download-scores").onclick = () => download(`${jobName}_scores.json`, currentScores, "application/json");
  window.__AFWEBGPU_PREDICTION__ = {
    meanPlddt: confidence.meanPlddt, ptm: confidence.ptm,
    elapsedMilliseconds: prediction.elapsedMilliseconds, modelLoadMilliseconds,
    recycles: prediction.recycles.map((result) => ({
      elapsedMilliseconds: result.elapsedMilliseconds, meanPlddt: result.confidence.meanPlddt,
      ptm: result.confidence.ptm, trunkSubmissions: result.trunkSubmissions,
    })),
  };
}

async function predictionInput(): Promise<{ a3m: string; sequence: string; depth: number }> {
  const mode = element<HTMLSelectElement>("input-mode").value;
  if (mode === "single") {
    const sequence = normalizedSequence();
    if (!/^[ARNDCQEGHILKMFPSTWYVX]+$/.test(sequence)) throw new Error("Sequence must contain only standard amino-acid letters or X");
    stage("msa", "done", "Single sequence");
    return { a3m: `>query\n${sequence}\n`, sequence, depth: 1 };
  }
  if (mode === "mmseqs2") {
    const sequence = normalizedSequence();
    const apiUrl = element<HTMLInputElement>("msa-api-url").value.trim();
    if (apiUrl === "") throw new Error("An MMseqs2 API URL is required");
    try { localStorage.setItem("afwebgpu.msaApiUrl", apiUrl); } catch { /* storage may be unavailable */ }
    const key = `${apiUrl}\n${sequence}`;
    if (generatedMsa?.key === key) {
      stage("msa", "done", `${generatedMsa.result.depth} rows · cached`); log("Reusing the generated MMseqs2 alignment.");
      return { a3m: generatedMsa.result.a3m, sequence, depth: generatedMsa.result.depth };
    }
    const result = await generateMmseqs2Msa(sequence, { apiUrl, onProgress: updateMsaProgress });
    generatedMsa = { key, result };
    stage("msa", "done", `${result.depth} rows · ${formatSeconds(result.elapsedMilliseconds)}`);
    const downloadButton = element<HTMLButtonElement>("download-msa"); downloadButton.hidden = false;
    downloadButton.onclick = () => download(`${safeJobName(element<HTMLInputElement>("job-name").value)}.a3m`, result.a3m, "text/plain");
    return { a3m: result.a3m, sequence, depth: result.depth };
  }
  const file = element<HTMLInputElement>("a3m-file").files?.[0];
  if (file === undefined) throw new Error("Choose a custom A3M file first");
  const a3m = await file.text(); const parsed = parseA3m(a3m);
  stage("msa", "done", `${parsed.depth} uploaded rows`);
  return { a3m, sequence: parsed.query, depth: parsed.depth };
}

element<HTMLFormElement>("prediction-form").addEventListener("submit", (event) => { event.preventDefault(); void runPrediction(); });

async function runPrediction(): Promise<void> {
  const button = element<HTMLButtonElement>("predict"); button.disabled = true;
  const clearCacheButton = element<HTMLButtonElement>("clear-model-cache"); clearCacheButton.disabled = true;
  element<HTMLElement>("results-section").hidden = true; resetStages(); log("Starting prediction…", false); lastMsaStatus = "";
  try {
    stage("device", "active", "Requesting adapter"); setPredictionStatus("Preparing WebGPU");
    if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable. Use a current Chrome or Edge browser on a supported GPU.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("No compatible WebGPU adapter was found");
    const adapterName = adapter.info.description || adapter.info.device || adapter.info.vendor || "WebGPU adapter";
    stage("device", "active", `${adapterName} · sizing buffers`);
    log(`GPU adapter: ${adapterName}`);
    log(`WebGPU features: ${[...adapter.features].sort().join(", ")}`);

    stage("msa", "active", "Preparing input");
    stage("model", "active", "Downloading one model"); setPredictionStatus("Preparing alignment and model");
    const manifestValue = element<HTMLInputElement>("model-url").value.trim();
    if (manifestValue === "") throw new Error("A model manifest URL is required");
    localStorage.setItem("afwebgpu.modelUrl", manifestValue);
    const modelStart = performance.now();
    const inputPromise = predictionInput();
    const modelRequest = modelWeights(manifestValue);
    if (modelRequest.cached) stage("model", "active", "Using in-memory model");
    const measuredModel = modelRequest.promise.then((weights) => ({
      weights, elapsedMilliseconds: performance.now() - modelStart,
    }));
    const [input, loadedModel] = await Promise.all([inputPromise, measuredModel]);
    const { weights, elapsedMilliseconds: modelLoadMilliseconds } = loadedModel;
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables, paeBreaks] = weights;
    stage("model", "done", `Model 1 PTM · ${modelRequest.cached ? "cached · " : ""}${formatSeconds(modelLoadMilliseconds)}`);
    log(`${modelRequest.cached ? "Reused" : "Loaded"} the reduced model-1 tensor bundle `
      + `${modelRequest.cached ? "from memory " : ""}in ${formatSeconds(modelLoadMilliseconds)}.`);

    stage("features", "active", "Parsing input"); setPredictionStatus("Building AF2 features");
    const requestedMaxMsa = element<HTMLInputElement>("max-msa").valueAsNumber;
    const requestedMaxExtra = element<HTMLInputElement>("max-extra").valueAsNumber;
    const clusteredRows = Math.min(requestedMaxMsa, input.depth);
    const extraRows = Math.max(1, Math.min(requestedMaxExtra, Math.max(0, input.depth - clusteredRows)));
    const memoryBudget = unifiedMemoryBudget(adapter);
    const devicePlan = planMonomerDevice(
      adapter, input.sequence.length, clusteredRows, extraRows, memoryBudget,
    );
    log(`Estimated peak GPU allocations: ${formatMib(devicePlan.memory.estimatedPeakBytes)} `
      + `(${formatMib(devicePlan.memory.persistentBytes)} persistent, `
      + `${formatMib(devicePlan.memory.scratchBytes)} scratch).`);
    if (memoryBudget !== undefined) {
      log(`Apple unified-memory safety budget: ${formatMib(memoryBudget)}.`);
      if (devicePlan.memory.estimatedPeakBytes > memoryBudget) {
        const suggestion = suggestMonomerRows(
          input.sequence.length, clusteredRows, extraRows, devicePlan.transitionMode, memoryBudget,
        );
        const advice = suggestion === undefined
          ? "This sequence is too large for the conservative safety budget even with one MSA row."
          : `Set Clustered MSA rows to ${suggestion.msaSequences} and Extra MSA rows to `
            + `${suggestion.extraSequences} or lower.`;
        throw new RangeError(`Estimated peak GPU allocation ${formatMib(devicePlan.memory.estimatedPeakBytes)} `
          + `exceeds this Mac's ${formatMib(memoryBudget)} safety budget. ${advice}`);
      }
    }
    const deviceResult = await predictionDevice(adapter, devicePlan.requirements);
    const device = deviceResult.device;
    stage("device", "done", `${adapterName}${deviceResult.cached ? " · cached device" : ""}`);
    log(`GPU: ${adapterName}${deviceResult.cached ? " (reusing device and pipelines)" : ""}`);
    log(`WebGPU limits: invocations=${device.limits.maxComputeInvocationsPerWorkgroup} `
      + `workgroupStorage=${device.limits.maxComputeWorkgroupStorageSize} `
      + `storageBinding=${device.limits.maxStorageBufferBindingSize} buffer=${device.limits.maxBufferSize}`);
    log(`Transition memory mode: ${devicePlan.transitionMode}.`);
    const featureOptions = {
      recycles: Number(element<HTMLSelectElement>("recycles").value),
      randomSeed: element<HTMLInputElement>("seed").valueAsNumber,
      maxMsaSequences: requestedMaxMsa,
      maxExtraSequences: requestedMaxExtra,
    };
    const features = makeA3mFeatures(input.a3m, featureTables, featureOptions);
    stage("features", "done", `${input.sequence.length} aa · ${input.depth} rows`); log(`Features: ${input.sequence.length} residues, A3M depth ${input.depth}.`);

    stage("inference", "active", `Recycle 0/${featureOptions.recycles}`); setPredictionStatus("Running AlphaFold2 on WebGPU");
    const reportRecycle = (result: MonomerRecycleResult, recycle: number): void => {
      stage("inference", "active", `Recycle ${recycle}/${featureOptions.recycles} · pLDDT ${result.confidence.meanPlddt.toFixed(1)}`);
      log(`recycle=${recycle} pLDDT=${result.confidence.meanPlddt.toFixed(1)} `
        + `pTM=${result.confidence.ptm.toFixed(3)} time=${formatSeconds(result.elapsedMilliseconds)} `
        + `trunkSubmissions=${result.trunkSubmissions.total}`);
      if (result.gpuProfile !== undefined) {
        for (const [stack, profile] of [
          ["extraMsa", result.gpuProfile.extraMsa],
          ["mainEvoformer", result.gpuProfile.mainEvoformer],
        ] as const) {
          const gpuMilliseconds = profile.entries.reduce((sum, entry) => sum + entry.nanoseconds, 0) / 1e6;
          log(`profile ${stack} block=${profile.block} method=${profile.method} `
            + `gpu=${gpuMilliseconds.toFixed(3)}ms wall=${profile.wallMilliseconds.toFixed(3)}ms`);
          for (const entry of profile.entries) log(`  ${entry.label} ${(entry.nanoseconds / 1e6).toFixed(3)}ms`);
        }
      }
    };
    const prediction = await new AlphaFoldMonomerGpu(device, {
      compactTransitions: devicePlan.transitionMode === "chunked",
      profile: parameter("profile", "0") === "1",
      profileRecycle: Number(parameter("profileRecycle", "0")),
      profileExtraMsaBlock: Number(parameter("profileExtraBlock", "0")),
      profileMainEvoformerBlock: Number(parameter("profileMainBlock", "0")),
    }).predict(features, {
      embedding, template, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
    }, paeBreaks, reportRecycle);
    stage("inference", "done", formatSeconds(prediction.elapsedMilliseconds));
    log(`Measured allocator peak: ${formatMib(prediction.memory.peakResidentBytes)} resident `
      + `(${prediction.memory.bufferCount} GPU buffers).`);

    stage("results", "active", "Rendering"); setPredictionStatus("Preparing results");
    const jobName = safeJobName(element<HTMLInputElement>("job-name").value);
    showResults(prediction, input.sequence, input.depth, jobName, input.a3m, modelLoadMilliseconds);
    stage("results", "done", "Ready"); setPredictionStatus("Prediction complete", "passed");
    log(`Finished in ${formatSeconds(prediction.elapsedMilliseconds)}.`);
    element<HTMLElement>("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const active = stageOrder.find((name) => document.querySelector<HTMLElement>(`[data-stage="${name}"]`)?.dataset.state === "active");
    if (active !== undefined) stage(active, "error", "Failed");
    const memoryFailure = /out.?of.?memory|allocation|device (?:was )?lost/i.test(message);
    if (memoryFailure) {
      sharedPredictionDevice?.destroy(); sharedPredictionDevice = undefined;
      log("The WebGPU device could not retain this allocation set. Reduce clustered/extra MSA rows or sequence length, then retry.");
    }
    setPredictionStatus("Prediction failed", "failed"); log(error instanceof Error ? error.stack ?? message : message);
  } finally { button.disabled = false; clearCacheButton.disabled = false; }
}

const inputMode = element<HTMLSelectElement>("input-mode");
function updateInputMode(): void {
  const custom = inputMode.value === "custom"; const remote = inputMode.value === "mmseqs2";
  element<HTMLElement>("sequence-field").hidden = custom; element<HTMLElement>("a3m-field").hidden = !custom;
  element<HTMLElement>("sequence-length").textContent = custom ? "A3M input" : `${normalizedSequence().length} residues`;
  element<HTMLElement>("sequence-hint").textContent = remote
    ? "One monomer. MMseqs2 mode sends this sequence to the public ColabFold MSA server."
    : "One monomer, using the 20 standard amino acids or X. This input stays on the device.";
  element<HTMLElement>("predict-label").textContent = remote ? "Generate MSA & predict" : "Run prediction";
}
inputMode.addEventListener("change", updateInputMode);
element<HTMLTextAreaElement>("sequence").addEventListener("input", () => { generatedMsa = undefined;
  element<HTMLButtonElement>("download-msa").hidden = true; updateInputMode(); });
element<HTMLInputElement>("a3m-file").addEventListener("change", (event) => {
  const file = (event.currentTarget as HTMLInputElement).files?.[0];
  element<HTMLElement>("a3m-file-name").textContent = file?.name ?? "Choose an A3M file";
});
try { element<HTMLInputElement>("model-url").value = localStorage.getItem("afwebgpu.modelUrl") ?? "./model/manifest.json"; } catch { /* storage may be unavailable */ }
try { element<HTMLInputElement>("msa-api-url").value = localStorage.getItem("afwebgpu.msaApiUrl") ?? "https://api.colabfold.com"; } catch { /* storage may be unavailable */ }
element<HTMLButtonElement>("clear-model-cache").addEventListener("click", () => { void (async () => {
  const button = element<HTMLButtonElement>("clear-model-cache"); button.disabled = true;
  try {
    const removed = await HttpTensorStore.clearPersistentCache();
    cachedModel = undefined;
    sharedPredictionDevice?.destroy(); sharedPredictionDevice = undefined;
    log(removed ? "Cleared the persistent and in-memory model caches." : "Cleared the in-memory model cache; no persistent cache was present.", false);
  } finally { button.disabled = false; }
})(); });
updateInputMode();

async function checkWebGpu(): Promise<void> {
  const summary = element<HTMLDivElement>("gpu-summary");
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null || adapter === undefined) throw new Error("No WebGPU adapter found");
    const name = adapter.info.description || adapter.info.device || adapter.info.vendor || "compatible GPU";
    summary.dataset.state = "ready"; summary.lastElementChild!.textContent = `WebGPU ready · ${name}`;
  } catch (error) {
    summary.dataset.state = "failed"; summary.lastElementChild!.textContent = error instanceof Error ? error.message : String(error);
  }
}
void checkWebGpu();

// Keep the small, model-free differential kernel diagnostic available for browser/CI checks.
const diagnosticForm = element<HTMLFormElement>("controls");
const diagnosticStatus = element<HTMLDivElement>("status");
const diagnosticResult = element<HTMLPreElement>("result");
element<HTMLInputElement>("length").value = parameter("length", "8");
element<HTMLInputElement>("cz").value = parameter("cz", "16");
element<HTMLInputElement>("hidden").value = parameter("hidden", "16");
element<HTMLSelectElement>("precision").value = parameter("precision", "f32");
diagnosticForm.addEventListener("submit", (event) => { event.preventDefault(); void runDiagnostic(); });

async function runDiagnostic(): Promise<void> {
  diagnosticStatus.dataset.state = "running"; diagnosticStatus.textContent = "Running…";
  diagnosticResult.textContent = "Requesting a WebGPU adapter.";
  let device: GPUDevice | undefined;
  try {
    if (navigator.gpu === undefined) throw new Error("WebGPU is not available in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("No compatible WebGPU adapter was found");
    const precision = element<HTMLSelectElement>("precision").value as Precision;
    if (precision === "f16" && !adapter.features.has("shader-f16")) throw new Error("This adapter does not expose shader-f16");
    device = await adapter.requestDevice({ requiredFeatures: precision === "f16" ? ["shader-f16"] : [] });
    const shape = { length: element<HTMLInputElement>("length").valueAsNumber, cZ: element<HTMLInputElement>("cz").valueAsNumber, cHidden: element<HTMLInputElement>("hidden").valueAsNumber };
    const input = createDeterministicTriangleInput(shape, 29);
    const expected = triangleMultiplicationOutgoingReference(input);
    const gpuResult = await new TriangleMultiplicationOutgoingGpu(device).run(input, { precision });
    const errors = errorMetrics(gpuResult.output, expected);
    window.__AFWEBGPU_RESULT__ = { elapsedMilliseconds: gpuResult.elapsedMilliseconds, peakBytes: gpuResult.memory.peakBytes, meanAbsoluteError: errors.meanAbsoluteError, maxAbsoluteError: errors.maxAbsoluteError };
    diagnosticStatus.dataset.state = "passed"; diagnosticStatus.textContent = "Differential test passed";
    diagnosticResult.textContent = JSON.stringify({ shape, precision, ...window.__AFWEBGPU_RESULT__ }, null, 2);
  } catch (error) {
    diagnosticStatus.dataset.state = "failed"; diagnosticStatus.textContent = "Run failed";
    diagnosticResult.textContent = error instanceof Error ? error.stack ?? error.message : String(error);
  } finally { device?.destroy(); }
}

if (parameter("autorun", "0") === "1") void runDiagnostic();
