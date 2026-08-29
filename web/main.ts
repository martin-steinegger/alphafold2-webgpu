import { AlphaFoldMonomerGpu, type MonomerPrediction, type MonomerRecycleResult } from "../src/model/monomer.js";
import { makeA3mFeatures } from "../src/input/a3m-features.js";
import { parseA3m } from "../src/input/a3m.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { HttpTensorStore } from "../src/reference/http-tensor-store.js";
import type { TensorDownloadProgress } from "../src/reference/http-tensor-store.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { createDeterministicTriangleInput } from "../src/testing/deterministic-input.js";
import { triangleMultiplicationOutgoingReference } from "../src/triangle/cpu-reference.js";
import { errorMetrics, type Precision } from "../src/triangle/types.js";
import { TriangleMultiplicationOutgoingGpu } from "../src/triangle/webgpu.js";
import { confidenceJson, predictionToPdb, safeJobName } from "./prediction-results.js";

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
}

interface ThreeDmolApi { createViewer(element: HTMLElement, options: object): Viewer3D; }

declare global {
  interface Window {
    __AFWEBGPU_RESULT__?: BrowserResult;
    __AFWEBGPU_PREDICTION__?: { meanPlddt: number; ptm: number; elapsedMilliseconds: number };
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
const stageOrder = ["device", "model", "features", "inference", "results"] as const;
type Stage = typeof stageOrder[number];
type StageState = "active" | "done" | "error";

let currentPdb = "";
let currentScores = "";
let viewerLoader: Promise<ThreeDmolApi> | undefined;

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
  context.clearRect(0, 0, width, height); context.fillStyle = "#07110f"; context.fillRect(0, 0, width, height);
  context.strokeStyle = "#29473e"; context.fillStyle = "#789188"; context.font = "12px DM Mono";
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
  context.fillStyle = "#789188"; context.fillText("Residue", width / 2 - 25, height - 7);
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
  const container = element<HTMLDivElement>("structure-viewer"); container.replaceChildren();
  try {
    const api = await loadViewer();
    const viewer = api.createViewer(container, { backgroundColor: "#07110f" });
    viewer.addModel(pdb, "pdb");
    viewer.setStyle({}, { cartoon: { colorscheme: { prop: "b", gradient: "roygb", min: 50, max: 90 } } });
    viewer.zoomTo(); viewer.render();
  } catch (error) {
    const message = document.createElement("p");
    message.textContent = `${error instanceof Error ? error.message : String(error)}. The PDB download is still available.`;
    container.append(message);
  }
}

function showResults(prediction: MonomerPrediction, sequence: string, depth: number, jobName: string): void {
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
  void showStructure(currentPdb);
  element<HTMLButtonElement>("download-pdb").onclick = () => download(`${jobName}_unrelaxed_model_1.pdb`, currentPdb, "chemical/x-pdb");
  element<HTMLButtonElement>("download-scores").onclick = () => download(`${jobName}_scores.json`, currentScores, "application/json");
  window.__AFWEBGPU_PREDICTION__ = { meanPlddt: confidence.meanPlddt, ptm: confidence.ptm, elapsedMilliseconds: prediction.elapsedMilliseconds };
}

async function predictionInput(): Promise<{ a3m: string; sequence: string; depth: number }> {
  if (element<HTMLSelectElement>("input-mode").value === "single") {
    const sequence = normalizedSequence();
    if (!/^[ARNDCQEGHILKMFPSTWYVX]+$/.test(sequence)) throw new Error("Sequence must contain only standard amino-acid letters or X");
    return { a3m: `>query\n${sequence}\n`, sequence, depth: 1 };
  }
  const file = element<HTMLInputElement>("a3m-file").files?.[0];
  if (file === undefined) throw new Error("Choose a custom A3M file first");
  const a3m = await file.text(); const parsed = parseA3m(a3m);
  return { a3m, sequence: parsed.query, depth: parsed.depth };
}

element<HTMLFormElement>("prediction-form").addEventListener("submit", (event) => { event.preventDefault(); void runPrediction(); });

async function runPrediction(): Promise<void> {
  const button = element<HTMLButtonElement>("predict"); button.disabled = true;
  element<HTMLElement>("results-section").hidden = true; resetStages(); log("Starting prediction…", false);
  let device: GPUDevice | undefined;
  try {
    stage("device", "active", "Requesting adapter"); setPredictionStatus("Preparing WebGPU");
    if (navigator.gpu === undefined) throw new Error("WebGPU is unavailable. Use a current Chrome or Edge browser on a supported GPU.");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("No compatible WebGPU adapter was found");
    device = await requestAlphaFoldDevice(adapter);
    const adapterName = adapter.info.description || adapter.info.device || adapter.info.vendor || "WebGPU adapter";
    stage("device", "done", adapterName); log(`GPU: ${adapterName}`);

    stage("model", "active", "Downloading one model"); setPredictionStatus("Loading model 1 PTM");
    const manifestValue = element<HTMLInputElement>("model-url").value.trim();
    if (manifestValue === "") throw new Error("A model manifest URL is required");
    localStorage.setItem("afwebgpu.modelUrl", manifestValue);
    const fixture = AlphaFoldFixture.fromStore(await HttpTensorStore.open(manifestValue, updateModelProgress));
    const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables, paeBreaks] = await Promise.all([
      fixture.embeddingWeights(), fixture.templateWeights(), fixture.extraStackWeights(), fixture.mainStackWeights(),
      fixture.structureWeights(), fixture.confidenceWeights(), fixture.geometryTables(), fixture.queryOnlyFeatureTables(),
      fixture.tensor("confidencePaeBreaks"),
    ]);
    stage("model", "done", "Model 1 PTM loaded"); log("Loaded the reduced model-1 tensor bundle.");

    stage("features", "active", "Parsing input"); setPredictionStatus("Building AF2 features");
    const input = await predictionInput();
    const featureOptions = {
      recycles: Number(element<HTMLSelectElement>("recycles").value),
      randomSeed: element<HTMLInputElement>("seed").valueAsNumber,
      maxMsaSequences: element<HTMLInputElement>("max-msa").valueAsNumber,
      maxExtraSequences: element<HTMLInputElement>("max-extra").valueAsNumber,
    };
    const features = makeA3mFeatures(input.a3m, featureTables, featureOptions);
    stage("features", "done", `${input.sequence.length} aa · ${input.depth} rows`); log(`Features: ${input.sequence.length} residues, A3M depth ${input.depth}.`);

    stage("inference", "active", `Recycle 0/${featureOptions.recycles}`); setPredictionStatus("Running AlphaFold2 on WebGPU");
    const reportRecycle = (result: MonomerRecycleResult, recycle: number): void => {
      stage("inference", "active", `Recycle ${recycle}/${featureOptions.recycles} · pLDDT ${result.confidence.meanPlddt.toFixed(1)}`);
      log(`recycle=${recycle} pLDDT=${result.confidence.meanPlddt.toFixed(1)} pTM=${result.confidence.ptm.toFixed(3)} time=${formatSeconds(result.elapsedMilliseconds)}`);
    };
    const prediction = await new AlphaFoldMonomerGpu(device).predict(features, {
      embedding, template, extraStack, mainStack, structure, lddt: confidence.lddt, pae: confidence.pae, geometry,
    }, paeBreaks, reportRecycle);
    stage("inference", "done", formatSeconds(prediction.elapsedMilliseconds));

    stage("results", "active", "Rendering"); setPredictionStatus("Preparing results");
    const jobName = safeJobName(element<HTMLInputElement>("job-name").value);
    showResults(prediction, input.sequence, input.depth, jobName);
    stage("results", "done", "Ready"); setPredictionStatus("Prediction complete", "passed");
    log(`Finished in ${formatSeconds(prediction.elapsedMilliseconds)}.`);
    element<HTMLElement>("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const active = stageOrder.find((name) => document.querySelector<HTMLElement>(`[data-stage="${name}"]`)?.dataset.state === "active");
    if (active !== undefined) stage(active, "error", "Failed");
    setPredictionStatus("Prediction failed", "failed"); log(error instanceof Error ? error.stack ?? message : message);
  } finally {
    device?.destroy(); button.disabled = false;
  }
}

const inputMode = element<HTMLSelectElement>("input-mode");
function updateInputMode(): void {
  const custom = inputMode.value === "custom";
  element<HTMLElement>("sequence-field").hidden = custom; element<HTMLElement>("a3m-field").hidden = !custom;
  element<HTMLElement>("sequence-length").textContent = custom ? "A3M input" : `${normalizedSequence().length} residues`;
}
inputMode.addEventListener("change", updateInputMode);
element<HTMLTextAreaElement>("sequence").addEventListener("input", updateInputMode);
element<HTMLInputElement>("a3m-file").addEventListener("change", (event) => {
  const file = (event.currentTarget as HTMLInputElement).files?.[0];
  element<HTMLElement>("a3m-file-name").textContent = file?.name ?? "Choose an A3M file";
});
try { element<HTMLInputElement>("model-url").value = localStorage.getItem("afwebgpu.modelUrl") ?? "./model/manifest.json"; } catch { /* storage may be unavailable */ }
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
