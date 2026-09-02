import { parseA3m } from "../src/input/a3m.js";
import { parseColabFoldComplexA3m } from "../src/input/colabfold-complex-a3m.js";
import { parseSequenceExpression } from "../src/input/sequence-expression.js";
import {
  generateMmseqs2ComplexMsa, generateMmseqs2Msa, type Mmseqs2ComplexMsaResult,
  type Mmseqs2MsaProgress, type Mmseqs2MsaResult,
} from "../src/input/mmseqs2-api.js";
import type { TensorDownloadProgress } from "../src/reference/http-tensor-store.js";
import type { MonomerPrediction, MonomerRecycleSummary } from "../src/model/monomer.js";
import type { MultimerPrediction, MultimerRecycleSummary } from "../src/model/multimer.js";
import {
  clearInferenceCaches, isMemoryFailure, resetInferenceDevice, runInference,
  type InferenceJob, type InferenceOutcome, type InferenceReporter, type InferenceStage, type InferenceStageState,
} from "./inference.js";
import type { WorkerRequest, WorkerResponse } from "./prediction-worker.js";
import { confidenceJson, predictionToPdb, safeJobName } from "./prediction-results.js";
import { packageResults } from "./result-package.js";
import { drawMsaCoverage } from "./msa-plot.js";
import {
  browserPreflightEnvironment, preflightErrorMessage, runWebGpuPreflight, type PreflightStatus, type WebGpuPreflight,
} from "./webgpu-preflight.js";

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
    __AFWEBGPU_PREDICTION__?: {
      meanPlddt: number; ptm: number; iptm?: number;
      elapsedMilliseconds: number; modelLoadMilliseconds: number;
      recycles: readonly {
        elapsedMilliseconds: number; meanPlddt: number; ptm: number; iptm?: number;
        trunkSubmissions: MonomerRecycleSummary["trunkSubmissions"];
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
let customA3mUsesMultimer = false;
const inputUsesMultimer = (): boolean => element<HTMLSelectElement>("input-mode").value === "custom"
  ? customA3mUsesMultimer : normalizedSequence().includes(":");
const formatSeconds = (milliseconds: number): string => `${(milliseconds / 1000).toFixed(2)} s`;
const formatMib = (bytes: number): string => `${(bytes / 1024 ** 2).toFixed(0)} MiB`;
const stageOrder = ["device", "msa", "model", "features", "inference", "results"] as const;
type Stage = InferenceStage;
type StageState = InferenceStageState;

let currentPdb = "";
let currentScores = "";
let generatedMsa: {
  readonly key: string; readonly result: Mmseqs2MsaResult | Mmseqs2ComplexMsaResult;
} | undefined;
let lastMsaStatus = "";
let viewerLoader: Promise<ThreeDmolApi> | undefined;
let viewerResizeObserver: ResizeObserver | undefined;

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
  const prefix = value.search === "paired" ? "Paired: " : value.search === "unpaired" ? "Unpaired: " : "";
  const label = `${prefix}${labels[value.phase] ?? value.status}`;
  stage("msa", value.phase === "complete" ? "done" : "active", `${label} · ${seconds}s`);
  setPredictionStatus(label);
  const statusKey = `${value.search ?? "monomer"}:${value.phase}:${value.status}`;
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

function downloadBlob(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function download(filename: string, contents: string, type: string): void {
  downloadBlob(filename, new Blob([contents], { type }));
}

/**
 * A plot as PNG bytes, composited onto white: the canvases are transparent
 * where nothing was drawn, and a transparent plot is unreadable in the image
 * viewers people open a downloaded archive with.
 */
async function canvasPng(canvas: HTMLCanvasElement): Promise<Uint8Array> {
  const opaque = document.createElement("canvas");
  opaque.width = canvas.width; opaque.height = canvas.height;
  const context = opaque.getContext("2d");
  if (context === null) throw new Error("this browser cannot render the plots to PNG");
  context.fillStyle = "#ffffff"; context.fillRect(0, 0, opaque.width, opaque.height);
  context.drawImage(canvas, 0, 0);
  const blob = await new Promise<Blob | null>((resolve) => opaque.toBlob(resolve, "image/png"));
  if (blob === null) throw new Error("this browser could not encode the plots as PNG");
  return new Uint8Array(await blob.arrayBuffer());
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

/** Everything the archive reports about a run beyond the prediction itself. */
interface ResultContext {
  readonly sequence: string;
  readonly depth: number;
  readonly jobName: string;
  readonly a3m: string;
  readonly modelLoadMilliseconds: number;
  readonly chainLengths?: readonly number[];
  readonly multimer: boolean;
  /** The page's input mode, which names how the alignment was obtained. */
  readonly inputMode: string;
  readonly job: InferenceJob;
  readonly adapterName: string;
}

const MSA_MODES: Readonly<Record<string, string>> = {
  mmseqs2: "mmseqs2_uniref_env", single: "single_sequence", custom: "custom",
};

const round = (value: number, decimals: number): number => Number(value.toFixed(decimals));

/** The settings and timings that produced this prediction, for the archive's config.json. */
function runSettings(prediction: MonomerPrediction | MultimerPrediction, context: ResultContext): Record<string, unknown> {
  const confidence = prediction.final.confidence as MonomerPrediction["final"]["confidence"] & { readonly iptm?: number };
  const { job } = context;
  return {
    job_name: context.jobName,
    date: new Date().toISOString(),
    implementation: "alphafold2-webgpu",
    model_type: context.multimer ? "alphafold2_multimer_v3" : "alphafold2_ptm",
    model_number: 1,
    model_manifest: job.manifestUrl,
    num_recycles: job.recycles,
    max_msa: `${job.maxMsaSequences}:${job.maxExtraSequences}`,
    msa_mode: MSA_MODES[context.inputMode] ?? context.inputMode,
    msa_depth: context.depth,
    random_seed: job.randomSeed,
    activation_storage: job.packedStorage ? "f16" : "f32",
    length: context.sequence.length,
    chain_lengths: context.chainLengths ?? [context.sequence.length],
    adapter: context.adapterName,
    user_agent: navigator.userAgent,
    model_load_seconds: round(context.modelLoadMilliseconds / 1000, 2),
    inference_seconds: round(prediction.elapsedMilliseconds / 1000, 2),
    mean_plddt: round(confidence.meanPlddt, 2),
    ptm: round(confidence.ptm, 4),
    ...(confidence.iptm === undefined ? {} : { iptm: round(confidence.iptm, 4) }),
    recycles: prediction.recycles.map((result, index) => ({
      recycle: index,
      mean_plddt: round(result.confidence.meanPlddt, 2),
      ptm: round(result.confidence.ptm, 4),
      ...((result as MultimerRecycleSummary).confidence.iptm === undefined
        ? {} : { iptm: round((result as MultimerRecycleSummary).confidence.iptm, 4) }),
      seconds: round(result.elapsedMilliseconds / 1000, 2),
    })),
  };
}

/**
 * Collects the run into one ColabFold-shaped archive: structure, scores, the
 * alignment error in AlphaFold-DB's format, the three plots, the alignment,
 * the settings, the run log and the citations for the methods used.
 */
async function downloadResultPackage(
  prediction: MonomerPrediction | MultimerPrediction, context: ResultContext,
): Promise<void> {
  const button = element<HTMLButtonElement>("download-results");
  const label = button.textContent ?? "Download results";
  button.disabled = true; button.textContent = "Packaging…";
  try {
    const plots = [["plddt", "plddt-plot"], ["pae", "pae-plot"], ["coverage", "msa-plot"]] as const;
    const images = await Promise.all(plots.map(async ([suffix, id]) => (
      { suffix, png: await canvasPng(element<HTMLCanvasElement>(id)) }
    )));
    const blob = await packageResults({
      jobName: context.jobName,
      sequence: context.sequence,
      chainLengths: context.chainLengths ?? [context.sequence.length],
      confidence: prediction.final.confidence,
      pdb: currentPdb,
      scoresJson: currentScores,
      a3m: context.a3m,
      depth: context.depth,
      images,
      settings: runSettings(prediction, context),
      log: element<HTMLPreElement>("run-log").textContent ?? "",
      usedMmseqs2: context.inputMode === "mmseqs2",
      multimer: context.multimer,
    });
    downloadBlob(`${context.jobName}.result.zip`, blob);
    log(`Packaged the results as ${context.jobName}.result.zip (${formatMib(blob.size)}).`);
  } catch (error) {
    log(`Could not package the results: ${error instanceof Error ? error.message : String(error)}.`);
  } finally { button.disabled = false; button.textContent = label; }
}

function showResults(prediction: MonomerPrediction | MultimerPrediction, context: ResultContext): void {
  const { sequence, depth, jobName, a3m, modelLoadMilliseconds, chainLengths } = context;
  const confidence = prediction.final.confidence;
  currentPdb = predictionToPdb(sequence, prediction.final.structure, confidence.plddt, chainLengths);
  currentScores = confidenceJson(sequence, confidence);
  element<HTMLElement>("results-section").hidden = false;
  element<HTMLElement>("mean-plddt").textContent = confidence.meanPlddt.toFixed(1);
  element<HTMLElement>("plddt-label").textContent = confidenceLabel(confidence.meanPlddt);
  element<HTMLElement>("ptm").textContent = confidence.ptm.toFixed(3);
  const multimerConfidence = confidence as typeof confidence & { readonly iptm?: number };
  element<HTMLElement>("iptm-card").hidden = multimerConfidence.iptm === undefined;
  if (multimerConfidence.iptm !== undefined) element<HTMLElement>("iptm").textContent = multimerConfidence.iptm.toFixed(3);
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
  const packageButton = element<HTMLButtonElement>("download-results");
  packageButton.hidden = false; packageButton.disabled = false;
  packageButton.onclick = () => { void downloadResultPackage(prediction, context); };
  window.__AFWEBGPU_PREDICTION__ = {
    meanPlddt: confidence.meanPlddt, ptm: confidence.ptm,
    ...(multimerConfidence.iptm === undefined ? {} : { iptm: multimerConfidence.iptm }),
    elapsedMilliseconds: prediction.elapsedMilliseconds, modelLoadMilliseconds,
    recycles: prediction.recycles.map((result) => ({
      elapsedMilliseconds: result.elapsedMilliseconds, meanPlddt: result.confidence.meanPlddt,
      ptm: result.confidence.ptm,
      ...((result as MultimerRecycleSummary).confidence.iptm === undefined
        ? {} : { iptm: (result as MultimerRecycleSummary).confidence.iptm }),
      trunkSubmissions: result.trunkSubmissions,
    })),
  };
}

interface PredictionInput {
  readonly a3m: string; readonly sequence: string; readonly depth: number;
  readonly multimer: boolean; readonly chains?: readonly string[];
  readonly alignmentMask?: Float32Array;
}

async function predictionInput(): Promise<PredictionInput> {
  const mode = element<HTMLSelectElement>("input-mode").value;
  const parsedExpression = mode === "custom" ? undefined
    : parseSequenceExpression(element<HTMLTextAreaElement>("sequence").value);
  if (parsedExpression?.multimer === true && mode === "single") {
      const { chains, sequence } = parsedExpression;
      stage("msa", "done", `${chains.length} query-only chains`);
      return { a3m: `>query\n${sequence}\n`, sequence, depth: 1, multimer: true, chains };
  }
  if (mode === "single") {
    const { sequence } = parsedExpression!;
    stage("msa", "done", "Single sequence");
    return { a3m: `>query\n${sequence}\n`, sequence, depth: 1, multimer: false };
  }
  if (mode === "mmseqs2") {
    const { sequence, chains, multimer } = parsedExpression!;
    const apiUrl = element<HTMLInputElement>("msa-api-url").value.trim();
    if (apiUrl === "") throw new Error("An MMseqs2 API URL is required");
    try { localStorage.setItem("afwebgpu.msaApiUrl", apiUrl); } catch { /* storage may be unavailable */ }
    const key = `${apiUrl}\n${multimer ? chains.join(":") : sequence}`;
    if (generatedMsa?.key === key) {
      stage("msa", "done", `${generatedMsa.result.depth} rows · cached`); log("Reusing the generated MMseqs2 alignment.");
      const cached = generatedMsa.result;
      return multimer && "mask" in cached
        ? { a3m: cached.a3m, alignmentMask: cached.mask, sequence, depth: cached.depth, multimer: true, chains }
        : { a3m: cached.a3m, sequence, depth: cached.depth, multimer: false };
    }
    const result = multimer
      ? await generateMmseqs2ComplexMsa(chains, { apiUrl, onProgress: updateMsaProgress })
      : await generateMmseqs2Msa(sequence, { apiUrl, onProgress: updateMsaProgress });
    generatedMsa = { key, result };
    stage("msa", "done", `${result.depth} rows · ${formatSeconds(result.elapsedMilliseconds)}`);
    const downloadButton = element<HTMLButtonElement>("download-msa"); downloadButton.hidden = false;
    downloadButton.onclick = () => download(`${safeJobName(element<HTMLInputElement>("job-name").value)}.a3m`, result.a3m, "text/plain");
    return multimer && "mask" in result
      ? { a3m: result.a3m, alignmentMask: result.mask, sequence, depth: result.depth, multimer: true, chains }
      : { a3m: result.a3m, sequence, depth: result.depth, multimer: false };
  }
  const file = element<HTMLInputElement>("a3m-file").files?.[0];
  if (file === undefined) throw new Error("Choose a custom A3M file first");
  const a3m = await file.text();
  const complex = parseColabFoldComplexA3m(a3m);
  if (complex !== undefined) {
    const sequence = complex.chains.join("");
    stage("msa", "done", `${complex.depth} uploaded complex rows`);
    return {
      a3m: complex.a3m, alignmentMask: complex.mask, sequence, depth: complex.depth,
      multimer: true, chains: complex.chains,
    };
  }
  const parsed = parseA3m(a3m);
  stage("msa", "done", `${parsed.depth} uploaded rows`);
  return { a3m, sequence: parsed.query, depth: parsed.depth, multimer: false };
}

element<HTMLFormElement>("prediction-form").addEventListener("submit", (event) => { event.preventDefault(); void runPrediction(); });

/**
 * The pipeline runs in a dedicated worker when the browser exposes WebGPU
 * there, so the document keeps painting and the Stop button keeps working
 * through a long prediction; otherwise it runs here, on the main thread.
 */
class PredictionRunner {
  #worker: Worker | undefined;
  #ready: Promise<boolean> | undefined;
  #nextId = 1;
  readonly #pending = new Map<number, {
    readonly reporter: InferenceReporter;
    readonly resolve: (outcome: InferenceOutcome) => void;
    readonly reject: (error: Error) => void;
  }>();
  #useWorker: boolean | undefined;

  /** Whether the worker path is available; resolved once per page. */
  async available(): Promise<boolean> {
    if (this.#useWorker !== undefined) return this.#useWorker;
    // ?worker=0 keeps the pipeline on the main thread, for comparison.
    if (typeof Worker === "undefined" || parameter("worker", "1") === "0") { this.#useWorker = false; return false; }
    try {
      this.#useWorker = await this.#ensureWorker();
    } catch { this.#useWorker = false; }
    if (!this.#useWorker) this.#dispose();
    return this.#useWorker;
  }

  #ensureWorker(): Promise<boolean> {
    if (this.#worker !== undefined && this.#ready !== undefined) return this.#ready;
    const worker = new Worker(new URL("./prediction-worker.ts", import.meta.url), { type: "module" });
    this.#worker = worker;
    this.#ready = new Promise<boolean>((resolve, reject) => {
      const onReady = (event: MessageEvent<WorkerResponse>): void => {
        if (event.data.type !== "ready") return;
        worker.removeEventListener("message", onReady);
        resolve(event.data.webgpu);
      };
      worker.addEventListener("message", onReady);
      worker.addEventListener("error", (event) => reject(new Error(event.message || "prediction worker failed to start")), { once: true });
    });
    worker.addEventListener("message", (event: MessageEvent<WorkerResponse>) => this.#onMessage(event.data));
    worker.addEventListener("error", (event) => {
      const error = new Error(event.message || "prediction worker failed");
      for (const entry of this.#pending.values()) entry.reject(error);
      this.#pending.clear();
    });
    return this.#ready;
  }

  #onMessage(message: WorkerResponse): void {
    if (message.type === "ready") return;
    const entry = this.#pending.get(message.id);
    if (entry === undefined) return;
    switch (message.type) {
      case "stage": entry.reporter.stage(message.stage as InferenceStage, message.state as InferenceStageState, message.detail); break;
      case "status": entry.reporter.status(message.text); break;
      case "log": entry.reporter.log(message.text); break;
      case "model-progress": entry.reporter.modelProgress(message.progress); break;
      case "recycle": entry.reporter.recycle(message.summary, message.recycle); break;
      case "result": this.#pending.delete(message.id); entry.resolve(message.outcome); break;
      case "cleared": this.#pending.delete(message.id); break;
      case "error": {
        this.#pending.delete(message.id);
        const error = new Error(message.message);
        if (message.stack !== undefined) error.stack = message.stack;
        entry.reject(error);
        break;
      }
    }
  }

  #post(request: WorkerRequest): void { this.#worker?.postMessage(request); }

  /** Start loading the model in the worker while the alignment is still being prepared. */
  async prepare(manifestUrl: string, reporter: InferenceReporter): Promise<void> {
    if (!(await this.available())) return;
    const id = this.#nextId; this.#nextId += 1;
    this.#pending.set(id, { reporter, resolve: () => { this.#pending.delete(id); }, reject: () => { this.#pending.delete(id); } });
    this.#post({ type: "prepare", id, manifestUrl });
  }

  async run(job: InferenceJob, reporter: InferenceReporter): Promise<InferenceOutcome> {
    if (!(await this.available())) return runInference(job, reporter);
    const id = this.#nextId; this.#nextId += 1;
    return new Promise<InferenceOutcome>((resolve, reject) => {
      this.#pending.set(id, { reporter, resolve, reject });
      this.#post({ type: "predict", id, job });
    });
  }

  /** Stop whatever is running: the worker is terminated, which also frees its GPU device. */
  stop(): void {
    const error = new Error("Prediction stopped");
    for (const entry of this.#pending.values()) entry.reject(error);
    this.#pending.clear();
    this.#dispose();
  }

  /** Forget the loaded model, both here and in the worker. */
  async clearCaches(): Promise<boolean> {
    const persistent = await clearInferenceCaches();
    resetInferenceDevice();
    if (this.#worker !== undefined) {
      const id = this.#nextId; this.#nextId += 1;
      this.#post({ type: "clear-caches", id });
    }
    return persistent;
  }

  /** After a memory failure the device is destroyed: here directly, in the worker by replacing it. */
  resetDevice(): void {
    resetInferenceDevice();
    if (this.#worker !== undefined) this.#dispose();
  }

  #dispose(): void {
    this.#worker?.terminate();
    this.#worker = undefined;
    this.#ready = undefined;
  }
}

const runner = new PredictionRunner();

const domReporter: InferenceReporter = {
  stage: (stageName, state, detail) => stage(stageName, state, detail),
  status: (text) => setPredictionStatus(text),
  log: (text) => log(text),
  modelProgress: (progress) => updateModelProgress(progress),
  recycle: () => { /* the pipeline logs each recycle itself */ },
};

async function runPrediction(): Promise<void> {
  const button = element<HTMLButtonElement>("predict"); button.disabled = true;
  const stopButton = element<HTMLButtonElement>("stop");
  const clearCacheButton = element<HTMLButtonElement>("clear-model-cache"); clearCacheButton.disabled = true;
  element<HTMLElement>("results-section").hidden = true;
  element<HTMLButtonElement>("download-results").hidden = true; resetStages(); log("Starting prediction…", false); lastMsaStatus = "";
  try {
    stage("device", "active", "Checking WebGPU support"); setPredictionStatus("Preparing WebGPU");
    const preflight = await webGpuPreflight();
    if (!preflight.usable) {
      log(`${preflight.headline}. ${preflight.detail}`);
      for (const remedy of preflight.remedies) log(`· ${remedy}`);
      throw new Error(preflight.headline);
    }
    if (preflight.status === "warning") log(`Warning: ${preflight.headline}. ${preflight.detail}`);
    const inWorker = await runner.available();
    log(inWorker ? "Inference runs in a dedicated worker; the page stays responsive."
      : "Inference runs on the page's main thread (this browser has no WebGPU in workers).");
    stopButton.hidden = !inWorker; stopButton.disabled = false;

    stage("msa", "active", "Preparing input");
    stage("model", "active", "Downloading one model"); setPredictionStatus("Preparing alignment and model");
    const inputPromise = predictionInput();
    const expectedMultimer = element<HTMLSelectElement>("input-mode").value === "custom"
      ? (await inputPromise).multimer : inputUsesMultimer();
    const modelUrlId = expectedMultimer ? "multimer-model-url" : "monomer-model-url";
    const manifestValue = element<HTMLInputElement>(modelUrlId).value.trim();
    if (manifestValue === "") throw new Error("A model manifest URL is required");
    try { localStorage.setItem(`afwebgpu.${expectedMultimer ? "multimer" : "monomer"}ModelUrl`, manifestValue); }
    catch { /* storage may be unavailable */ }
    const manifestUrl = new URL(manifestValue, location.href).href;
    // The model downloads while the alignment is being generated.
    await runner.prepare(manifestUrl, domReporter);
    const input = await inputPromise;
    const job: InferenceJob = {
      manifestUrl, input,
      maxMsaSequences: element<HTMLInputElement>("max-msa").valueAsNumber,
      maxExtraSequences: element<HTMLInputElement>("max-extra").valueAsNumber,
      recycles: Number(element<HTMLSelectElement>("recycles").value),
      randomSeed: element<HTMLInputElement>("seed").valueAsNumber,
      packedStorage: element<HTMLSelectElement>("monomer-storage").value === "f16",
      compactPolicy: parameter("compact", "0") === "1",
      ...(parameter("profile", "0") === "1" ? { profile: {
        recycle: Number(parameter("profileRecycle", "0")),
        extraMsaBlock: Number(parameter("profileExtraBlock", "0")),
        mainEvoformerBlock: Number(parameter("profileMainBlock", "0")),
      } } : {}),
    };
    const { prediction, modelLoadMilliseconds, adapterName } = await runner.run(job, domReporter);

    stage("results", "active", "Rendering"); setPredictionStatus("Preparing results");
    const jobName = safeJobName(element<HTMLInputElement>("job-name").value);
    showResults(prediction, {
      sequence: input.sequence, depth: input.depth, jobName, a3m: input.a3m, modelLoadMilliseconds,
      ...(input.chains === undefined ? {} : { chainLengths: input.chains.map((chain) => chain.length) }),
      multimer: input.multimer, inputMode: element<HTMLSelectElement>("input-mode").value, job, adapterName,
    });
    stage("results", "done", "Ready"); setPredictionStatus("Prediction complete", "passed");
    log(`Finished in ${formatSeconds(prediction.elapsedMilliseconds)}.`);
    element<HTMLElement>("results-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const active = stageOrder.find((name) => document.querySelector<HTMLElement>(`[data-stage="${name}"]`)?.dataset.state === "active");
    if (active !== undefined) stage(active, "error", "Failed");
    if (isMemoryFailure(message)) {
      runner.resetDevice();
      log("The WebGPU device could not retain this allocation set. Reduce clustered/extra MSA rows or sequence length, then retry.");
    }
    setPredictionStatus(message === "Prediction stopped" ? "Prediction stopped" : "Prediction failed", "failed");
    log(error instanceof Error ? error.stack ?? message : message);
  } finally { button.disabled = false; clearCacheButton.disabled = false; stopButton.hidden = true; }
}
element<HTMLButtonElement>("stop").addEventListener("click", () => {
  element<HTMLButtonElement>("stop").disabled = true;
  log("Stopping the prediction; the worker and its GPU device are discarded.");
  runner.stop();
});

const inputMode = element<HTMLSelectElement>("input-mode");
function updateInputMode(): void {
  const custom = inputMode.value === "custom"; const remote = inputMode.value === "mmseqs2";
  const multimer = inputUsesMultimer();
  element<HTMLElement>("sequence-field").hidden = custom; element<HTMLElement>("a3m-field").hidden = !custom;
  const raw = element<HTMLTextAreaElement>("sequence").value.replace(/\s+/g, "").toUpperCase();
  const chainCount = raw === "" ? 0 : raw.split(":").length;
  element<HTMLElement>("sequence-length").textContent = custom ? "A3M input"
    : multimer ? `${raw.replaceAll(":", "").length} residues · ${chainCount} chains`
      : `${normalizedSequence().length} residues`;
  element<HTMLElement>("sequence-hint").textContent = multimer
    ? remote
      ? `Detected ${chainCount} chains. ColabFold searches paired and unpaired MSAs before local Multimer-v3 inference.`
      : `Detected ${chainCount} chains. Runs local query-only Multimer-v3.`
    : remote
      ? "Detected one chain. MMseqs2 sends it to the public ColabFold MSA server before local inference."
      : "Detected one chain. Single-sequence inference stays on this device.";
  element<HTMLElement>("predict-label").textContent = multimer && remote ? "Generate complex MSA & predict"
    : multimer ? "Run Multimer-v3"
    : remote ? "Generate MSA & predict" : "Run prediction";
  element<HTMLInputElement>("max-msa").disabled = multimer && inputMode.value === "single";
  element<HTMLSelectElement>("monomer-storage").disabled = false;
  const maxExtra = element<HTMLInputElement>("max-extra");
  maxExtra.disabled = multimer && inputMode.value === "single";
  maxExtra.max = multimer ? "2048" : "1024";
  if (multimer && maxExtra.value === "1024") maxExtra.value = "2048";
  else if (!multimer && maxExtra.valueAsNumber > 1024) maxExtra.value = "1024";
  const recycleSelect = element<HTMLSelectElement>("recycles");
  if (multimer && recycleSelect.value === "3") recycleSelect.value = "20";
  else if (!multimer && recycleSelect.value === "20") recycleSelect.value = "3";
}
inputMode.addEventListener("change", updateInputMode);
element<HTMLTextAreaElement>("sequence").addEventListener("input", () => { generatedMsa = undefined;
  element<HTMLButtonElement>("download-msa").hidden = true; updateInputMode(); });
element<HTMLInputElement>("a3m-file").addEventListener("change", (event) => {
  const input = event.currentTarget as HTMLInputElement;
  const file = input.files?.[0];
  element<HTMLElement>("a3m-file-name").textContent = file?.name ?? "Choose an A3M file";
  void (async () => {
    customA3mUsesMultimer = file !== undefined
      && /^#[1-9][0-9]*(?:,[1-9][0-9]*)*\t[1-9][0-9]*(?:,[1-9][0-9]*)*/.test(
        (await file.slice(0, 256).text()).replace(/\r/g, "").trimStart(),
      );
    if (input.files?.[0] === file) updateInputMode();
  })();
});
try {
  const legacy = localStorage.getItem("afwebgpu.modelUrl");
  element<HTMLInputElement>("monomer-model-url").value = localStorage.getItem("afwebgpu.monomerModelUrl")
    ?? legacy ?? "./model/manifest.json";
  element<HTMLInputElement>("multimer-model-url").value = localStorage.getItem("afwebgpu.multimerModelUrl")
    ?? "./model-multimer/manifest.json";
} catch { /* storage may be unavailable */ }
try { element<HTMLInputElement>("msa-api-url").value = localStorage.getItem("afwebgpu.msaApiUrl") ?? "https://api.colabfold.com"; } catch { /* storage may be unavailable */ }
if (parameter("precision", "") === "f16") element<HTMLSelectElement>("monomer-storage").value = "f16";
element<HTMLButtonElement>("clear-model-cache").addEventListener("click", () => { void (async () => {
  const button = element<HTMLButtonElement>("clear-model-cache"); button.disabled = true;
  try {
    const removed = await runner.clearCaches();
    log(removed ? "Cleared the persistent and in-memory model caches." : "Cleared the in-memory model cache; no persistent cache was present.", false);
  } finally { button.disabled = false; }
})(); });
updateInputMode();

let preflightPromise: Promise<WebGpuPreflight> | undefined;

/** Runs the capability check once per page and reuses its verdict for every prediction. */
function webGpuPreflight(refresh = false): Promise<WebGpuPreflight> {
  if (refresh || preflightPromise === undefined) {
    preflightPromise = runWebGpuPreflight(browserPreflightEnvironment());
  }
  return preflightPromise;
}

const capabilityState: Readonly<Record<PreflightStatus, string>> = {
  ready: "ready", warning: "warning", unsupported: "failed", blocked: "failed", insufficient: "failed",
};

function renderPreflight(preflight: WebGpuPreflight): void {
  element<HTMLDivElement>("gpu-summary").dataset.state = capabilityState[preflight.status];
  element<HTMLElement>("gpu-summary-text").textContent = preflight.headline;
  element<HTMLElement>("gpu-detail-text").textContent = preflight.detail;
  const remedies = element<HTMLUListElement>("gpu-remedies");
  remedies.replaceChildren(...preflight.remedies.map((remedy) => {
    const item = document.createElement("li"); item.textContent = remedy; return item;
  }));
  const details = element<HTMLDetailsElement>("gpu-details");
  details.hidden = preflight.status === "ready";
  details.open = !preflight.usable;
}

void webGpuPreflight().then(renderPreflight, (error: unknown) => {
  renderPreflight({
    status: "blocked", usable: false, headline: "The WebGPU compatibility check failed",
    detail: error instanceof Error ? error.message : String(error), remedies: [],
    adapter: undefined, shortfalls: [], browser: { engine: "unknown", name: "This browser", version: undefined, platform: "unknown" },
  });
});
