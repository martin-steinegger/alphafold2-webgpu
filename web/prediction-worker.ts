/**
 * Runs the prediction pipeline off the document's thread.
 *
 * WebGPU work is asynchronous, but encoding forty-eight Evoformer blocks a
 * pass keeps a thread busy, and a page whose main thread is busy cannot paint
 * progress or take a click. In a dedicated worker the GPU work is the same and
 * the document stays responsive; the page falls back to the main thread where
 * a browser does not expose WebGPU to workers.
 */
import {
  clearInferenceCaches, prepareModel, runInference, type InferenceJob, type InferenceReporter, type InferenceOutcome,
} from "./inference.js";
import type { TensorDownloadProgress } from "../src/reference/http-tensor-store.js";
import type { MonomerRecycleSummary } from "../src/model/monomer.js";
import type { MultimerRecycleSummary } from "../src/model/multimer.js";

export type WorkerRequest =
  | { readonly type: "prepare"; readonly id: number; readonly manifestUrl: string }
  | { readonly type: "predict"; readonly id: number; readonly job: InferenceJob }
  | { readonly type: "clear-caches"; readonly id: number };

export type WorkerResponse =
  | { readonly type: "ready"; readonly webgpu: boolean }
  | { readonly type: "stage"; readonly id: number; readonly stage: string; readonly state: string; readonly detail: string }
  | { readonly type: "status"; readonly id: number; readonly text: string }
  | { readonly type: "log"; readonly id: number; readonly text: string }
  | { readonly type: "model-progress"; readonly id: number; readonly progress: TensorDownloadProgress }
  | { readonly type: "recycle"; readonly id: number; readonly recycle: number;
    readonly summary: MonomerRecycleSummary | MultimerRecycleSummary }
  | { readonly type: "result"; readonly id: number; readonly outcome: InferenceOutcome }
  | { readonly type: "cleared"; readonly id: number; readonly persistent: boolean }
  | { readonly type: "error"; readonly id: number; readonly message: string; readonly stack?: string };

const post = (message: WorkerResponse): void => { (self as unknown as Worker).postMessage(message); };

self.addEventListener("message", (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  if (request.type === "clear-caches") {
    void clearInferenceCaches().then(
      (persistent) => post({ type: "cleared", id: request.id, persistent }),
      (error: unknown) => post({ type: "error", id: request.id, message: error instanceof Error ? error.message : String(error) }),
    );
    return;
  }
  const id = request.id;
  const reporter: InferenceReporter = {
    stage: (stage, state, detail) => post({ type: "stage", id, stage, state, detail }),
    status: (text) => post({ type: "status", id, text }),
    log: (text) => post({ type: "log", id, text }),
    modelProgress: (progress) => post({ type: "model-progress", id, progress }),
    recycle: (summary, recycle) => post({ type: "recycle", id, recycle, summary }),
  };
  if (request.type === "prepare") {
    // Errors surface when the prediction itself asks for the model.
    void prepareModel(request.manifestUrl, reporter).catch(() => undefined);
    return;
  }
  void runInference(request.job, reporter).then(
    (outcome) => post({ type: "result", id, outcome }),
    (error: unknown) => post({
      type: "error", id,
      message: error instanceof Error ? error.message : String(error),
      ...(error instanceof Error && error.stack !== undefined ? { stack: error.stack } : {}),
    }),
  );
});

post({ type: "ready", webgpu: typeof navigator.gpu !== "undefined" });
