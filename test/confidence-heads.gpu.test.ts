import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { ConfidenceHeadsGpu, type PredictedAlignedErrorWeights, type PredictedLddtWeights } from "../src/heads/confidence.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";
type ParameterMap = Readonly<Record<string, Readonly<Record<string, string>>>>;
interface Manifest {
  readonly confidenceHeads: {
    readonly parameters: { readonly predictedLddt: ParameterMap; readonly predictedAlignedError: ParameterMap };
    readonly reference: { readonly meanPlddt: number; readonly ptm: number };
  };
}

function tmTermsReference(logits: Float32Array, length: number, breaks: Float32Array): Float32Array {
  const bins = breaks.length + 1;
  const step = breaks[1]! - breaks[0]!;
  const centers = Float32Array.from({ length: bins }, (_, bin) =>
    (bin < breaks.length ? breaks[bin]! : breaks[breaks.length - 1]!) + step / 2);
  const d0 = 1.24 * Math.cbrt(Math.max(length, 19) - 15) - 1.8;
  const terms = new Float32Array(length * length);
  for (let row = 0; row < terms.length; row += 1) {
    const base = row * bins;
    let maximum = -Infinity;
    for (let bin = 0; bin < bins; bin += 1) maximum = Math.max(maximum, logits[base + bin]!);
    let denominator = 0;
    let numerator = 0;
    for (let bin = 0; bin < bins; bin += 1) {
      const probability = Math.exp(logits[base + bin]! - maximum);
      denominator += probability;
      numerator += probability / (1 + centers[bin]! ** 2 / d0 ** 2);
    }
    terms[row] = numerator / denominator;
  }
  return terms;
}

describe.skipIf(!enabled)("AlphaFold confidence heads WebGPU", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals); gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("matches official model-1 pLDDT and PAE logits", async () => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest as unknown as Manifest;
    const parameter = async (map: ParameterMap, module: string, name: string): Promise<Float32Array> => {
      const tensor = map[module]?.[name]; if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const lp = manifest.confidenceHeads.parameters.predictedLddt;
    const pp = manifest.confidenceHeads.parameters.predictedAlignedError;
    const lddtWeights: PredictedLddtWeights = {
      normScale: await parameter(lp, "input_layer_norm", "scale"),
      normOffset: await parameter(lp, "input_layer_norm", "offset"),
      act0Weight: await parameter(lp, "act_0", "weights"),
      act0Bias: await parameter(lp, "act_0", "bias"),
      act1Weight: await parameter(lp, "act_1", "weights"),
      act1Bias: await parameter(lp, "act_1", "bias"),
      logitsWeight: await parameter(lp, "logits", "weights"),
      logitsBias: await parameter(lp, "logits", "bias"),
    };
    const paeWeights: PredictedAlignedErrorWeights = {
      logitsWeight: await parameter(pp, "logits", "weights"),
      logitsBias: await parameter(pp, "logits", "bias"),
    };
    const length = store.shape("structureFinalRepresentation")[0]!;
    const structure = await store.tensor("structureFinalRepresentation");
    const pair = await store.tensor("structureInputPair");
    const breaks = await store.tensor("confidencePaeBreaks");
    const heads = new ConfidenceHeadsGpu(device);
    const result = await heads.run(structure, pair, length, lddtWeights, paeWeights, breaks);
    const lddtMetrics = errorMetrics(result.lddtLogits, await store.tensor("confidenceLddtLogits"));
    const plddtMetrics = errorMetrics(result.plddt, await store.tensor("confidencePlddt"));
    const paeMetrics = errorMetrics(result.paeLogits, await store.tensor("confidencePaeLogits"));
    expect(lddtMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(lddtMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(plddtMetrics.maxAbsoluteError).toBeLessThan(2e-3);
    expect(paeMetrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(paeMetrics.maxAbsoluteError).toBeLessThan(3e-3);
    expect(Math.abs(result.meanPlddt - manifest.confidenceHeads.reference.meanPlddt)).toBeLessThan(2e-3);
    expect(Math.abs(result.ptm - manifest.confidenceHeads.reference.ptm)).toBeLessThan(2e-4);

    // Force seven small windows so the bounded path is checked at window boundaries,
    // then compare its GPU softmax reductions with the independent CPU reductions above.
    const reduced = await new ConfidenceHeadsGpu(device).runReduced(
      structure, pair, length, lddtWeights, paeWeights, breaks, { maxPaeLogitsBytes: 128 * 1024 },
    );
    const predictedErrorMetrics = errorMetrics(reduced.predictedAlignedError, result.predictedAlignedError);
    const tmTermMetrics = errorMetrics(reduced.tmScoreTerms, tmTermsReference(result.paeLogits, length, breaks));
    expect(predictedErrorMetrics.meanAbsoluteError).toBeLessThan(2e-5);
    expect(predictedErrorMetrics.maxAbsoluteError).toBeLessThan(2e-4);
    expect(tmTermMetrics.meanAbsoluteError).toBeLessThan(2e-6);
    expect(tmTermMetrics.maxAbsoluteError).toBeLessThan(2e-5);
    expect(Math.abs(reduced.ptm - result.ptm)).toBeLessThan(2e-5);
    expect(errorMetrics(reduced.plddt, result.plddt).maxAbsoluteError).toBe(0);
    expect(reduced.memory!.peakResidentBytes).toBeLessThan(result.memory!.peakResidentBytes);
  });
});
