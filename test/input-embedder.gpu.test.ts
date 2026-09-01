import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { encodeInputEmbedder, InputEmbedderGpu, type InputEmbedderWeights } from "../src/evoformer/input-embedder.js";
import { WebGpuExecution } from "../src/runtime/execution.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

interface Manifest {
  readonly embedding: {
    readonly parameters: Readonly<Record<string, Readonly<Record<string, string>>>>;
  };
}

describe.skipIf(!enabled)("AlphaFold input/recycling embedder WebGPU", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("matches recycle-0 embeddings before the template branch", async () => {
    const store = await FileTensorStore.open(MANIFEST);
    const parameters = (store.manifest as unknown as Manifest).embedding.parameters;
    const parameter = async (module: string, name: string): Promise<Float32Array> => {
      const tensor = parameters[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const weights: InputEmbedderWeights = {
      preprocess1dWeight: await parameter("preprocess_1d", "weights"),
      preprocess1dBias: await parameter("preprocess_1d", "bias"),
      preprocessMsaWeight: await parameter("preprocess_msa", "weights"),
      preprocessMsaBias: await parameter("preprocess_msa", "bias"),
      leftSingleWeight: await parameter("left_single", "weights"),
      leftSingleBias: await parameter("left_single", "bias"),
      rightSingleWeight: await parameter("right_single", "weights"),
      rightSingleBias: await parameter("right_single", "bias"),
      previousPositionWeight: await parameter("prev_pos_linear", "weights"),
      previousPositionBias: await parameter("prev_pos_linear", "bias"),
      previousMsaNormScale: await parameter("prev_msa_first_row_norm", "scale"),
      previousMsaNormOffset: await parameter("prev_msa_first_row_norm", "offset"),
      previousPairNormScale: await parameter("prev_pair_norm", "scale"),
      previousPairNormOffset: await parameter("prev_pair_norm", "offset"),
      relativePositionWeight: await parameter("pair_activiations", "weights"),
      relativePositionBias: await parameter("pair_activiations", "bias"),
      extraMsaWeight: await parameter("extra_msa_activations", "weights"),
      extraMsaBias: await parameter("extra_msa_activations", "bias"),
    };
    const target = await store.tensor("feature_target_feat_recycle0");
    const msaFeatures = await store.tensor("feature_msa_feat_recycle0");
    const targetShape = store.shape("feature_target_feat_recycle0");
    const msaShape = store.shape("feature_msa_feat_recycle0");
    const extraShape = store.shape("feature_extra_msa_recycle0");
    const length = targetShape[0]!;
    const result = await new InputEmbedderGpu(device).run({
      targetFeatures: target,
      msaFeatures,
      extraMsa: await store.tensor("feature_extra_msa_recycle0"),
      extraHasDeletion: await store.tensor("feature_extra_has_deletion_recycle0"),
      extraDeletionValue: await store.tensor("feature_extra_deletion_value_recycle0"),
      residueIndex: await store.tensor("feature_residue_index_recycle0"),
      aatype: await store.tensor("feature_aatype_recycle0"),
      previousMsaFirstRow: new Float32Array(length * 256),
      previousPair: new Float32Array(length * length * 128),
      previousPositions: new Float32Array(length * 37 * 3),
      length,
      msaSequences: msaShape[0]!,
      extraSequences: extraShape[0]!,
      targetChannels: targetShape[1]!,
      msaFeatureChannels: msaShape[2]!,
      msaChannels: 256,
      pairChannels: 128,
      extraMsaChannels: 64,
      weights,
    });
    const expectedMsaAll = await store.tensor("stackRecycle0InputMsa");
    const expectedMsa = expectedMsaAll.subarray(0, length * 256);
    const msaMetrics = errorMetrics(result.msa, expectedMsa);
    const pairWithTemplates = await store.tensor("extraStackRecycle0InputPair");
    const templateUpdate = await store.tensor("templatePairUpdateRecycle0");
    const expectedPair = new Float32Array(pairWithTemplates.length);
    for (let index = 0; index < expectedPair.length; index += 1) {
      expectedPair[index] = pairWithTemplates[index]! - templateUpdate[index]!;
    }
    const pairMetrics = errorMetrics(result.pairWithoutTemplates, expectedPair);
    const extraMetrics = errorMetrics(result.extraMsa, await store.tensor("extraStackRecycle0InputMsa"));
    expect(msaMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(msaMetrics.maxAbsoluteError).toBeLessThan(1e-3);
    expect(pairMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(pairMetrics.maxAbsoluteError).toBeLessThan(1e-3);
    expect(extraMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(extraMetrics.maxAbsoluteError).toBeLessThan(5e-4);
  });
});

describe.skipIf(!enabled)("AlphaFold input embedder in-place encoder WebGPU", () => {
  let gpu: GPU;
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals);
    gpu = create([]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter");
    device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("writes the pair over the recycled pair and defers the MSA bit-exactly", async () => {
    const store = await FileTensorStore.open(MANIFEST);
    const parameters = (store.manifest as unknown as Manifest).embedding.parameters;
    const parameter = async (module: string, name: string): Promise<Float32Array> => {
      const tensor = parameters[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const weights: InputEmbedderWeights = {
      preprocess1dWeight: await parameter("preprocess_1d", "weights"),
      preprocess1dBias: await parameter("preprocess_1d", "bias"),
      preprocessMsaWeight: await parameter("preprocess_msa", "weights"),
      preprocessMsaBias: await parameter("preprocess_msa", "bias"),
      leftSingleWeight: await parameter("left_single", "weights"),
      leftSingleBias: await parameter("left_single", "bias"),
      rightSingleWeight: await parameter("right_single", "weights"),
      rightSingleBias: await parameter("right_single", "bias"),
      previousPositionWeight: await parameter("prev_pos_linear", "weights"),
      previousPositionBias: await parameter("prev_pos_linear", "bias"),
      previousMsaNormScale: await parameter("prev_msa_first_row_norm", "scale"),
      previousMsaNormOffset: await parameter("prev_msa_first_row_norm", "offset"),
      previousPairNormScale: await parameter("prev_pair_norm", "scale"),
      previousPairNormOffset: await parameter("prev_pair_norm", "offset"),
      relativePositionWeight: await parameter("pair_activiations", "weights"),
      relativePositionBias: await parameter("pair_activiations", "bias"),
      extraMsaWeight: await parameter("extra_msa_activations", "weights"),
      extraMsaBias: await parameter("extra_msa_activations", "bias"),
    };
    const targetShape = store.shape("feature_target_feat_recycle0");
    const msaShape = store.shape("feature_msa_feat_recycle0");
    const extraShape = store.shape("feature_extra_msa_recycle0");
    const length = targetShape[0]!;
    // Non-zero recycled tensors so the in-place LayerNorm and residual are exercised.
    const noise = (elements: number, seed: number): Float32Array => {
      const values = new Float32Array(elements);
      let state = seed >>> 0;
      for (let index = 0; index < elements; index += 1) {
        state = (state * 1664525 + 1013904223) >>> 0;
        values[index] = (state / 2 ** 32 - 0.5) * 4;
      }
      return values;
    };
    const previousMsaFirstRow = noise(length * 256, 1);
    const previousPair = noise(length * length * 128, 2);
    const previousPositions = noise(length * 37 * 3, 3);
    const input = {
      targetFeatures: await store.tensor("feature_target_feat_recycle0"),
      msaFeatures: await store.tensor("feature_msa_feat_recycle0"),
      extraMsa: await store.tensor("feature_extra_msa_recycle0"),
      extraHasDeletion: await store.tensor("feature_extra_has_deletion_recycle0"),
      extraDeletionValue: await store.tensor("feature_extra_deletion_value_recycle0"),
      residueIndex: await store.tensor("feature_residue_index_recycle0"),
      aatype: await store.tensor("feature_aatype_recycle0"),
      previousMsaFirstRow, previousPair, previousPositions, length,
      msaSequences: msaShape[0]!, extraSequences: extraShape[0]!,
      targetChannels: targetShape[1]!, msaFeatureChannels: msaShape[2]!,
      msaChannels: 256, pairChannels: 128, extraMsaChannels: 64, weights,
    };
    const separate = await new InputEmbedderGpu(device).run(input);

    const execution = new WebGpuExecution(device);
    try {
      const residentMsa = execution.upload("test.previous-msa", previousMsaFirstRow);
      const residentPair = execution.upload("test.previous-pair", previousPair,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      const residentPositions = execution.upload("test.previous-positions", previousPositions);
      const encoder = device.createCommandEncoder();
      const embedding = await encodeInputEmbedder(execution, encoder, {
        ...input, previousMsaFirstRow: new Float32Array(0), previousPair: new Float32Array(0),
        previousPositions: new Float32Array(0),
      }, residentMsa, residentPair, residentPositions);
      expect(embedding.pairWithoutTemplates.allocation.buffer).toBe(residentPair.allocation.buffer);
      execution.endComputePass(encoder);
      device.queue.submit([encoder.finish()]);
      const msaEncoder = device.createCommandEncoder();
      const msa = embedding.encodeMsa(msaEncoder);
      execution.endComputePass(msaEncoder);
      const pairReadback = execution.createReadback("test.pair", embedding.pairWithoutTemplates, msaEncoder);
      const msaReadback = execution.createReadback("test.msa", msa, msaEncoder);
      device.queue.submit([msaEncoder.finish()]);
      expect(errorMetrics(await execution.mapFloat32(pairReadback), separate.pairWithoutTemplates).maxAbsoluteError).toBe(0);
      expect(errorMetrics(await execution.mapFloat32(msaReadback), separate.msa).maxAbsoluteError).toBe(0);
    } finally {
      execution.release();
    }
  });
});
