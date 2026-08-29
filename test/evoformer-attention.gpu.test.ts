import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  AttentionGpu,
  supportsAttentionSubgroup64x64,
  type AttentionFlashVariant,
  type AttentionPairBias,
  type AttentionWeights,
} from "../src/evoformer/attention.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-block0/manifest.json";

interface EvoformerManifest {
  readonly evoformerBlock: {
    readonly parameters: Readonly<Record<string, Readonly<Record<string, string>>>>;
    readonly referenceStages: Readonly<Record<string, Readonly<Record<string, string>>>>;
  };
}

interface Case {
  readonly stage: string;
  readonly root: string;
  readonly transpose: boolean;
  readonly mask: "blockMsaMask" | "blockPairMask";
  readonly pair: "none" | "separate" | "normalized-input";
}

const CASES: readonly Case[] = [
  {
    stage: "msa_row_attention_with_pair_bias", root: "msa_row_attention_with_pair_bias",
    transpose: false, mask: "blockMsaMask", pair: "separate",
  },
  {
    stage: "msa_column_attention", root: "msa_column_attention",
    transpose: true, mask: "blockMsaMask", pair: "none",
  },
  {
    stage: "triangle_attention_starting_node", root: "triangle_attention_starting_node",
    transpose: false, mask: "blockPairMask", pair: "normalized-input",
  },
  {
    stage: "triangle_attention_ending_node", root: "triangle_attention_ending_node",
    transpose: true, mask: "blockPairMask", pair: "normalized-input",
  },
];

const RUNS: ReadonlyArray<Case & { readonly variant: AttentionFlashVariant }> = CASES.flatMap(
  (testCase) => [
    { ...testCase, variant: "auto" },
    { ...testCase, variant: "subgroup-key32" },
    { ...testCase, variant: "subgroup-64x64" },
  ],
);

describe.skipIf(!enabled)("Evoformer attention WebGPU", () => {
  let gpu: GPU;
  let device: GPUDevice;
  let store: FileTensorStore;
  let manifest: EvoformerManifest;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.AFWEBGPU_ADAPTER;
    gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    device = await requestAlphaFoldDevice(adapter);
    store = await FileTensorStore.open(MANIFEST);
    manifest = store.manifest as unknown as EvoformerManifest;
  });

  afterAll(() => device?.destroy());

  it.each(RUNS)("matches official $stage using $variant", async (testCase) => {
    if (testCase.variant !== "auto" && !supportsAttentionSubgroup64x64(device)) return;
    const modules = manifest.evoformerBlock.parameters;
    const stage = manifest.evoformerBlock.referenceStages[testCase.stage];
    if (stage === undefined) throw new Error(`missing ${testCase.stage} reference stage`);
    const parameter = async (module: string, name: string): Promise<Float32Array> => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const attentionRoot = `${testCase.root}/attention`;
    const activations = await store.tensor(stage.input!);
    const expected = await store.tensor(stage.output!);
    const shape = store.shape(stage.input!);
    const gatingBiasName = modules[attentionRoot]?.gating_b;
    if (gatingBiasName === undefined) throw new Error(`missing ${attentionRoot}/gating_b`);
    const heads = store.shape(gatingBiasName)[0]!;
    const weights: AttentionWeights = {
      queryNormScale: await parameter(`${testCase.root}/query_norm`, "scale"),
      queryNormOffset: await parameter(`${testCase.root}/query_norm`, "offset"),
      queryWeight: await parameter(attentionRoot, "query_w"),
      keyWeight: await parameter(attentionRoot, "key_w"),
      valueWeight: await parameter(attentionRoot, "value_w"),
      gatingWeight: await parameter(attentionRoot, "gating_w"),
      gatingBias: await parameter(attentionRoot, "gating_b"),
      outputWeight: await parameter(attentionRoot, "output_w"),
      outputBias: await parameter(attentionRoot, "output_b"),
    };
    let pairBias: AttentionPairBias | undefined;
    if (testCase.pair === "separate") {
      const pairTensor = stage.pair;
      if (pairTensor === undefined) throw new Error("missing row-attention pair activation");
      const pairShape = store.shape(pairTensor);
      pairBias = {
        source: "separate",
        activations: await store.tensor(pairTensor),
        channels: pairShape[2]!,
        layerNormScale: await parameter(`${testCase.root}/feat_2d_norm`, "scale"),
        layerNormOffset: await parameter(`${testCase.root}/feat_2d_norm`, "offset"),
        projectionWeight: await parameter(testCase.root, "feat_2d_weights"),
      };
    } else if (testCase.pair === "normalized-input") {
      pairBias = {
        source: "normalized-input",
        projectionWeight: await parameter(testCase.root, "feat_2d_weights"),
      };
    }
    const first = shape[0]!;
    const second = shape[1]!;
    const result = await new AttentionGpu(device, { flashVariant: testCase.variant }).run({
      activations,
      mask: await store.tensor(testCase.mask),
      batch: testCase.transpose ? second : first,
      queryLength: testCase.transpose ? first : second,
      channels: shape[2]!,
      heads,
      transpose: testCase.transpose,
      weights,
      ...(pairBias === undefined ? {} : { pairBias }),
    });
    const metrics = errorMetrics(result.output, expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(2e-5);
    expect(metrics.maxAbsoluteError).toBeLessThan(3e-4);
  });
});
