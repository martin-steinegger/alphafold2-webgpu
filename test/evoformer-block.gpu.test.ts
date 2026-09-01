import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  EvoformerBlockGpu,
  type AttentionModuleWeights,
  type EvoformerBlockWeights,
  type RowAttentionModuleWeights,
  type TriangleAttentionModuleWeights,
} from "../src/evoformer/block.js";
import type { AttentionWeights } from "../src/evoformer/attention.js";
import type { OuterProductMeanWeights } from "../src/evoformer/outer-product-mean.js";
import type { TransitionWeights } from "../src/evoformer/transition.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics, type TriangleMultiplicationWeights } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-block0/manifest.json";

interface EvoformerManifest {
  readonly evoformerBlock: {
    readonly parameters: Readonly<Record<string, Readonly<Record<string, string>>>>;
  };
}

function transpose(input: Float32Array, rows: number, columns: number): Float32Array {
  const output = new Float32Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      output[column * rows + row] = input[row * columns + column]!;
    }
  }
  return output;
}

describe.skipIf(!enabled)("complete Evoformer block WebGPU", () => {
  let gpu: GPU;
  let device: GPUDevice;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const adapterName = process.env.AFWEBGPU_ADAPTER;
    gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    device = await adapter.requestDevice();
  });

  afterAll(() => device?.destroy());

  it("matches official AlphaFold with one command buffer and no CPU neural operations", async () => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest as unknown as EvoformerManifest;
    const modules = manifest.evoformerBlock.parameters;
    const parameter = async (module: string, name: string): Promise<Float32Array> => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const shapeOfParameter = (module: string, name: string): readonly number[] => {
      const tensor = modules[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.shape(tensor);
    };
    const attention = async (root: string): Promise<AttentionModuleWeights> => {
      const attentionRoot = `${root}/attention`;
      const heads = shapeOfParameter(attentionRoot, "gating_b")[0]!;
      const weights: AttentionWeights = {
        queryNormScale: await parameter(`${root}/query_norm`, "scale"),
        queryNormOffset: await parameter(`${root}/query_norm`, "offset"),
        queryWeight: await parameter(attentionRoot, "query_w"),
        keyWeight: await parameter(attentionRoot, "key_w"),
        valueWeight: await parameter(attentionRoot, "value_w"),
        gatingWeight: await parameter(attentionRoot, "gating_w"),
        gatingBias: await parameter(attentionRoot, "gating_b"),
        outputWeight: await parameter(attentionRoot, "output_w"),
        outputBias: await parameter(attentionRoot, "output_b"),
      };
      return { heads, attention: weights };
    };
    const transition = async (root: string): Promise<TransitionWeights> => ({
      layerNormScale: await parameter(`${root}/input_layer_norm`, "scale"),
      layerNormOffset: await parameter(`${root}/input_layer_norm`, "offset"),
      firstWeight: await parameter(`${root}/transition1`, "weights"),
      firstBias: await parameter(`${root}/transition1`, "bias"),
      secondWeight: await parameter(`${root}/transition2`, "weights"),
      secondBias: await parameter(`${root}/transition2`, "bias"),
    });
    const triangle = async (root: string, cZ: number): Promise<TriangleMultiplicationWeights> => {
      const hidden = shapeOfParameter(`${root}/left_projection`, "bias")[0]!;
      const projection = async (module: string, inputChannels: number, outputChannels: number): Promise<Float32Array> =>
        transpose(await parameter(`${root}/${module}`, "weights"), inputChannels, outputChannels);
      return {
        layerNormInWeight: await parameter(`${root}/layer_norm_input`, "scale"),
        layerNormInBias: await parameter(`${root}/layer_norm_input`, "offset"),
        linearAPWeight: await projection("left_projection", cZ, hidden),
        linearAPBias: await parameter(`${root}/left_projection`, "bias"),
        linearAGWeight: await projection("left_gate", cZ, hidden),
        linearAGBias: await parameter(`${root}/left_gate`, "bias"),
        linearBPWeight: await projection("right_projection", cZ, hidden),
        linearBPBias: await parameter(`${root}/right_projection`, "bias"),
        linearBGWeight: await projection("right_gate", cZ, hidden),
        linearBGBias: await parameter(`${root}/right_gate`, "bias"),
        layerNormOutWeight: await parameter(`${root}/center_layer_norm`, "scale"),
        layerNormOutBias: await parameter(`${root}/center_layer_norm`, "offset"),
        linearZWeight: await projection("output_projection", hidden, cZ),
        linearZBias: await parameter(`${root}/output_projection`, "bias"),
        linearGWeight: await projection("gating_linear", cZ, cZ),
        linearGBias: await parameter(`${root}/gating_linear`, "bias"),
      };
    };

    const msa = await store.tensor("blockInputMsa");
    const pair = await store.tensor("blockInputPair");
    const msaShape = store.shape("blockInputMsa");
    const pairShape = store.shape("blockInputPair");
    const cZ = pairShape[2]!;
    const rowBase = await attention("msa_row_attention_with_pair_bias");
    const row: RowAttentionModuleWeights = {
      ...rowBase,
      pairLayerNormScale: await parameter("msa_row_attention_with_pair_bias/feat_2d_norm", "scale"),
      pairLayerNormOffset: await parameter("msa_row_attention_with_pair_bias/feat_2d_norm", "offset"),
      pairProjectionWeight: await parameter("msa_row_attention_with_pair_bias", "feat_2d_weights"),
    };
    const startingBase = await attention("triangle_attention_starting_node");
    const endingBase = await attention("triangle_attention_ending_node");
    const starting: TriangleAttentionModuleWeights = {
      ...startingBase,
      pairProjectionWeight: await parameter("triangle_attention_starting_node", "feat_2d_weights"),
    };
    const ending: TriangleAttentionModuleWeights = {
      ...endingBase,
      pairProjectionWeight: await parameter("triangle_attention_ending_node", "feat_2d_weights"),
    };
    const outerProductMean: OuterProductMeanWeights = {
      layerNormScale: await parameter("outer_product_mean/layer_norm_input", "scale"),
      layerNormOffset: await parameter("outer_product_mean/layer_norm_input", "offset"),
      leftWeight: await parameter("outer_product_mean/left_projection", "weights"),
      leftBias: await parameter("outer_product_mean/left_projection", "bias"),
      rightWeight: await parameter("outer_product_mean/right_projection", "weights"),
      rightBias: await parameter("outer_product_mean/right_projection", "bias"),
      outputWeight: await parameter("outer_product_mean", "output_w"),
      outputBias: await parameter("outer_product_mean", "output_b"),
    };
    const weights: EvoformerBlockWeights = {
      msaRowAttention: row,
      msaColumnAttention: await attention("msa_column_attention"),
      msaTransition: await transition("msa_transition"),
      outerProductMean,
      triangleMultiplicationOutgoing: await triangle("triangle_multiplication_outgoing", cZ),
      triangleMultiplicationIncoming: await triangle("triangle_multiplication_incoming", cZ),
      triangleAttentionStarting: starting,
      triangleAttentionEnding: ending,
      pairTransition: await transition("pair_transition"),
    };
    const descriptor = {
      msa,
      pair,
      msaMask: await store.tensor("blockMsaMask"),
      pairMask: await store.tensor("blockPairMask"),
      sequences: msaShape[0]!,
      length: msaShape[1]!,
      cM: msaShape[2]!,
      cZ,
      cOuter: outerProductMean.leftBias.length,
      triangleHidden: weights.triangleMultiplicationOutgoing.linearAPBias.length,
      weights,
    };
    const result = await new EvoformerBlockGpu(device).run(descriptor);
    const msaMetrics = errorMetrics(result.msa, await store.tensor("blockExpectedMsa"));
    const pairMetrics = errorMetrics(result.pair, await store.tensor("blockExpectedPair"));
    expect(msaMetrics.meanAbsoluteError).toBeLessThan(5e-5);
    expect(msaMetrics.maxAbsoluteError).toBeLessThan(5e-4);
    expect(pairMetrics.meanAbsoluteError).toBeLessThan(1e-4);
    expect(pairMetrics.maxAbsoluteError).toBeLessThan(1e-3);

    // Longer chains cover the attention batch in windows. That path never runs
    // at 59 residues, so force it: windowing partitions the batch and must not
    // change any value, including for triangle attention, whose pair bias reads
    // every batch entry and so keeps its normalized input whole.
    for (const attentionWindowBytes of [64 * 1024, 512 * 1024, 4 * 1024 * 1024]) {
      const windowed = await new EvoformerBlockGpu(device).run({ ...descriptor, attentionWindowBytes });
      expect(errorMetrics(windowed.msa, result.msa).maxAbsoluteError).toBe(0);
      expect(errorMetrics(windowed.pair, result.pair).maxAbsoluteError).toBe(0);
    }
  }, 60_000);
});
