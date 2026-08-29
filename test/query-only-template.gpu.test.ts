import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import type { AttentionWeights } from "../src/evoformer/attention.js";
import type { TemplatePairBlockWeights, TriangleAttentionModuleWeights } from "../src/evoformer/block.js";
import { QueryOnlyTemplateGpu, type QueryOnlyTemplateWeights } from "../src/evoformer/template.js";
import type { TransitionWeights } from "../src/evoformer/transition.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics, type TriangleMultiplicationWeights } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

interface Manifest {
  readonly templateEmbedding: {
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

describe.skipIf(!enabled)("query-only mock-template branch WebGPU", () => {
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

  it("matches the official template pair update", async () => {
    const store = await FileTensorStore.open(MANIFEST);
    const modules = (store.manifest as unknown as Manifest).templateEmbedding.parameters;
    const name = (module: string, parameter: string): string => {
      const tensor = modules[module]?.[parameter];
      if (tensor === undefined) throw new Error(`missing ${module}/${parameter}`);
      return tensor;
    };
    const parameter = async (module: string, parameterName: string, block?: number): Promise<Float32Array> => {
      const tensorName = name(module, parameterName);
      const value = await store.tensor(tensorName);
      if (block === undefined) return value;
      const size = value.length / 2;
      return value.subarray(block * size, (block + 1) * size);
    };
    const parameterShape = (module: string, parameterName: string): readonly number[] =>
      store.shape(name(module, parameterName));
    const stackRoot = "single_template_embedding/template_pair_stack/__layer_stack_no_state";
    const attention = async (root: string, block: number): Promise<TriangleAttentionModuleWeights> => {
      const attentionRoot = `${root}/attention`;
      const weights: AttentionWeights = {
        queryNormScale: await parameter(`${root}/query_norm`, "scale", block),
        queryNormOffset: await parameter(`${root}/query_norm`, "offset", block),
        queryWeight: await parameter(attentionRoot, "query_w", block),
        keyWeight: await parameter(attentionRoot, "key_w", block),
        valueWeight: await parameter(attentionRoot, "value_w", block),
        gatingWeight: await parameter(attentionRoot, "gating_w", block),
        gatingBias: await parameter(attentionRoot, "gating_b", block),
        outputWeight: await parameter(attentionRoot, "output_w", block),
        outputBias: await parameter(attentionRoot, "output_b", block),
      };
      return {
        heads: parameterShape(attentionRoot, "gating_b")[1]!, attention: weights,
        pairProjectionWeight: await parameter(root, "feat_2d_weights", block),
      };
    };
    const transition = async (root: string, block: number): Promise<TransitionWeights> => ({
      layerNormScale: await parameter(`${root}/input_layer_norm`, "scale", block),
      layerNormOffset: await parameter(`${root}/input_layer_norm`, "offset", block),
      firstWeight: await parameter(`${root}/transition1`, "weights", block),
      firstBias: await parameter(`${root}/transition1`, "bias", block),
      secondWeight: await parameter(`${root}/transition2`, "weights", block),
      secondBias: await parameter(`${root}/transition2`, "bias", block),
    });
    const triangle = async (root: string, channels: number, block: number): Promise<TriangleMultiplicationWeights> => {
      const hidden = parameterShape(`${root}/left_projection`, "bias")[1]!;
      const projection = async (module: string, inputChannels: number, outputChannels: number): Promise<Float32Array> =>
        transpose(await parameter(`${root}/${module}`, "weights", block), inputChannels, outputChannels);
      return {
        layerNormInWeight: await parameter(`${root}/layer_norm_input`, "scale", block),
        layerNormInBias: await parameter(`${root}/layer_norm_input`, "offset", block),
        linearAPWeight: await projection("left_projection", channels, hidden),
        linearAPBias: await parameter(`${root}/left_projection`, "bias", block),
        linearAGWeight: await projection("left_gate", channels, hidden),
        linearAGBias: await parameter(`${root}/left_gate`, "bias", block),
        linearBPWeight: await projection("right_projection", channels, hidden),
        linearBPBias: await parameter(`${root}/right_projection`, "bias", block),
        linearBGWeight: await projection("right_gate", channels, hidden),
        linearBGBias: await parameter(`${root}/right_gate`, "bias", block),
        layerNormOutWeight: await parameter(`${root}/center_layer_norm`, "scale", block),
        layerNormOutBias: await parameter(`${root}/center_layer_norm`, "offset", block),
        linearZWeight: await projection("output_projection", hidden, channels),
        linearZBias: await parameter(`${root}/output_projection`, "bias", block),
        linearGWeight: await projection("gating_linear", channels, channels),
        linearGBias: await parameter(`${root}/gating_linear`, "bias", block),
      };
    };
    const templateChannels = 64;
    const blockWeights: TemplatePairBlockWeights[] = [];
    for (let block = 0; block < 2; block += 1) {
      blockWeights.push({
        triangleAttentionStarting: await attention(`${stackRoot}/triangle_attention_starting_node`, block),
        triangleAttentionEnding: await attention(`${stackRoot}/triangle_attention_ending_node`, block),
        triangleMultiplicationOutgoing: await triangle(
          `${stackRoot}/triangle_multiplication_outgoing`, templateChannels, block,
        ),
        triangleMultiplicationIncoming: await triangle(
          `${stackRoot}/triangle_multiplication_incoming`, templateChannels, block,
        ),
        pairTransition: await transition(`${stackRoot}/pair_transition`, block),
      });
    }
    const pointwiseValue = await parameter("attention", "value_w");
    const weights: QueryOnlyTemplateWeights = {
      embeddingBias: await parameter("single_template_embedding/embedding2d", "bias"),
      blockWeights,
      outputNormScale: await parameter("single_template_embedding/output_layer_norm", "scale"),
      outputNormOffset: await parameter("single_template_embedding/output_layer_norm", "offset"),
      valueWeight: pointwiseValue,
      outputWeight: await parameter("attention", "output_w"),
      outputBias: await parameter("attention", "output_b"),
      heads: parameterShape("attention", "value_w")[1]!,
    };
    const result = await new QueryOnlyTemplateGpu(device).run({
      length: 59,
      templateChannels,
      pairChannels: 128,
      pairMask: await store.tensor("extraStackPairMask"),
      weights,
    });
    const metrics = errorMetrics(result.pairUpdate, await store.tensor("templatePairUpdateRecycle0"));
    expect(metrics.meanAbsoluteError).toBeLessThan(5e-4);
    expect(metrics.maxAbsoluteError).toBeLessThan(1e-2);
  });
});
