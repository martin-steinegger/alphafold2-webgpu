import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import type { AttentionWeights } from "../src/evoformer/attention.js";
import type { EvoformerPairBlockWeights, TriangleAttentionModuleWeights } from "../src/evoformer/block.js";
import type { OuterProductMeanWeights } from "../src/evoformer/outer-product-mean.js";
import { ExtraMsaPairStackGpu } from "../src/evoformer/stack.js";
import type { TransitionWeights } from "../src/evoformer/transition.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics, type TriangleMultiplicationWeights } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";
const MANIFEST = "test/fixtures/evoformer/model1-query-59-stack/manifest.json";

interface Manifest {
  readonly extraMsaStack: {
    readonly blocks: number;
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

describe.skipIf(!enabled)("query-only extra-MSA pair stack WebGPU", () => {
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

  it("matches all four official pair updates without CPU neural operations", async () => {
    const store = await FileTensorStore.open(MANIFEST);
    const manifest = store.manifest as unknown as Manifest;
    const modules = manifest.extraMsaStack.parameters;
    const blocks = manifest.extraMsaStack.blocks;
    const tensorName = (module: string, name: string): string => {
      const value = modules[module]?.[name];
      if (value === undefined) throw new Error(`missing ${module}/${name}`);
      return value;
    };
    const parameter = async (module: string, name: string, block: number): Promise<Float32Array> => {
      const nameValue = tensorName(module, name);
      const value = await store.tensor(nameValue);
      const size = value.length / blocks;
      return value.subarray(block * size, (block + 1) * size);
    };
    const shape = (module: string, name: string): readonly number[] => store.shape(tensorName(module, name)).slice(1);
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
        heads: shape(attentionRoot, "gating_b")[0]!, attention: weights,
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
    const triangle = async (root: string, cZ: number, block: number): Promise<TriangleMultiplicationWeights> => {
      const hidden = shape(`${root}/left_projection`, "bias")[0]!;
      const projection = async (module: string, inputChannels: number, outputChannels: number): Promise<Float32Array> =>
        transpose(await parameter(`${root}/${module}`, "weights", block), inputChannels, outputChannels);
      return {
        layerNormInWeight: await parameter(`${root}/layer_norm_input`, "scale", block),
        layerNormInBias: await parameter(`${root}/layer_norm_input`, "offset", block),
        linearAPWeight: await projection("left_projection", cZ, hidden),
        linearAPBias: await parameter(`${root}/left_projection`, "bias", block),
        linearAGWeight: await projection("left_gate", cZ, hidden),
        linearAGBias: await parameter(`${root}/left_gate`, "bias", block),
        linearBPWeight: await projection("right_projection", cZ, hidden),
        linearBPBias: await parameter(`${root}/right_projection`, "bias", block),
        linearBGWeight: await projection("right_gate", cZ, hidden),
        linearBGBias: await parameter(`${root}/right_gate`, "bias", block),
        layerNormOutWeight: await parameter(`${root}/center_layer_norm`, "scale", block),
        layerNormOutBias: await parameter(`${root}/center_layer_norm`, "offset", block),
        linearZWeight: await projection("output_projection", hidden, cZ),
        linearZBias: await parameter(`${root}/output_projection`, "bias", block),
        linearGWeight: await projection("gating_linear", cZ, cZ),
        linearGBias: await parameter(`${root}/gating_linear`, "bias", block),
      };
    };
    const msaShape = store.shape("extraStackInputMsa");
    const pairShape = store.shape("extraStackInputPair");
    const cZ = pairShape[2]!;
    const blockWeights: EvoformerPairBlockWeights[] = [];
    for (let block = 0; block < blocks; block += 1) {
      const outerProductMean: OuterProductMeanWeights = {
        layerNormScale: await parameter("outer_product_mean/layer_norm_input", "scale", block),
        layerNormOffset: await parameter("outer_product_mean/layer_norm_input", "offset", block),
        leftWeight: await parameter("outer_product_mean/left_projection", "weights", block),
        leftBias: await parameter("outer_product_mean/left_projection", "bias", block),
        rightWeight: await parameter("outer_product_mean/right_projection", "weights", block),
        rightBias: await parameter("outer_product_mean/right_projection", "bias", block),
        outputWeight: await parameter("outer_product_mean", "output_w", block),
        outputBias: await parameter("outer_product_mean", "output_b", block),
      };
      blockWeights.push({
        outerProductMean,
        triangleMultiplicationOutgoing: await triangle("triangle_multiplication_outgoing", cZ, block),
        triangleMultiplicationIncoming: await triangle("triangle_multiplication_incoming", cZ, block),
        triangleAttentionStarting: await attention("triangle_attention_starting_node", block),
        triangleAttentionEnding: await attention("triangle_attention_ending_node", block),
        pairTransition: await transition("pair_transition", block),
      });
    }
    const result = await new ExtraMsaPairStackGpu(device).run({
      msa: await store.tensor("extraStackInputMsa"),
      pair: await store.tensor("extraStackInputPair"),
      msaMask: await store.tensor("extraStackMsaMask"),
      pairMask: await store.tensor("extraStackPairMask"),
      sequences: msaShape[0]!, length: msaShape[1]!, cM: msaShape[2]!, cZ,
      cOuter: blockWeights[0]!.outerProductMean.leftBias.length,
      triangleHidden: blockWeights[0]!.triangleMultiplicationOutgoing.linearAPBias.length,
      blockWeights,
    });
    const metrics = errorMetrics(result.pair, await store.tensor("extraStackExpectedPair"));
    expect(metrics.meanAbsoluteError).toBeLessThan(5e-3);
    expect(metrics.maxAbsoluteError).toBeLessThan(7e-2);
  });
});
