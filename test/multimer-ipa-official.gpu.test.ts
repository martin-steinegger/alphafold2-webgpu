import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import {
  adaptMultimerInvariantPointAttentionWeights,
  InvariantPointAttentionGpu,
  type MultimerInvariantPointAttentionWeights,
} from "../src/structure/ipa.js";
import { errorMetrics } from "../src/triangle/types.js";

const manifestPath = process.env.AFWEBGPU_MULTIMER_REFERENCE;
const enabled = process.env.AFWEBGPU_GPU_TESTS === "1" && manifestPath !== undefined;

interface Manifest {
  readonly ipaParameters: Readonly<Record<string, Readonly<Record<string, string>>>>;
}

describe.skipIf(!enabled)("official AlphaFold-Multimer-v3 IPA WebGPU reference", () => {
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals); const adapter = await create([]).requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("matches the captured official JAX float32 first structure iteration", async () => {
    const store = await FileTensorStore.open(manifestPath!);
    const parameters = (store.manifest as unknown as Manifest).ipaParameters;
    const p = (module: string, name: string): Promise<Float32Array> => {
      const tensor = parameters[module]?.[name];
      if (tensor === undefined) throw new Error(`missing ${module}/${name}`);
      return store.tensor(tensor);
    };
    const native: MultimerInvariantPointAttentionWeights = {
      pairNormScale: await p("pair_layer_norm", "scale"), pairNormOffset: await p("pair_layer_norm", "offset"),
      queryScalarWeight: await p("q_scalar_projection", "weights"),
      keyScalarWeight: await p("k_scalar_projection", "weights"),
      valueScalarWeight: await p("v_scalar_projection", "weights"),
      queryPointWeight: await p("q_point_projection/point_projection", "weights"),
      queryPointBias: await p("q_point_projection/point_projection", "bias"),
      keyPointWeight: await p("k_point_projection/point_projection", "weights"),
      keyPointBias: await p("k_point_projection/point_projection", "bias"),
      valuePointWeight: await p("v_point_projection/point_projection", "weights"),
      valuePointBias: await p("v_point_projection/point_projection", "bias"),
      trainablePointWeights: await p("", "trainable_point_weights"),
      attention2dWeight: await p("attention_2d", "weights"), attention2dBias: await p("attention_2d", "bias"),
      outputWeight: await p("output_projection", "weights"), outputBias: await p("output_projection", "bias"),
    };
    const activations = await store.tensor("ipaActivations"); const length = store.shape("ipaActivations")[0]!;
    const result = await new InvariantPointAttentionGpu(device).run({
      activations, pair: await store.tensor("structureInputPair"), mask: await store.tensor("ipaMask"),
      affine: await store.tensor("ipaAffine"), length, channels: 384, pairChannels: 128,
      heads: 12, scalarQk: 16, scalarV: 16, pointQk: 4, pointV: 8, multimer: true,
      weights: adaptMultimerInvariantPointAttentionWeights(native, 384, 128, 12, 16, 16, 4, 8),
    });
    const metrics = errorMetrics(result.output, await store.tensor("ipaExpected"));
    expect(metrics.meanAbsoluteError).toBeLessThan(2e-4);
    expect(metrics.maxAbsoluteError).toBeLessThan(3e-3);
  });
});
