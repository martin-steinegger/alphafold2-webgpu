import { create, globals } from "webgpu";
import {
  AttentionGpu,
  type AttentionFlashVariant,
  type AttentionInput,
  type AttentionWeights,
} from "../src/evoformer/attention.js";
import { requestAlphaFoldDevice } from "../src/runtime/device.js";
import { AlphaFoldFixture } from "../src/reference/alphafold-fixture.js";
import { FileTensorStore } from "../src/reference/tensor-store.js";
import { errorMetrics } from "../src/triangle/types.js";

const MANIFEST = "test/fixtures/evoformer/model1-a3m-59-stack/manifest.json";
const VARIANTS: readonly AttentionFlashVariant[] = [
  "portable", "auto", "subgroup-4x8", "subgroup-key32", "subgroup-8x16", "subgroup-8x32", "subgroup-8x64",
  "subgroup-16x64", "subgroup-32x64", "subgroup-64x64",
];

Object.assign(globalThis, globals);
const adapterName = process.env.AFWEBGPU_ADAPTER;
const gpu = create(adapterName === undefined ? [] : [`adapter=${adapterName}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("no WebGPU adapter is available");
const device = await requestAlphaFoldDevice(adapter);

try {
  const store = await FileTensorStore.open(MANIFEST);
  const fixture = AlphaFoldFixture.fromStore(store);
  const shape = store.shape("blockInputMsa");
  const module = (await fixture.mainStackWeights())[0]!.msaColumnAttention;
  const weights: AttentionWeights = module.attention;
  const input: AttentionInput = {
    activations: await store.tensor("blockInputMsa"),
    mask: await store.tensor("blockMsaMask"),
    batch: shape[1]!,
    queryLength: shape[0]!,
    channels: shape[2]!,
    heads: module.heads,
    transpose: true,
    weights,
  };
  const expected = (await new AttentionGpu(device, { flashVariant: "portable" }).run(input)).output;
  const reports = [];
  for (const variant of VARIANTS) {
    const runner = new AttentionGpu(device, { flashVariant: variant });
    await runner.run(input);
    const times: number[] = [];
    let output: Float32Array<ArrayBufferLike> = expected;
    for (let iteration = 0; iteration < 5; iteration += 1) {
      const result = await runner.run(input);
      times.push(result.elapsedMilliseconds);
      output = result.output;
    }
    reports.push({
      variant,
      meanMilliseconds: times.reduce((sum, value) => sum + value, 0) / times.length,
      minimumMilliseconds: Math.min(...times),
      times,
      error: errorMetrics(output, expected),
    });
  }
  process.stdout.write(`${JSON.stringify({ adapter: adapter.info, shape, reports }, null, 2)}\n`);
} finally {
  device.destroy();
}
