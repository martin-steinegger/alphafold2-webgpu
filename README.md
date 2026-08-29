# AFWebGPU

AFWebGPU is an end-to-end WebGPU implementation of AlphaFold 2 monomer model 1. It accepts either a raw amino-acid sequence or A3M text, runs recycling, the extra-MSA stack, all 48 main Evoformer blocks, the eight-layer structure module, atom geometry, and pLDDT/PAE heads.

All learned model operations execute in WGSL. The CPU is used only for non-neural input preprocessing, scheduling, readback, and confidence aggregation. There is no ONNX runtime and no CPU neural-network fallback.

## Verified reference predictions

The acceptance sequence is:

```text
PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK
```

Against official `alphafold2_ptm_model_1_seed_000` FP32 intermediates:

| Input | Recycle | WebGPU pLDDT | Official pLDDT | WebGPU pTM | Official pTM |
|---|---:|---:|---:|---:|---:|
| Sequence only | 0 | 57.000 | 56.994 | 0.37598 | 0.37594 |
| Sequence only | 3 | 64.517 | 64.511 | 0.43466 | 0.43464 |
| `test.a3m` processed oracle | 0 | 96.615 | 96.625 | 0.75294 | 0.75293 |
| `test.a3m` processed oracle | 3 | 96.049 | 96.063 | 0.75355 | 0.75342 |

The literal 8,076-row `test.a3m`, independently parsed and clustered in TypeScript, produced 96.82 pLDDT and 0.7548 pTM after its first WebGPU recycle.

Development-host timings on an NVIDIA GB10 were 9.75 and 9.76 seconds for four A3M passes (508 clustered + 1,024 extra rows) when the optional key-parallel subgroup fast path was available. Run `npm run bench:a3m-model` to reproduce the full-model measurement. The original correctness-first scheduler took approximately 95 seconds. These are engineering measurements, not cross-device claims.

## Public API

Load a browser-hosted model manifest and predict a sequence:

```ts
import { AlphaFoldFixture, AlphaFoldQueryOnlyGpu, HttpTensorStore, requestAlphaFoldDevice } from "afwebgpu";

const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
if (adapter === null) throw new Error("WebGPU unavailable");
const device = await requestAlphaFoldDevice(adapter);

const model = AlphaFoldFixture.fromStore(
  await HttpTensorStore.open(new URL("/model/manifest.json", location.href)),
);
const [embedding, template, extraStack, mainStack, structure, confidence, geometry, featureTables] =
  await Promise.all([
    model.embeddingWeights(), model.templateWeights(), model.extraPairStackWeights(), model.mainStackWeights(),
    model.structureWeights(), model.confidenceWeights(), model.geometryTables(), model.queryOnlyFeatureTables(),
  ]);

const prediction = await new AlphaFoldQueryOnlyGpu(device).predictSequence(
  "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
  { embedding, template, extraStack, mainStack, structure,
    lddt: confidence.lddt, pae: confidence.pae, geometry },
  featureTables,
  { recycles: 3, randomSeed: 0 },
  await model.tensor("confidencePaeBreaks"),
);
console.log(prediction.final.confidence.meanPlddt, prediction.final.confidence.ptm);
```

For A3M input, load `extraStackWeights()` and call `AlphaFoldMonomerGpu.predictA3m(...)`. `makeA3mFeatures(...)` is also exported for applications that want preprocessing and inference as separate steps.

The manifest is a JSON tensor table whose values are little-endian float32 binary files. `HttpTensorStore` fetches tensors lazily and caches them. `FileTensorStore` provides the equivalent Node test/development loader.

## Architecture and memory

Implemented components include input/recycling embeddings, mock-template pair embedding for model 1, MSA row and column attention, extra-MSA global column attention, transitions, OuterProductMean, both triangle multiplication directions, both triangle attention orientations, IPA, backbone updates, sidechain torsions, atom14/atom37 geometry, pLDDT, PAE, and pTM.

Attention uses online softmax and never materializes attention-logit cubes. When `subgroups` and `subgroup-size-control` are available, eight 32-lane subgroups process eight queries while sharing a 32-key K/V tile. Within each subgroup, the lanes calculate 32 key scores concurrently, reduce the tile softmax, and shuffle probabilities while accumulating one output channel per lane. Other devices use the portable workgroup kernel. The earlier lane-per-channel 8x16, 8x32, 8x64, 16x64, 32x64, and 64x64 variants remain selectable for differential benchmarking. QKV/gate projections use register-blocked 16x16 tiles, attention output uses 16x32 tiles, and transition GEMMs use 16x64 tiles. Triangle multiplication uses cooperative 16x16 joint tiles for its split A/B projection and gates and for its output projection/gate, inspired by the fused split projection in [steineggerlab/alphafold](https://github.com/steineggerlab/alphafold), and never materializes an `O(L³)` tensor. For deep, short MSAs, OuterProductMean uses AF2's canonical outer-first contraction (13.6 MiB at length 59); a 32-sequence bounded path remains available when that temporary would exceed 64 MiB.

Evoformer blocks are submitted ahead without host waits and alias a pooled set of scratch buffers. WebGPU queue ordering preserves block dependencies while reducing the 48-block A3M trunk from about 21.7 seconds to about 2.3 seconds. Final projections commit directly into residual tensors, reducing the measured trunk scratch peak from 767 MB to approximately 664 MB. Embedding, extra-MSA, and main-stack activations stay device-resident across stage and recycle boundaries; only the first MSA row and pair representation required by the current structure API are read back.

## Browser demo and GitHub Pages

The static browser UI supports the one implemented parameter set, `model_1_ptm`, with either a single sequence or a custom A3M. It reports per-recycle pLDDT/pTM, renders pLDDT and PAE plots, displays the unrelaxed structure with 3Dmol, and exports PDB and score JSON files.

Build the UI alone, or—when the local full-model fixture is available—build a self-contained site with the reduced model bundle:

```bash
npm run build:web
npm run build:web:standalone
```

The model exporter copies only the 335 tensors required for inference and discards captured activations and reference outputs. It packs them into eight balanced binary shards to avoid hundreds of HTTP round trips. The resulting model-1 PTM download is approximately 355 MiB, versus 419 MiB for the query reference fixture and 1.1 GiB for the A3M reference fixture. It remains float32 to avoid silently changing predictions. Full-model captures are intentionally excluded from the published source history.

The Pages workflow deploys the UI on pushes to `main`. To keep Git history small, the reduced browser model is stored once in the private repository's `model1-ptm` GitHub Release as `afwebgpu-model1-ptm.tar.gz`; the workflow downloads that asset before constructing the Pages artifact. Because GitHub Pages is normally public even when its source repository is private, the workflow does not publish model parameters unless the repository variable `AFWEBGPU_INCLUDE_MODEL` is explicitly set to `true`. Without the bundled model, enter a CORS-enabled manifest URL in Advanced settings.

From a development checkout containing the full model fixture, prepare the release asset with:

```bash
npm run build:web:standalone
mkdir -p artifacts
tar -C dist/web -czf artifacts/afwebgpu-model1-ptm.tar.gz model
```

Create a private-repository release tagged `model1-ptm` and attach the archive. The archive contains a top-level `model/` directory, so the Pages workflow can extract it directly into the built site.

After pushing the repository, select **Settings → Pages → Source: GitHub Actions**. To publish the model with the demo, add the Actions variable under **Settings → Secrets and variables → Actions → Variables**. The full Pages artifact is about 356 MiB and remains below GitHub's 1 GiB Pages artifact/site limit.

## Development

```bash
npm install
npm run build
npm run build:web
npm test
npm run test:gpu
npm run test:browser
```

GPU tests use Dawn's native Node WebGPU implementation. The suite contains operator-level official AlphaFold differential tests, complete block/stack tests, four-recycle query-only inference, four-recycle A3M inference, and a literal raw-A3M acceptance test.

Small reference fixtures needed by the default test suite are committed. Full Evoformer/model captures can be regenerated with the scripts under `tools/`; they require a ColabFold/AlphaFold JAX environment and official model parameters and are not part of published Git history.

## Current scope

- Monomer `model_1_ptm`, FP32 inference, mock templates, and fixed model channel sizes are supported.
- A3M sampling and clustering are implemented in TypeScript with a deterministic application PRNG. It is distribution-equivalent to AlphaFold preprocessing but does not reproduce TensorFlow's private RNG stream unless exact masked-MSA codes are supplied.
- Multimer models, real template hits, other AF2 parameter sets, weight quantization, remote MSA search, and result relaxation remain outside the current scope.
- Further structure-module residency, weight-layout specialization, and Apple-GPU profiling can improve runtime and peak memory without changing model semantics.
