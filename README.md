# AlphaFold2 WebGPU

<p align="center">
  <strong>Protein structure prediction directly in your browser</strong>
</p>

<p align="center">
  <a href="https://martin-steinegger.github.io/alphafold2-webgpu/"><strong>Open AlphaFold2 WebGPU</strong></a>
</p>

AlphaFold2 WebGPU runs AlphaFold2 model 1 PTM locally on a WebGPU-capable device. Paste a protein sequence, generate an MSA through the public ColabFold MMseqs2 API or upload your own A3M, and inspect the predicted structure, confidence, and alignment in the browser.

The neural network runs in WGSL on the GPU. There is no ONNX runtime, server-side model execution, or CPU neural-network fallback. The CPU is used only for input preprocessing, scheduling, readback, and confidence aggregation.

> [!WARNING]
> AlphaFold2 WebGPU is experimental research software. Predictions have not been validated for clinical, diagnostic, or other safety-critical use.

## Try it

Open the hosted application:

**https://martin-steinegger.github.io/alphafold2-webgpu/**

| Application | Monomers | Complexes | MMseqs2 MSA | Custom A3M | Templates | Relaxation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [AlphaFold2 WebGPU](https://martin-steinegger.github.io/alphafold2-webgpu/) | Yes | No | Yes | Yes | No | No |

Only the AlphaFold2 `model_1_ptm` parameter set is supported. Keeping a single model reduces the model download and keeps the browser implementation focused and testable.

## Quick start

1. Open [AlphaFold2 WebGPU](https://martin-steinegger.github.io/alphafold2-webgpu/) in a WebGPU-capable browser.
2. Paste a protein sequence using the one-letter amino-acid alphabet.
3. Choose an input mode:
   - **MMseqs2 MSA** searches for related sequences through the public ColabFold API and normally produces the best predictions.
   - **Single sequence** runs without a remote search and is useful for testing, but confidence can be substantially lower.
   - **Custom A3M** uses an alignment that you provide.
4. Choose the number of recycles and press **Fold protein**.
5. Inspect the MSA coverage, per-recycle confidence, pLDDT, PAE, and interactive 3D structure.
6. Download the predicted PDB, scores JSON, and generated A3M.

The first prediction downloads approximately 355 MiB of model parameters and compiles WebGPU pipelines. Later predictions can be much faster when browser and operating-system caches are warm.

## Input modes and privacy

### MMseqs2 MSA

The default mode submits the protein sequence to `https://api.colabfold.com`, polls the search ticket, and combines the returned UniRef and environmental A3Ms in the browser. Model inference remains local.

The public API is intended for interactive, serial use. Please do not send parallel or automated high-volume requests. Selecting this mode means that the sequence leaves your device and is processed by the remote service.

### Single sequence

Single-sequence mode creates an alignment containing only the query. Neither the sequence nor the prediction is sent to an MSA or inference service. AlphaFold2 usually benefits strongly from evolutionary information, so a low single-sequence confidence does not necessarily indicate an implementation problem.

### Custom A3M

Paste or upload A3M text. The first FASTA entry must be the ungapped query sequence. Custom A3M input and model inference stay on the device.

## Results

The application provides:

- An identity-sorted MSA coverage visualization.
- pLDDT and pTM for every recycle.
- A per-residue pLDDT plot.
- A predicted aligned error (PAE) plot.
- An interactive 3D structure colored by pLDDT.
- An unrelaxed PDB file with pLDDT stored in the B-factor field.
- A scores JSON file containing confidence values and PAE.
- The generated or supplied A3M alignment.

## FAQ

### Which browser should I use?

Use a current browser with WebGPU enabled. Current Chrome is the primary development target. Browser, operating-system, driver, and GPU differences can affect both feature availability and speed.

### Does this run on a MacBook?

It is designed to use standards-compliant WebGPU and should run on WebGPU-capable Apple Silicon Macs. Apple-specific profiling and tuning are ongoing; please report the Mac model, chip, memory, macOS version, browser version, and timing when filing a performance issue.

### Why is the first run slow?

The browser must download roughly 355 MiB of float32 model parameters and compile many GPU pipelines. The parameters are split into eight balanced files to avoid hundreds of small HTTP requests. A warm second run is the useful measure of inference performance.

### Why is the model download still large?

The application uses the original float32 `model_1_ptm` parameters. Quantization or float16 storage would reduce the download, but can change numerical behavior and is not enabled silently.

### What is the maximum sequence length?

There is no single portable limit. Available GPU memory, WebGPU buffer limits, MSA depth, browser implementation, and adapter performance all matter. The current project is optimized and validated first on short monomers; longer inputs may exhaust memory or take substantially longer.

### Why is single-sequence confidence lower than the MSA prediction?

This is expected for many proteins. For the 59-residue acceptance sequence below, model 1 PTM starts near 57 pLDDT with a single sequence but reaches approximately 97 pLDDT with a deep MSA.

### Are complexes, templates, or Amber relaxation supported?

No. The current implementation supports monomer `model_1_ptm`, mock templates, and unrelaxed output. AlphaFold-Multimer, real template hits, other parameter sets, and Amber relaxation are outside the current scope.

### Is this the same as ColabFold?

No. ColabFold is a complete protein-folding pipeline with multiple models and deployment options. AlphaFold2 WebGPU uses the ColabFold public MMseqs2 API for optional MSA generation, then runs its own independent WebGPU port of AlphaFold2 model 1 PTM locally in the browser.

### Can the predicted PDB be used for molecular replacement?

The PDB B-factor column contains pLDDT, where higher values mean higher predicted confidence. Crystallographic programs normally interpret a conventional B-factor in the opposite direction, so convert or otherwise account for this before molecular-replacement workflows.

## Verified reference predictions

The acceptance sequence is:

```text
PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK
```

Against official `alphafold2_ptm_model_1_seed_000` FP32 intermediates:

| Input | Recycle | WebGPU pLDDT | Official pLDDT | WebGPU pTM | Official pTM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Sequence only | 0 | 57.000 | 56.994 | 0.37598 | 0.37594 |
| Sequence only | 3 | 64.517 | 64.511 | 0.43466 | 0.43464 |
| `test.a3m` processed oracle | 0 | 96.615 | 96.625 | 0.75294 | 0.75293 |
| `test.a3m` processed oracle | 3 | 96.049 | 96.063 | 0.75355 | 0.75342 |

The literal 8,076-row `test.a3m`, independently parsed and clustered in TypeScript, produced 96.82 pLDDT and 0.7548 pTM after its first WebGPU recycle.

## Performance

Development-host timings on an NVIDIA GB10 were 9.75 and 9.76 seconds for four A3M passes with 508 clustered and 1,024 extra rows when the optional key-parallel subgroup fast path was available. The original correctness-first scheduler took approximately 95 seconds.

Run `npm run bench:a3m-model` to reproduce the full-model measurement. These values are engineering measurements from one adapter and software configuration, not cross-device performance claims. Compare warm medians on the same machine before and after an optimization.

## Installation for development

Node.js 22 or newer is required.

```bash
git clone https://github.com/martin-steinegger/alphafold2-webgpu.git
cd alphafold2-webgpu
npm install
npm run check
npm run dev
```

Open `http://127.0.0.1:4173/`. If a local model bundle is unavailable, enter this CORS-enabled manifest under **Advanced settings**:

```text
https://martin-steinegger.github.io/alphafold2-webgpu/model/manifest.json
```

Useful commands:

```bash
npm run build
npm run build:web
npm test
npm run test:gpu
npm run test:browser
npm run bench:attention
npm run bench:a3m-model
```

GPU tests use Dawn's native Node WebGPU implementation. The suite includes operator-level official AlphaFold differential tests, complete block and stack tests, four-recycle single-sequence inference, four-recycle A3M inference, and a literal raw-A3M acceptance test.

Small reference fixtures required by the default tests are committed. Full Evoformer and model captures can be regenerated with scripts under `tools/`; they require a ColabFold/AlphaFold JAX environment and official model parameters and are intentionally excluded from published Git history.

## Architecture

Implemented components include input and recycling embeddings, mock-template pair embedding for model 1, MSA row and column attention, extra-MSA global column attention, transitions, OuterProductMean, both triangle multiplication directions, both triangle attention orientations, invariant point attention, backbone updates, sidechain torsions, atom14/atom37 geometry, pLDDT, PAE, and pTM.

Attention uses online softmax and never materializes attention-logit cubes. When `subgroups` and `subgroup-size-control` are available, eight 32-lane subgroups process eight queries while sharing a 32-key K/V tile. Other devices use the portable workgroup kernel. Earlier lane-per-channel variants remain selectable for differential benchmarking.

QKV and gate projections use register-blocked 16x16 tiles, attention output uses 16x32 tiles, and transition GEMMs use 16x64 tiles. Triangle multiplication uses cooperative 16x16 joint tiles for split A/B projection, gates, output projection, and output gate. It never materializes an `O(L³)` tensor. For deep, short MSAs, OuterProductMean uses AlphaFold2's canonical outer-first contraction; a bounded path is available when the temporary would exceed 64 MiB.

Evoformer blocks are submitted ahead without host waits and alias a pooled set of scratch buffers. Final projections commit directly into residual tensors. Embedding, extra-MSA, and main-stack activations stay device-resident across stage and recycle boundaries; only the first MSA row and pair representation required by the current structure API are read back.

## TypeScript API

Load a browser-hosted model manifest and predict a sequence:

```ts
import {
  AlphaFoldFixture,
  AlphaFoldQueryOnlyGpu,
  HttpTensorStore,
  requestAlphaFoldDevice,
} from "afwebgpu";

const adapter = await navigator.gpu.requestAdapter({
  powerPreference: "high-performance",
});
if (adapter === null) throw new Error("WebGPU unavailable");

const device = await requestAlphaFoldDevice(adapter);
const model = AlphaFoldFixture.fromStore(
  await HttpTensorStore.open(new URL("/model/manifest.json", location.href)),
);

const [
  embedding,
  template,
  extraStack,
  mainStack,
  structure,
  confidence,
  geometry,
  featureTables,
] = await Promise.all([
  model.embeddingWeights(),
  model.templateWeights(),
  model.extraPairStackWeights(),
  model.mainStackWeights(),
  model.structureWeights(),
  model.confidenceWeights(),
  model.geometryTables(),
  model.queryOnlyFeatureTables(),
]);

const prediction = await new AlphaFoldQueryOnlyGpu(device).predictSequence(
  "PIAQIHILEGRSDEQKETLIREVSEAISRSLDAPLTSVRVIITEMAKGHFGIGGELASK",
  {
    embedding,
    template,
    extraStack,
    mainStack,
    structure,
    lddt: confidence.lddt,
    pae: confidence.pae,
    geometry,
  },
  featureTables,
  { recycles: 3, randomSeed: 0 },
  await model.tensor("confidencePaeBreaks"),
);

console.log(
  prediction.final.confidence.meanPlddt,
  prediction.final.confidence.ptm,
);
```

For A3M input, load `extraStackWeights()` and call `AlphaFoldMonomerGpu.predictA3m(...)`. `makeA3mFeatures(...)` is exported for applications that want preprocessing and inference as separate steps.

The manifest is a JSON tensor table referencing little-endian float32 binary files. `HttpTensorStore` fetches tensors lazily and caches them. `FileTensorStore` provides the equivalent Node test and development loader.

## Building and deploying the browser application

Build the UI alone or, when the local full-model fixture is available, build a site with the reduced model bundle:

```bash
npm run build:web
npm run build:web:standalone
```

The exporter copies only the 335 tensors required for inference and discards captured activations and reference outputs. It packs the tensors into eight balanced binary shards. The model-1 PTM download is approximately 355 MiB, compared with 419 MiB for the query reference fixture and 1.1 GiB for the A3M reference fixture. It remains float32 to avoid silently changing predictions.

Full-model captures are excluded from source history. The reduced browser model is stored in the repository's `model1-ptm` GitHub Release as `afwebgpu-model1-ptm.tar.gz`. The Pages workflow downloads that asset and constructs the deployment artifact.

To prepare the release asset from a checkout containing the full fixture:

```bash
npm run build:web:standalone
mkdir -p artifacts
tar -C dist/web -czf artifacts/afwebgpu-model1-ptm.tar.gz model
```

Create a release tagged `model1-ptm` and attach the archive. It must contain a top-level `model/` directory. Select **Settings → Pages → Source: GitHub Actions** to deploy the site.

## Current scope

- Monomer `model_1_ptm`, float32 inference, mock templates, and fixed model channel sizes are supported.
- A3M sampling and clustering are implemented in TypeScript with a deterministic application PRNG. They are distribution-equivalent to AlphaFold preprocessing but do not reproduce TensorFlow's private RNG stream unless exact masked-MSA codes are supplied.
- Multimer models, real template hits, other AlphaFold2 parameter sets, weight quantization, and result relaxation are not supported.
- Structure-module residency, weight-layout specialization, and Apple-GPU profiling remain opportunities to improve speed and peak memory without changing model semantics.

## Acknowledgments and citation

AlphaFold2 WebGPU is an independent browser/WebGPU port. It did not originate the protein-structure prediction method or model parameters.

If you use AlphaFold2 WebGPU, please cite the original AlphaFold2 publication:

> Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021). [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

Please also consult and cite the [official AlphaFold repository](https://github.com/google-deepmind/alphafold) as appropriate. The AlphaFold parameters are subject to DeepMind's [CC BY 4.0 parameters license](https://github.com/google-deepmind/alphafold/blob/main/WEIGHTS_LICENSE); this repository does not alter their ownership or license.

Remote MSA generation follows the public API client used by [ColabFold](https://github.com/sokrypton/ColabFold). When using MMseqs2 MSA generation, please also cite:

> Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. ColabFold: making protein folding accessible to all. *Nature Methods* 19, 679–682 (2022). [doi:10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)

We thank the AlphaFold and ColabFold developers for making their methods, software, parameters, and services available to the community. They were not involved in implementing AlphaFold2 WebGPU.
