# AlphaFold2 WebGPU

<p align="center">
  <strong>Protein structure prediction directly in your browser</strong>
</p>

<p align="center">
  <a href="https://martin-steinegger.github.io/alphafold2-webgpu/"><strong>Open AlphaFold2 WebGPU</strong></a>
</p>

AlphaFold2 WebGPU runs AlphaFold2 model 1 PTM and AlphaFold-Multimer-v3 model 1 locally on a WebGPU-capable device. One sequence field accepts either a monomer or a colon-separated complex and selects the matching model automatically.

The neural network runs in WGSL on the GPU. There is no ONNX runtime, server-side model execution, or CPU neural-network fallback. The CPU is used only for input preprocessing, scheduling, readback, and confidence aggregation.

> [!WARNING]
> AlphaFold2 WebGPU is experimental research software. Predictions have not been validated for clinical, diagnostic, or other safety-critical use.

## Try it

Open the hosted application:

**https://martin-steinegger.github.io/alphafold2-webgpu/**

| Application | Monomers | Complexes | MMseqs2 MSA | Custom A3M | Templates | Relaxation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [AlphaFold2 WebGPU](https://martin-steinegger.github.io/alphafold2-webgpu/) | Yes | Yes | Yes | Yes | No | No |

The browser uses `model_1_ptm` for a single chain and `model_1_multimer_v3` for two or more colon-separated chains. Models 2–5 are intentionally not shipped. Separate manifest settings allow both bundles to remain available without manual switching.

## Quick start

1. Open [AlphaFold2 WebGPU](https://martin-steinegger.github.io/alphafold2-webgpu/) in a WebGPU-capable browser.
2. Paste a protein sequence using the one-letter amino-acid alphabet.
3. Choose an alignment mode:
   - **MMseqs2 MSA** searches for related sequences through the public ColabFold API and normally produces the best predictions.
   - **Single sequence** runs without a remote search and is useful for testing, but confidence can be substantially lower.
   - **Custom A3M** uses a monomer alignment or a ColabFold serialized complex A3M that you provide.
   Colon-separated sequences are detected as complexes. MMseqs2 mode generates ColabFold-style paired and unpaired complex MSAs; single-sequence mode creates a query-only complex input.
4. Choose the number of recycles and press **Fold protein**.
5. Inspect the MSA coverage, per-recycle confidence, pLDDT, PAE, and interactive 3D structure.
6. Download the predicted PDB, scores JSON, and generated A3M.

The first prediction downloads the selected model bundle and compiles WebGPU pipelines. The qualified monomer
mixed-q8 bundle is approximately 97 MiB, and the Multimer mixed-f16 bundle is approximately 182 MiB. Their
float32 qualification bundles are approximately 355 and 356 MiB respectively. Versioned model shards are
byte-length and SHA-256 validated and retained in the browser's persistent cache when storage policy permits.
Later predictions can be much faster; **Clear downloaded model** under Advanced settings removes both persistent
and in-memory copies.

## Input modes and privacy

### MMseqs2 MSA

The default mode submits the protein sequence to `https://api.colabfold.com`, polls the search ticket, and combines the returned UniRef and environmental A3Ms in the browser. Model inference remains local.

The public API is intended for interactive, serial use. Please do not send parallel or automated high-volume requests. Selecting this mode means that the sequence leaves your device and is processed by the remote service.

### Single sequence

Single-sequence mode creates an alignment containing only the query. Neither the sequence nor the prediction is sent to an MSA or inference service. AlphaFold2 usually benefits strongly from evolutionary information, so a low single-sequence confidence does not necessarily indicate an implementation problem.

### Custom A3M

Upload A3M text. An ordinary A3M uses its first ungapped FASTA entry as a monomer query. A ColabFold serialized complex A3M with a `#lengths<TAB>cardinalities` header is split back into its paired and unpaired per-chain alignments, cropped, and merged through the same Multimer path as a live MMseqs2 result. Custom A3M input and model inference stay on the device.

### Multimer-v3

Separate chains with colons, for example `ACDE:FGHI`. In MMseqs2 mode the browser requests both paired and unpaired per-entity alignments, then merges paired rows densely and unpaired rows block-diagonally as ColabFold does. Repeated homomer chains share the dense unpaired alignment and do not require a redundant pairing request. In single-sequence mode the complex is query-only and stays local.

Chain-relative, entity, and symmetry features follow the official Multimer-v3 encoding. Multimer MSA sampling, masked-MSA augmentation, nearest-neighbor clustering, and recycling keys reproduce ColabFold/AlphaFold's JAX `process_features` path, including partitionable Threefry keys. The structure module uses native Multimer-v3 Q/K/V projections, a position scale of 20, ipTM, and the `0.8 × ipTM + 0.2 × pTM` ranking score.

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

The browser must download model parameters and compile many GPU pipelines. Model bundles are split into eight balanced files to avoid hundreds of small HTTP requests. The page retains the resolved tensors, WebGPU device, and device-scoped pipeline cache, so repeating a prediction without closing the page avoids downloading and parsing the model again. A warm second run is the useful measure of inference performance.

### How are the model weights stored?

The loader accepts float32, float16, and mixed q8 bundles; neural-network arithmetic remains float32 after one-time decoding. Monomer model 1 is 355.3 MiB in float32 and 97.3 MiB in the qualified mixed-q8 form. Multimer model 1 is 355.5 MiB in float32 and 181.6 MiB in the qualified mixed-f16 form; its structure tensors remain float32. Multimer q8 failed the fixed paired-MSA pLDDT envelope and is not a release format. Every published shard has a declared SHA-256 digest, and the loader verifies it before use or persistent caching. The official parameters remain under DeepMind's CC BY 4.0 weights license, which is copied into each exported bundle.

### What is the maximum sequence length?

There is no single portable limit. Available GPU memory, WebGPU buffer limits, MSA depth, browser implementation, and adapter performance all matter. The application calculates both individual-buffer requirements and a conservative aggregate peak from sequence length and retained MSA depth instead of requesting a fixed limit. When the complete AF2 transition intermediate fits the adapter and memory budget it uses the fast path; otherwise it processes aligned row windows with at most 96 MiB of transition scratch. Chrome on macOS automatically uses bounded transitions and a bounded reusable-scratch pool even when adapter identity is hidden. Inputs that exceed a RAM-derived safety budget are rejected before GPU allocation with explicit suggested MSA row limits. Append `?compact=1` to force this policy on another platform. Persistent MSA memory grows as `Nseq × L`, while pair memory and compute grow at least as `L²`, so longer inputs can still take substantially longer.

### How are activations stored?

Weights are decoded once to float32 and every operation computes in float32. The activations the trunk carries between operations, the MSA, the pair, and the triangle multiplication's whole projection, are stored as half precision packed two to a 32-bit word, which needs no device feature. That is the only storage the model has: there is no exact mode to choose, and Multimer uses the same one. It halves the memory a prediction needs and, on real alignments of 164 to 396 residues, returned mean pLDDT and pTM identical to the float32 storage to two decimals; on a complex with a live ColabFold alignment, pLDDT 91.58 against 91.56 and ipTM 0.836 against 0.836. The float32 storage remains in the code as `EXACT_STORAGE` so the kernels can be compared against AlphaFold's own float32 tensors.

### Why is single-sequence confidence lower than the MSA prediction?

This is expected for many proteins. For the 59-residue acceptance sequence below, model 1 PTM starts near 57 pLDDT with a single sequence but reaches approximately 97 pLDDT with a deep MSA.

### Are complexes, templates, or Amber relaxation supported?

No-template AlphaFold-Multimer-v3 model 1 complexes are supported with ColabFold paired/unpaired MMseqs2 MSAs, ColabFold serialized complex A3Ms, or query-only input. Real template hits, models 2–5, and Amber relaxation are not supported.

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

To force the bounded transition path for portability testing, run:

```bash
AFWEBGPU_COMPACT=1 npm run bench:a3m-model
```

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
npm run qualify:hardware
```

GPU tests use Dawn's native Node WebGPU implementation. The suite includes operator-level official AlphaFold differential tests, complete block and stack tests, four-recycle single-sequence inference, four-recycle A3M inference, and a literal raw-A3M acceptance test.

The full-model hardware qualification runs both automatic and compact memory paths, checks all recycle confidence values against the official reference, and emits a machine-readable adapter/timing/memory report. See [hardware qualification](docs/HARDWARE_QUALIFICATION.md).

The browser model is distributed as a separate GitHub release asset. See [model bundle release](docs/MODEL_RELEASE.md) for the versioning, validation, packaging, and Pages deployment procedure.

For browser GPU profiling, append `?profile=1` to the development URL. The selected recycle's first extra-MSA and main Evoformer blocks report every dispatch through `timestamp-query` when available, with synchronized wall-clock block timing as the fallback. The `profileRecycle`, `profileExtraBlock`, and `profileMainBlock` query parameters select different zero-based targets.

Small reference fixtures required by the default tests are committed. Full Evoformer and model captures can be regenerated with scripts under `tools/`; they require a ColabFold/AlphaFold JAX environment and official model parameters and are intentionally excluded from published Git history.

## Architecture

Implemented components include input and recycling embeddings, mock-template pair embedding for model 1, MSA row and column attention, extra-MSA global column attention, transitions, OuterProductMean, both triangle multiplication directions, both triangle attention orientations, invariant point attention, backbone updates, sidechain torsions, atom14/atom37 geometry, pLDDT, PAE, and pTM.

Attention uses online softmax and never materializes attention-logit cubes. When `subgroups` and `subgroup-size-control` are available, eight 32-lane subgroups process eight queries while sharing a 32-key K/V tile. Other devices, including Chrome-on-Metal, use a register-resident kernel in which one invocation owns a complete 8- or 32-channel head and requires no workgroup barriers in the key loop. The original portable workgroup kernel remains selectable as an independent differential baseline.

QKV and gate projections use register-blocked 16x16 tiles, attention output uses 16x32 tiles, and transition GEMMs use 16x64 tiles. Triangle multiplication uses cooperative 16x16 joint tiles for split A/B projection, gates, output projection, and output gate. Its contraction intermediates are channel-major so each hidden-channel slice is contiguous, and it never materializes an `O(L³)` tensor. For deep, short MSAs, OuterProductMean uses AlphaFold2's canonical outer-first contraction; a bounded path is available when the temporary would exceed 64 MiB.

Device limits are negotiated per prediction shape. A capable adapter receives the exact full-transition requirement rounded to a WebGPU capability tier. Constrained adapters retain the same AF2 weights and operations but execute transition rows through aligned storage-buffer windows capped at 96 MiB. The compact path does not fall back to CPU or truncate the MSA silently; if even the persistent MSA or pair tensor exceeds the adapter limit, the application reports the required and available sizes.

Evoformer blocks are submitted ahead without host waits and alias a pooled set of scratch buffers. Compact execution uses bounded best-fit reuse so a larger physical allocation can serve a smaller logical tensor, while bind groups expose only the tensor's logical range. The unbounded accelerator path retains exact-size pooling. Final projections commit directly into residual tensors. Embedding, extra-MSA, and main-stack activations stay device-resident across stage and recycle boundaries. The structure module uploads and normalizes its invariant pair representation once for all eight IPA iterations.

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

The manifest is a JSON tensor table referencing versioned little-endian shards. Records may be float32, float16, or symmetric block-int8 with explicit scale offsets; every representation is decoded to validated float32 tensors before inference. `HttpTensorStore` fetches tensors lazily and caches them. `FileTensorStore` provides the equivalent Node test and development loader.

## Building and deploying the browser application

Build the UI alone or, when the local full-model fixture is available, build a site with the reduced model bundle:

```bash
npm run build:web
npm run build:web:standalone
```

The monomer exporter can produce and qualify its mixed-q8 bundle:

```bash
npm run export:web-model -- test/fixtures/evoformer/model1-query-59-stack/manifest.json \
  /tmp/afwebgpu-model-f32 --weights-license="$HOME/.cache/colabfold/params/LICENSE"
npm run verify:web-model -- /tmp/afwebgpu-model-f32/manifest.json --require-sha256 --require-license
npm run quantize:web-model -- /tmp/afwebgpu-model-f32 dist/web/model --format=int8
npm run verify:web-model -- dist/web/model/manifest.json --require-sha256 --require-license
AFWEBGPU_GPU_TESTS=1 AFWEBGPU_QUANTIZED_MANIFEST=dist/web/model/manifest.json \
  npx vitest run test/quantized-model.gpu.test.ts
```

The quantized qualification is deliberately separate from the float32 reference tests: it compares four-recycle pLDDT and pTM, final per-residue confidence, PAE, and atom coordinates against a float32 run without changing any existing reference tensor or tolerance.

The Multimer exporter copies the 355 tensors required for no-template-search ColabFold inference, including the learned mock-template pair stack and torsion rows, and packs them into eight balanced shards. To produce its qualified 181.6 MiB mixed-f16 bundle:

```bash
python tools/export_alphafold_multimer_model.py \
  --data-dir ~/.cache/colabfold --model-number 1 \
  --output /tmp/afwebgpu-multimer-model1-f32-v2
npm run verify:web-model -- /tmp/afwebgpu-multimer-model1-f32-v2/manifest.json --require-sha256
npm run quantize:web-model -- /tmp/afwebgpu-multimer-model1-f32-v2 \
  /tmp/model-multimer --format=float16
npm run verify:web-model -- /tmp/model-multimer/manifest.json --require-sha256
```

The browser automatically uses `./model-multimer/manifest.json` for colon-separated input and serialized ColabFold complex A3Ms. Official differential captures can be regenerated with `tools/capture_alphafold_multimer_reference.py`; pass one `--unpaired-a3m` and `--paired-a3m` file per unique entity to capture a paired/unpaired case. Qualification accepts comma-separated captures through `AFWEBGPU_MULTIMER_REFERENCES`, plus `AFWEBGPU_MULTIMER_F32_MANIFEST` and `AFWEBGPU_MULTIMER_COMPRESSED_MANIFEST`. It compares official JAX to WebGPU float32, then mixed-f16 to WebGPU float32 without changing reference data or tolerances.

Full-model captures are excluded from source history. The qualified monomer q8 bundle is stored under release tag
`model1-ptm-q8-v1`, while the previous `model1-ptm` float32 release is retained for rollback. The Multimer
mixed-f16 bundle is stored separately under tag `model1-multimer-v3-f16-v1`. The Pages workflow downloads the
enabled immutable assets and constructs the deployment artifact.

To prepare the release asset from a checkout containing the full fixture:

```bash
npm run build:web
mkdir -p artifacts
tar -C /tmp -czf artifacts/afwebgpu-model1-ptm-q8-v1.tar.gz model
```

Create a release tagged `model1-ptm-q8-v1` and attach the q8 archive. For Multimer, archive the f16 directory as
top-level `model-multimer/`, attach it to tag `model1-multimer-v3-f16-v1`, and set
`AFWEBGPU_INCLUDE_MULTIMER_MODEL=true`. `AFWEBGPU_INCLUDE_MODEL=true` independently enables the q8 monomer
bundle. The workflow verifies manifests, licenses, shard hashes, byte ranges, and a conservative 900 MiB
site-size ceiling before upload.

## Current scope

- Monomer `model_1_ptm` and no-template `model_1_multimer_v3` are supported with fixed model channel sizes. Monomer accepts float32 and qualified mixed block-int8 storage; Multimer accepts float32 and the qualified mixed-f16 bundle. Neural-network arithmetic remains float32 after one-time weight decoding.
- Multimer A3M sampling, masking, and clustering reproduce the JAX/ColabFold `process_features` keys and ordering. The monomer A3M path uses a deterministic application PRNG and remains distribution-equivalent rather than reproducing TensorFlow's private RNG stream.
- ColabFold paired/unpaired Multimer MSA preprocessing and serialized complex A3M upload are supported. Searched/custom Multimer templates, models 2–5, sub-eight-bit weight formats, and result relaxation are not supported; ColabFold's learned no-search mock-template path is included.
- Pair state remains GPU-resident through structure, confidence, and recycle boundaries. The PAE confidence head uses a fixed 16 MiB logits window and GPU-side expectation reduction instead of materializing and reading back the full `L² × 64` tensor. Renewed Apple-GPU profiling after these kernel rewrites remains release work.

## Acknowledgments and citation

AlphaFold2 WebGPU is an independent browser/WebGPU port. It did not originate the protein-structure prediction method or model parameters.

If you use AlphaFold2 WebGPU, please cite the original AlphaFold2 publication:

> Jumper J, Evans R, Pritzel A, et al. Highly accurate protein structure prediction with AlphaFold. *Nature* 596, 583–589 (2021). [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)

Please also consult and cite the [official AlphaFold repository](https://github.com/google-deepmind/alphafold) as appropriate. The AlphaFold parameters are subject to DeepMind's [CC BY 4.0 parameters license](https://github.com/google-deepmind/alphafold/blob/main/WEIGHTS_LICENSE); this repository does not alter their ownership or license.

Remote MSA generation follows the public API client used by [ColabFold](https://github.com/sokrypton/ColabFold). When using MMseqs2 MSA generation, please also cite:

> Mirdita M, Schütze K, Moriwaki Y, Heo L, Ovchinnikov S, Steinegger M. ColabFold: making protein folding accessible to all. *Nature Methods* 19, 679–682 (2022). [doi:10.1038/s41592-022-01488-1](https://doi.org/10.1038/s41592-022-01488-1)

We thank the AlphaFold and ColabFold developers for making their methods, software, parameters, and services available to the community. They were not involved in implementing AlphaFold2 WebGPU.
