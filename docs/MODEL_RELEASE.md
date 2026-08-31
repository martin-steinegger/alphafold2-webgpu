# Model bundle release

The browser models are release assets rather than Git objects. GitHub Pages downloads the monomer bundle when `AFWEBGPU_INCLUDE_MODEL=true` and the Multimer bundle when `AFWEBGPU_INCLUDE_MULTIMER_MODEL=true`.

## Monomer model 1 PTM

Build and validate the current bundle:

```bash
npm run build:web:standalone
npm run verify:web-model -- dist/web/model/manifest.json
tar -czf afwebgpu-model1-ptm.tar.gz -C dist/web model
```

Upload the archive to the `model1-ptm` release. It must contain `model/manifest.json` and its sibling shards. The workflow retains a compatibility migration for the legacy monomer bundle, then validates it before deployment.

## AlphaFold-Multimer-v3 model 1

Models 2–5 are not part of this release. Export official model 1 parameters from a ColabFold installation, verify the float32 reference bundle, then generate and verify the mixed-f16 web bundle:

```bash
python tools/export_alphafold_multimer_model.py \
  --data-dir ~/.cache/colabfold \
  --model-number 1 \
  --output /tmp/afwebgpu-multimer-model1-f32-v2
npm run verify:web-model -- \
  /tmp/afwebgpu-multimer-model1-f32-v2/manifest.json --require-sha256
npm run quantize:web-model -- \
  /tmp/afwebgpu-multimer-model1-f32-v2 \
  /tmp/model-multimer --format=float16
npm run verify:web-model -- \
  /tmp/model-multimer/manifest.json --require-sha256
tar -C /tmp -czf afwebgpu-model1-multimer-v3-f16-v1.tar.gz model-multimer
```

The exporter requires the official `WEIGHTS_LICENSE` from the parameter directory, copies it into the output, records its CC BY 4.0 provenance in the manifest, and hashes every shard. The quantizer preserves the license and adds a modification record. Do not publish a bundle that fails `--require-sha256` validation.

Before release, run the end-to-end differential qualification against an independently generated official JAX capture:

```bash
AFWEBGPU_GPU_TESTS=1 \
AFWEBGPU_MULTIMER_REFERENCES=/path/to/query/manifest.json,/path/to/paired/manifest.json \
AFWEBGPU_MULTIMER_F32_MANIFEST=/tmp/afwebgpu-multimer-model1-f32-v2/manifest.json \
AFWEBGPU_MULTIMER_COMPRESSED_MANIFEST=/tmp/model-multimer/manifest.json \
npx vitest run test/multimer-model-official.gpu.test.ts
```

This first compares WebGPU float32 with official JAX outputs, then mixed-f16 with WebGPU float32. It covers pLDDT, pTM, ipTM, ranking confidence, PAE, and atom coordinates with fixed thresholds. The release workflow runs these comparisons in Chrome, plus the 118-residue homodimer smoke test, on macOS Apple Silicon before the Pages bundle is enabled.

For repeatable Apple Silicon qualification, create a draft release tagged `model1-multimer-v3-qualification-v1` containing `afwebgpu-model1-multimer-v3-f32-v2.tar.gz`, `afwebgpu-model1-multimer-v3-f16-v1.tar.gz`, and `afwebgpu-multimer-references-v2.tar.gz`. Dispatch **Qualify Multimer on Apple Silicon**; the manual workflow runs Chrome against both query-only and paired references and the homodimer smoke case on GitHub's `macos-15` Apple Silicon runner without deploying Pages. Promote the f16 asset only after that workflow succeeds.

Upload the archive to release tag `model1-multimer-v3-f16-v1` as `afwebgpu-model1-multimer-v3-f16-v1.tar.gz`. Its top-level directory must be `model-multimer/` so the default browser URL resolves to `./model-multimer/manifest.json`.

## Immutability and Pages limits

Shard filenames include the bundle version and are persistently cached by exact URL. Increment the bundle version and filename suffix whenever model bytes or packing rules change. Never replace a released shard with different bytes under the same versioned filename.

The Pages workflow validates each enabled manifest and calls `npm run verify:pages-size` before upload. The verifier rejects a site over 900 MiB, leaving margin below GitHub's 1 GiB Pages limit. Keep monomer and Multimer release variables independent so a Pages deployment without the new Multimer release continues to work.

Push-triggered deployment is additionally gated by `AFWEBGPU_AUTO_DEPLOY=true`. A manual Pages dispatch always runs. Keep automatic deployment disabled while qualifying a new model so pushing its workflow cannot alter the current live site; enable it only after qualification and smoke testing.
