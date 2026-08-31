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

Models 2–5 are not part of this release. Export official model 1 parameters from a ColabFold installation, verify the float32 reference bundle, then generate and verify the q8 web bundle:

```bash
python tools/export_alphafold_multimer_model.py \
  --data-dir ~/.cache/colabfold \
  --model-number 1 \
  --output /tmp/afwebgpu-multimer-model1-f32-v1
npm run verify:web-model -- \
  /tmp/afwebgpu-multimer-model1-f32-v1/manifest.json --require-sha256
npm run quantize:web-model -- \
  /tmp/afwebgpu-multimer-model1-f32-v1 \
  /tmp/model-multimer --format=int8
npm run verify:web-model -- \
  /tmp/model-multimer/manifest.json --require-sha256
tar -C /tmp -czf afwebgpu-model1-multimer-v3-q8-v1.tar.gz model-multimer
```

The exporter requires the official `WEIGHTS_LICENSE` from the parameter directory, copies it into the output, records its CC BY 4.0 provenance in the manifest, and hashes every shard. The quantizer preserves the license and adds a modification record. Do not publish a bundle that fails `--require-sha256` validation.

Before release, run the end-to-end differential qualification against an independently generated official JAX capture:

```bash
AFWEBGPU_GPU_TESTS=1 \
AFWEBGPU_MULTIMER_REFERENCES=/path/to/query/manifest.json,/path/to/paired/manifest.json \
AFWEBGPU_MULTIMER_F32_MANIFEST=/tmp/afwebgpu-multimer-model1-f32-v1/manifest.json \
AFWEBGPU_MULTIMER_Q8_MANIFEST=/tmp/model-multimer/manifest.json \
npx vitest run test/multimer-model-official.gpu.test.ts
```

This first compares WebGPU float32 with official JAX outputs, then q8 with WebGPU float32. It covers pLDDT, pTM, ipTM, ranking confidence, PAE, and atom coordinates with fixed thresholds. Chrome on macOS Apple Silicon is the release target and must be qualified before enabling the Pages bundle.

Upload the archive to release tag `model1-multimer-v3-q8-v1` as `afwebgpu-model1-multimer-v3-q8-v1.tar.gz`. Its top-level directory must be `model-multimer/` so the default browser URL resolves to `./model-multimer/manifest.json`.

## Immutability and Pages limits

Shard filenames include the bundle version and are persistently cached by exact URL. Increment the bundle version and filename suffix whenever model bytes or packing rules change. Never replace a released shard with different bytes under the same versioned filename.

The Pages workflow validates each enabled manifest and calls `npm run verify:pages-size` before upload. The verifier rejects a site over 900 MiB, leaving margin below GitHub's 1 GiB Pages limit. Keep monomer and Multimer release variables independent so a Pages deployment without the new Multimer release continues to work.
