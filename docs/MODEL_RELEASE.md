# Model bundle release

The browser model is not committed to Git. GitHub Pages downloads the single `model1-ptm` release asset when the repository variable `AFWEBGPU_INCLUDE_MODEL` is `true`.

Build and validate the current FP32 bundle:

```bash
npm run build:web:standalone
npm run verify:web-model
tar -czf afwebgpu-model1-ptm.tar.gz -C dist/web model
```

Upload `afwebgpu-model1-ptm.tar.gz` to the `model1-ptm` GitHub release, replacing its previous asset, and redeploy Pages. The archive must contain `model/manifest.json` and its eight sibling shards.

The Pages workflow also contains a one-time compatibility migration. If it downloads a legacy unversioned bundle, it renames the shards, adds the v1 metadata, validates every byte range, and replaces the release asset with the repository-scoped Actions token. Once the release is versioned, later deployments skip this migration and re-upload.

Shard filenames include the bundle version and are persistently cached by exact URL. When any model byte or packing rule changes, increment the bundle version and filename suffix in `tools/export-web-model.ts` before publishing. Never replace a released shard with different bytes under the same versioned filename.

The validation command checks declared sizes, tensor byte ranges, shard membership, and the expected bundle schema without rehashing all 355 MiB in the browser.
