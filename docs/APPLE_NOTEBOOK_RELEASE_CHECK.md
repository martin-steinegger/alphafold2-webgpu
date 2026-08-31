# Apple notebook release check

These instructions are for an engineering agent running the release candidate on an Apple Silicon Mac. Do not
change reference tensors, thresholds, model archives, or production code to make a check pass. Report a failure
with the complete command output, hardware details, commit SHA, macOS version, and Chrome version.

## Preconditions

- Work from the exact release-candidate commit supplied by the primary agent.
- Use Node.js 22 or newer, GitHub CLI authenticated for this repository, and stable Google Chrome.
- Keep at least 5 GiB of free disk space. The qualification downloads both float32 reference bundles and both
  compressed browser bundles.
- Do not publish a release, change repository variables, deploy Pages, or push commits from the notebook.

## Prepare the checkout and qualification assets

```bash
git status --short
git rev-parse HEAD
node --version
gh auth status
npm ci
npx playwright install chrome

qualification_root="$(mktemp -d /tmp/afwebgpu-apple-qualification.XXXXXX)"
gh release download model1-multimer-v3-qualification-v1 --dir "$qualification_root" \
  --pattern afwebgpu-model1-ptm-f32-v2.tar.gz \
  --pattern afwebgpu-model1-ptm-q8-v1.tar.gz \
  --pattern afwebgpu-model1-multimer-v3-f32-v2.tar.gz \
  --pattern afwebgpu-model1-multimer-v3-f16-v1.tar.gz \
  --pattern afwebgpu-multimer-references-v2.tar.gz \
  --pattern afwebgpu-browser-acceptance-v1.tar.gz

for archive in "$qualification_root"/*.tar.gz; do
  tar -xzf "$archive" -C "$qualification_root"
done
```

Keep `qualification_root` set in the same shell for the remaining commands.

## Record the Apple and browser environment

```bash
system_profiler SPHardwareDataType SPDisplaysDataType
sw_vers
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version
```

The machine must report an Apple chip. Chrome must expose WebGPU without a CPU neural-network fallback.

## Verify the immutable model bundles

```bash
npm run verify:web-model -- \
  "$qualification_root/afwebgpu-monomer-model1-f32-v2/manifest.json" \
  --require-sha256 --require-license
npm run verify:web-model -- \
  "$qualification_root/model/manifest.json" \
  --require-sha256 --require-license
npm run verify:web-model -- \
  "$qualification_root/afwebgpu-multimer-model1-f32-v2/manifest.json" \
  --require-sha256 --require-license
npm run verify:web-model -- \
  "$qualification_root/model-multimer/manifest.json" \
  --require-sha256 --require-license
```

All four commands must pass. Confirm the bundle IDs are `model_1_ptm-f32-v2`, `model_1_ptm-q8-v1`,
`model_1_multimer_v3-f32-v2`, and `model_1_multimer_v3-f16-v1`.

## Run source and stable-Chrome qualification

```bash
npm run check

AFWEBGPU_BROWSER_CHANNEL=chrome \
AFWEBGPU_BROWSER_MONOMER_QUALIFICATION=1 \
AFWEBGPU_BROWSER_MULTIMER_QUALIFICATION=1 \
AFWEBGPU_CUSTOM_MULTIMER_A3M="$qualification_root/acceptance/blablub.a3m" \
AFWEBGPU_QUALIFICATION_ASSET_ROOT="$qualification_root" \
npx playwright test \
  test/browser/monomer-compressed-qualification.spec.ts \
  test/browser/multimer-qualification.spec.ts \
  test/browser/multimer-homodimer-prediction.spec.ts \
  test/browser/multimer-custom-a3m.spec.ts \
  --workers=1 | tee apple-release-qualification.log
```

This is a release gate. Every test must pass without altered tolerances. It covers:

- monomer q8 versus float32 for four query-only recycles;
- monomer q8 versus float32 for a 508-clustered/1,024-extra deep MSA recycle;
- Multimer float32 versus independent official JAX captures;
- Multimer mixed-f16 versus float32 for query-only and paired/unpaired features;
- automatic Multimer routing through the unified UI for the homodimer and homotrimer; and
- the supplied full ColabFold complex A3M through three recycles.

## Report back

Return the following to the primary agent:

- exact `git rev-parse HEAD` value;
- Mac model, Apple chip, unified memory, macOS version, and Chrome version;
- pass/fail summary and `apple-release-qualification.log`;
- final pLDDT, pTM, and ipTM printed for the custom heterodimer;
- any WebGPU validation error, device loss, peak-memory rejection, or timeout.

Do not mark the release qualified if any required test is skipped.
