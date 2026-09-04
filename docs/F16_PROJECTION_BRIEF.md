# Brief: half-precision projections on Apple

You are working on a MacBook, which is the only machine either of us has with
`shader-f16`. The Linux workstation that wrote this brief cannot compile a
single half-precision shader, which is why this work is yours.

## Goal

Decide whether the dense projections should compute in half precision on
devices that offer `shader-f16`, and if so, ship it. The f32 kernel stays as
the fallback for every device without the feature.

## What is already measured, on your machine

Chromium, Apple Metal 3, via `AFWEBGPU_GEMM_CALIBRATION=1 npx playwright test
gemm-calibration --browser=chromium`. Do not re-derive these.

| candidate | speed vs production | worst error, K=256 | worst error, K=1024 |
|---|---|---|---|
| production (f32) | 1.00x | 0.000% | 0.000% |
| `f16-math-*` (f16 multiply and accumulate) | 1.35–1.49x | 0.243% | 0.948% |
| `f16-mixed-*` (f16 multiply, f32 accumulate) | 1.09–1.15x | 0.021% | 0.030% |
| `f16-source-*` (packed f16 source, f32 math) | ~1.0x | 0.034% | 0.036% |
| `matrix-f32-8x8x8` | 0.72x | 0.000% | 0.000% |

The subgroup-matrix prototype gives one subgroup one 8x8 output tile and walks
all of K from storage, with no reuse: it is memory-bound by construction, so
it is a poor kernel rather than evidence against the matrix units. Leave it
alone unless everything else is done.

The model already accepts 0.036% error from packed half-precision storage, so
`f16-mixed` costs nothing it has not already paid.

## The question to settle first

A third arrangement is unmeasured and may dominate both: accumulate in f16
within one k-tile and add that into an f32 running sum once per tile. The
error then grows with the square root of the tile depth (8 or 16) rather than
of K, so it should land near 0.05% while most multiplies stay in half
precision. Add it to `tools/gemm-candidates.ts` as `f16-chunked-*` and
measure. If it reaches 1.3x at under 0.1%, it is the answer and the rest of
this brief is about shipping it.

## Accuracy gate

The end-to-end result decides. The microbenchmark is a screen, not a gate: a
worst-case error on random inputs does not predict the model's output error,
because real activations are not random, errors partly cancel, and recycling
is self-correcting. A candidate that fails the screen has to work harder to
pass, it is not disqualified.

1. **Screen.** The calibration's accuracy check at both depths. Under 0.1% and
   the candidate is as safe as the packed storage already shipped. Over 0.5%
   and it owes the extra evidence in step 2.
2. **Differential prediction.** The same input predicted with and without the
   new kernel, same model and same weights, compared recycle by recycle on
   mean pLDDT, pTM and predicted aligned error. Under 0.05 pLDDT and 0.005 pTM
   at every recycle, it passes. A candidate that failed the screen must show
   this on at least two inputs, one of them a complex, since a single sequence
   can pass by luck and errors can compound across recycles.
3. How you toggle the kernel for that comparison is yours. Do not ship a
   user-facing switch: the project has one model and no modes.

Ship the fastest candidate that passes. Report the numbers either way,
including for the candidates that did not.

The absolute anchor, 96.0625 pLDDT and 0.75342 pTM in
`test/fixtures/alphafold-a3m/model1-reference-59/manifest.json`, is not a gate
for this work: it measures the model, not the kernel, and no machine today has
both an f32 bundle and `shader-f16`. The f32 kernel already matches it to
0.005. Once the branch lands, that check runs where the bundle is.

## Guardrails

- `AGENTS.md` holds the project's invariants. The relevant ones: never change
  reference tensors or tolerances to make a check pass, every kernel needs a
  differential test against an independent reference, and correctness comes
  before performance.
- Half precision must be selected by capability, never by a user setting.
  Devices without `shader-f16` keep the f32 kernel.
- Put the selection in a new `src/runtime/gemm-selection.ts`. It owns both
  questions, which precision and which tile, since both are "measure the
  candidates on this device once and cache the winner". Model it on
  `src/evoformer/attention-calibration.ts`, which already does exactly that
  for the flash kernels, but do not edit that file: it is outside your scope
  and a new file cannot conflict.
- Tile selection is yours as well. The other agent will not build a competing
  mechanism. What is known: this workstation prefers a 64x64 tile for the two
  outer-product shapes and 128x128 for the wide ones, and your machine
  disagreed, which is the whole reason the choice should be measured rather
  than written down.
- Commits in this repository carry no Claude attribution: no `Co-Authored-By`
  and no session trailer.
- Work on a branch named `f16-projection` and push the branch. Do not merge to
  `main`: another agent is changing kernel selection there.

## Files you own

`src/runtime/gemm.ts`, `src/evoformer/transition.ts`,
`tools/gemm-candidates.ts`, `test/browser/gemm-calibration.spec.ts`, and new
files under `src/runtime/`. Not the GEMM callers: `attention.ts`, `block.ts`,
`outer-product-mean.ts` and `triangle/shaders.ts` belong to the other agent.
You should not need them, as you observed: the epilogue contract holds if the
accumulator arrives at it as `vec4<f32>`. If a design needs a caller changed,
say so rather than reaching for it.

## Commands

```sh
npm ci
npx playwright install chromium
AFWEBGPU_GEMM_CALIBRATION=1 npx playwright test gemm-calibration --browser=chromium
npm run build && npx vitest run          # types and the CPU suite
AFWEBGPU_GPU_TESTS=1 npx vitest run      # the GPU suite, if dawn-node works there
```

There are no model bundles on that machine and only the quantised one is
deployed. That is enough: gate 2 compares the same model against itself, so
what the weights are does not matter, only that both sides use the same ones.
Run `npm run dev` and point the page's monomer model URL at the deployed
manifest,
`https://martin-steinegger.github.io/alphafold2-webgpu/model/manifest.json`.
Report which bundle you used.

## What to report

The accuracy table at both depths including the chunked candidate, the speed
table against the production baseline, the end-to-end before/after numbers,
and which kernel you shipped and why. If you ship nothing, say what failed
which gate.
