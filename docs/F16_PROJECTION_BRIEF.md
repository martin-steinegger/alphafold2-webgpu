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

## Accuracy gate, in order

1. Microbenchmark: the calibration's own accuracy check at both depths.
2. End to end: the same input predicted with and without the new kernel, same
   model, comparing mean pLDDT, pTM and predicted aligned error. How you
   toggle it is yours; do not ship a user-facing switch, the project has one
   model and no modes.
3. Absolute anchor: the 59-residue acceptance alignment reaches 96.058 pLDDT
   and 0.7536 pTM, against AlphaFold's own 96.0625 and 0.7534 recorded in
   `test/fixtures/evoformer/model1-a3m-59-stack/manifest.json`. A change that
   moves pLDDT by more than 0.05 has failed.

If pure f16 clears gate 3, ship it. If it does not, ship whichever of chunked
or mixed is fastest among those that do. Report the numbers either way.

## Guardrails

- `AGENTS.md` holds the project's invariants. The relevant ones: never change
  reference tensors or tolerances to make a check pass, every kernel needs a
  differential test against an independent reference, and correctness comes
  before performance.
- Half precision must be selected by capability, never by a user setting.
  Devices without `shader-f16` keep the f32 kernel, and the selection belongs
  next to the existing runtime calibration in `attention-calibration.ts`,
  which measures candidates on the real device and caches the winner.
- Commits in this repository carry no Claude attribution: no `Co-Authored-By`
  and no session trailer.
- Work on a branch named `f16-projection` and push the branch. Do not merge to
  `main`: another agent is changing kernel selection there.

## Files you own

`src/runtime/gemm.ts`, `src/evoformer/transition.ts`,
`tools/gemm-candidates.ts`, `test/browser/gemm-calibration.spec.ts`. Stay out
of everything else so the branches do not collide.

## Commands

```sh
npm ci
npx playwright install chromium
AFWEBGPU_GEMM_CALIBRATION=1 npx playwright test gemm-calibration --browser=chromium
npm run build && npx vitest run          # types and the CPU suite
AFWEBGPU_GPU_TESTS=1 npx vitest run      # the GPU suite, if dawn-node works there
```

For an end-to-end prediction without local model assets, run `npm run dev` and
point the page's monomer model URL at the deployed manifest:
`https://martin-steinegger.github.io/alphafold2-webgpu/model/manifest.json`.

## What to report

The accuracy table at both depths including the chunked candidate, the speed
table against the production baseline, the end-to-end before/after numbers,
and which kernel you shipped and why. If you ship nothing, say what failed
which gate.
