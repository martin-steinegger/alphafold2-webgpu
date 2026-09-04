# Half-precision projections on Apple Metal 3

Measured on an M4 Pro MacBook, Chromium, Apple Metal 3, on branch
`f16-projection`. Harness revision `6dc1068`.

## Conclusion

**Ship `f16-chunked`, with the k depth measured per device.** It runs the
projection shapes 1.12x to 1.26x faster than the f32 kernel and moves the
59-residue acceptance prediction by 0.008 pLDDT, against a gate of 0.05.
Across a whole prediction it is worth 1.044x, because the projections are only
part of a recycle.

**Pure f16 is rejected, and this is the main result.** It is much the fastest
candidate, at 1.28x to 1.54x, and it is unusable: the 508-row acceptance
alignment went from 96.80 pLDDT to 69.94 and its pTM to `NaN`. A `NaN` pTM
means the running sum reached infinity.

The cause is isolated, not inferred. All three arrangements stage their
operands in f16 identically; they differ only in where the sum is kept. The
two that reduce in f32 came back within 0.015 pLDDT of the f32 kernel on that
same input, so the staged half-precision operands are fine and it is
accumulating the whole of K in half precision that fails.

The microbenchmark did not predict this at all. It reported 0.948% worst
error, which reads like a tolerable rounding cost, because random operands of
uniform magnitude never build a partial sum large enough to overflow. Only the
end-to-end differential on the deep input found it, and only because the brief
demanded a second input.

Half precision therefore has to be bought in the arrangement that keeps the
reduction in f32, not in the one the microbenchmark ranked first.

## Adapter

```text
granted: shader-f16, subgroups, chromium-experimental-subgroup-matrix
adapter: apple metal-3
subgroup matrix configs:
  componentType=f32 resultComponentType=f32 M=8 N=8 K=8
  componentType=f16 resultComponentType=f16 M=8 N=8 K=8
```

`shader-f16` was not previously requested by `requestAlphaFoldDevice`, so no
device the model built had the feature at all.

## The three arrangements

Half precision is not one choice. Where the rounding goes decides both the
speed and whether the kernel is safe.

| arrangement | staged tiles | multiply | accumulate |
|---|---|---|---|
| `f16` | f16 | f16 | f16 over all of K |
| `f16-mixed` | f16 | f16 | f32 |
| `f16-chunked` | f16 | f16 | f16 within one k tile, f32 across tiles |

`f16-chunked` was the brief's unmeasured third arrangement. Its error grows
with the square root of the tile depth, 8 or 16 terms, rather than of K, while
every multiply and all but one add per tile stay in half precision.

## Accuracy screen

Worst error against a reference summed in the harness, as a percentage of the
largest value. Random inputs, which is the limitation this table has.

| candidate | K=256 | K=1024 |
|---|---|---|
| production (f32) | 0.000% | 0.000% |
| `tiled-64x128k16` (f32) | 0.000% | 0.000% |
| `f16-source-64x128k8` | 0.034% | 0.036% |
| `f16-mixed-64x128k8` | 0.021% | 0.030% |
| `f16-mixed-64x64k16` | 0.021% | 0.030% |
| `f16-chunked-64x128k8` | 0.044% | 0.046% |
| `f16-chunked-64x128k16` | 0.060% | 0.089% |
| `f16-chunked-64x64k16` | 0.060% | 0.089% |
| `f16-math-64x128k8` | 0.243% | 0.948% |
| `f16-math-64x128k16` | 0.243% | 0.948% |
| `f16-math-64x64k16` | 0.243% | 0.948% |
| `matrix-f32-8x8x8` | 0.000% | 0.000% |
| `matrix-f16-8x8x8` | 0.310% | 0.866% |

Chunked lands under 0.1% at both depths, as predicted, so it clears the screen
outright. Its error barely grows from K=256 to K=1024, which is the point of
the arrangement.

## Kernel speed, against the production f32 kernel

| shape | `f16` k8 | `f16` k16 | `chunked` k8 | `chunked` k16 | `mixed` k8 | f32 k16 |
|---|---|---|---|---|---|---|
| opm-contract | 1.40x | 1.45x | 1.17x | 1.26x | 1.14x | 1.07x |
| opm-out | 1.54x | 1.54x | 1.16x | 1.16x | 1.13x | 1.03x |
| opm-out2 | 1.28x | 1.29x | 1.15x | 1.21x | 1.11x | 1.01x |
| project | 1.36x | 1.35x | 1.13x | 1.15x | 1.11x | 1.02x |
| output | 1.37x | 1.36x | 1.12x | 1.15x | 1.12x | 1.02x |
| trans1 | 1.37x | 1.36x | 1.14x | 1.16x | 1.12x | 1.03x |
| trans2 | 1.39x | 1.39x | 1.14x | 1.19x | 1.12x | 1.04x |
| extra1 | 1.50x | 1.42x | 1.12x | 1.13x | 1.16x | 1.02x |

Chunked beats mixed at every shape, so mixed is kept only as a candidate the
per-device probe may still prefer elsewhere. The brief's threshold for chunked
was 1.3x; it reaches 1.26x at best and 1.15x typically, so the hoped-for
outcome — chunked being nearly as fast as pure f16 — did not happen. It is
still the fastest arrangement that is safe.

The f32 k16 tile is a consistent 1.01x to 1.07x for free on devices without
`shader-f16`, and is now measured rather than fixed.

## What half precision costs a prediction

Same weights, same input, same seed; only the projection kernel changes. The
q8 browser bundle `model_1_ptm-q8-v1`, since no f32 bundle is deployed and the
comparison is against itself. Gate: 0.05 pLDDT and 0.005 pTM at every recycle.

| variant | worst ΔpLDDT | worst ΔpTM | PAE MAE | atom37 RMS | verdict |
|---|---|---|---|---|---|
| `f32-k16` | 0.0000 | 0.00000 | 0.0000 | 0.0000 Å | passes |
| `f16-chunked-k8` | 0.0079 | 0.0002 | 0.0058 | 0.0042 Å | passes |
| `f16-chunked-k16` | 0.0098 | 0.0003 | 0.0064 | 0.0106 Å | passes |
| `f16-mixed-k8` | 0.0153 | 0.0002 | 0.0069 | 0.0036 Å | passes |
| `f16-mixed-k16` | 0.0153 | 0.0002 | 0.0069 | 0.0036 Å | passes |
| `f16-k8` | 26.8628 | `NaN` | `NaN` | `NaN` | **fails** |
| `f16-k16` | 26.8628 | `NaN` | `NaN` | `NaN` | **fails** |

Pure f16 recycle by recycle, query-only, showing the compounding the shallow
case does exhibit even before the deep case breaks:

| recycle | f32 pLDDT | f16 pLDDT | shift |
|---|---|---|---|
| 0 | 56.7213 | 56.7242 | 0.0029 |
| 1 | 59.9038 | 59.8973 | 0.0065 |
| 2 | 61.3672 | 61.3358 | 0.0314 |
| 3 | 63.2314 | 63.1510 | 0.0804 |

Four recycles are enough to take pure f16 past the gate on the easy input on
its own, before the deep MSA takes it to `NaN`.

Chunked, for comparison, on the same recycles: 56.7134, 59.9037, 61.3638,
63.2275, and 96.8032 against 96.8022 on the deep MSA.

## Whole-prediction timing

Four interleaved rounds, order reversed on alternate rounds, each variant
scored by its fastest round. Interleaving matters: run eight predictions in
blocks and the later ones are several percent slower whatever kernel they use,
which is enough to invert a ranking this close. A single pass in a fixed order
measured `f32-k16` at 0.92x, which is an artefact.

| variant | best | vs f32 |
|---|---|---|
| `f32-k8` | 4.44 s | 1.000x |
| `f32-k16` | 4.42 s | 1.005x |
| `f16-mixed-k8` | 4.36 s | 1.018x |
| `f16-chunked-k8` | 4.33 s | 1.026x |
| `f16-chunked-k16` | 4.26 s | 1.044x |

A 1.15x to 1.26x kernel is worth 1.044x here, so the dense projections are
roughly a fifth to a quarter of this prediction. That fraction is
input-dependent, and the 59-residue query is a small input.

## Complexes and long inputs

A 118-residue homodimer, query-only, one recycle, `f16-chunked-k16` against
`f32-k8`:

| recycle | ΔpLDDT | ΔpTM | ΔipTM |
|---|---|---|---|
| 0 | 0.0094 | 0.00012 | 0.00027 |
| 1 | 0.0131 | 0.00001 | 0.00011 |

The 24-chain, 1,416-residue homomer is reported in
`docs/LONG_COMPLEX_APPLE_METAL_3.md`.

## How the selection works

`src/runtime/gemm-selection.ts` measures the candidates on the real device
once, keeps the fastest that reproduces a reference, and caches it. Two
properties make this invisible to the call sites, which belong to another
branch:

- the accumulator is rebound to `vec4<f32>` before the epilogue, so every
  store and epilogue fragment a caller wrote keeps compiling; and
- the k depth never enters the dispatch grid, which `gemmGrid` still derives
  from the output tile alone.

The choice is installed inside `requestAlphaFoldDevice`, before the device is
handed out and so before any projection shader exists. Installing it later
lets one pipeline cache key describe two different shaders, which
`ComputePipelineCache` reports as a collision rather than running — this is
not hypothetical, it is what the differential harness hit when it reused a
device between variants.

Pure f16 is excluded from the candidate list in code, not left to the runtime
probe. A cheap probe cannot rediscover an overflow that needs a deep MSA's
depth and magnitudes to appear, so the exclusion is recorded where it can be
read.

Half precision must also beat f32 by a margin of 1.1x before it is chosen, so
an adapter that merely emulates f16 stays exact rather than trading accuracy
for measurement noise.

## A side effect worth someone's attention

`src/triangle/webgpu.ts` already accepts a `precision: "f16"` option, guards it
on `device.features.has("shader-f16")`, and defaults to `"f32"`. Until now that
guard could never pass, because no device the model built requested the
feature. Requesting it makes the triangle multiply's existing half-precision
path reachable on Apple for the first time.

Nothing here enables it. That file belongs to the other branch, and the
evidence above is a reason for caution rather than enthusiasm: the triangle
multiply reduces over the residue axis, which is exactly the long reduction
that made pure f16 unusable in the projections. If it is tried, it should be
tried against a deep MSA, recycle by recycle, and the `f16-chunked`
arrangement is the one to reach for.

## Subgroup matrix units

`matrix-f32-8x8x8` and `matrix-f16-8x8x8` remain slower than production, at
about 0.72x. The prototype gives one subgroup one 8x8 output tile and walks all
of K from storage with no reuse, so it is memory-bound by construction and says
nothing about the hardware. Untouched by this work.

## Commands

```sh
AFWEBGPU_GEMM_CALIBRATION=1 npx playwright test gemm-calibration --browser=chromium

# The end-to-end gates need a model bundle. Nothing is checked in; these use
# the deployed browser bundles.
export AFWEBGPU_QUALIFICATION_ASSET_ROOT=/path/to/assets   # model/, model-multimer/, acceptance/test.a3m
AFWEBGPU_GEMM_DIFFERENTIAL=1 npx playwright test gemm-differential --browser=chromium
AFWEBGPU_GEMM_TIMING=1 npx playwright test gemm-timing --browser=chromium
AFWEBGPU_LONG_COMPLEX=1 AFWEBGPU_COMPLEX_CHAINS=24 npx playwright test long-complex --browser=chromium

AFWEBGPU_GPU_TESTS=1 npx vitest run test/gemm-half-precision.gpu.test.ts
```
