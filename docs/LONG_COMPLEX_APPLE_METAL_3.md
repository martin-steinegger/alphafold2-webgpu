# The 24-chain complex on Apple Metal 3

A 24-mer of the 59-residue acceptance sequence, 1,416 residues, query-only,
through the deployed `model_1_multimer_v3-f16-v1` bundle on an M4 Pro.

This exists to answer one question that is not about speed: does the
half-precision projection cost the project its longest runnable input? A
capacity ceiling that was always there looks exactly like one a kernel change
introduced, so the same spec, `test/browser/capacity.spec.ts`, runs on `main`
and on `f16-projection`. It imports nothing this branch added and pins no
kernel, so each branch uses whatever it would select for itself.

## Result

| branch | kernel | recycles | elapsed | mean pLDDT | pTM | ipTM |
|---|---|---|---|---|---|---|
| `main` | f32 64x128k8 | 0 | 746 s | 32.3469 | 0.19455 | 0.17473 |

The confidence is low because this is a 24-copy homomer with no alignment at
all, one sequence deep. That is expected and is not what the run measures: the
question is whether 1,416 residues completes and comes back finite.

Estimated peak allocation was 2,440 MiB — 502 MiB persistent, 605 MiB scratch,
433 MiB resident allowance — against the 5,734 MiB Apple unified-memory safety
budget, with bounded transitions and scratch pooling on. So the input is not
close to the memory ceiling at one recycle; it is roughly 12 minutes of GPU
work, at about 13 to 16 seconds per Evoformer block over 4 extra-MSA blocks
and 48 main blocks.

## Recycling

An earlier attempt at this input with one extra recycle killed the renderer
after 4.4 minutes, mid-way through the first pass, and it was running the f32
kernel at the time. Recycling retains the previous pass's positions and pair
representation, so at this length it costs memory as well as time, which makes
it a separate question from the one above rather than more of the same.

`AFWEBGPU_COMPLEX_RECYCLES` selects it, and it is measured on both branches
for the same reason the single pass is.

## Command

```sh
export AFWEBGPU_QUALIFICATION_ASSET_ROOT=/path/to/assets   # needs model-multimer/
AFWEBGPU_CAPACITY=1 AFWEBGPU_COMPLEX_CHAINS=24 AFWEBGPU_COMPLEX_RECYCLES=0 \
  npx playwright test capacity --browser=chromium
```
