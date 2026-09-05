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
| `f16-projection` | f16-chunked 64x128k16 | 0 | 668 s | 32.3388 | 0.19457 | 0.17472 |
| `f16-projection` | f16-chunked 64x128k16 | 1 | 1,334 s | 33.7754 | 0.19168 | 0.17200 |
| `f16-projection`, with the attention work | matrix 64x128, 1 query/invocation | 0 | **407 s** | 32.3419 | 0.19456 | 0.17474 |

**The final number is 407 seconds against `main`'s 746: 1.83x.** Almost all of
that is the two attention changes in `docs/ATTENTION_APPLE_METAL_3.md` rather
than the projections — the projections are 1.12x of it and the attention path
the rest. Everything is selected by measurement; nothing is forced.

**The input still runs, and it runs 1.117x faster.** That is a much larger
gain than the 1.044x the 59-residue monomer showed, which is what you would
expect: the dense projections are a bigger share of the work at 1,416
residues, so the same kernel is worth more of it. This is the longest input
either branch can predict and it is also where half precision pays best.

The confidence is low because this is a 24-copy homomer with no alignment at
all, one sequence deep. That is expected and is not what the run measures: the
question is whether 1,416 residues completes and comes back finite.

The kernel in the second row was identified rather than read off, because
that run predates a fix to the capacity probe: it asked which variant was
installed before the prediction rather than after, and so reported the f32
default. The third row read it correctly and confirms it, on the same adapter
and the same selector, as `f16-chunked-64x128k16`. Its recycle 0 also
reproduces the second row exactly — 32.3388, 0.19457, 0.17472 — so the kernel
is deterministic run to run and the two rows ran the same one.

Mean pLDDT moves by 0.008 and pTM and ipTM by 0.00002, far inside the gate.
An earlier run of this input recorded confidence identical to `main` in every
printed digit, which looked like the kernel toggle having failed; it was the
probe having selected f32 that time, before its timing batch was fixed. Note
also that a mean over 1,416 residues damps per-residue perturbation by about
a factor of 38, so this input is a weak accuracy test however it comes out —
the 59-residue differential is the sensitive one.

Estimated peak allocation was 2,440 MiB — 502 MiB persistent, 605 MiB scratch,
433 MiB resident allowance — against the 5,734 MiB Apple unified-memory safety
budget, with bounded transitions and scratch pooling on. So the input is not
close to the memory ceiling at one recycle; it is roughly 12 minutes of GPU
work, at about 13 to 16 seconds per Evoformer block over 4 extra-MSA blocks
and 48 main blocks.

## Recycling

An earlier attempt at this input with one extra recycle killed the renderer
after 4.4 minutes, mid-way through the first pass, while running the f32
kernel. Recycling retains the previous pass's positions and pair
representation, so at this length it costs memory as well as time, and that
looked like the explanation.

**It is not, and the crash does not reproduce.** The same input with one extra
recycle now completes in 1,334 seconds, both passes, on `f16-chunked-64x128k16`
— the row above. So the failure was neither a recycling ceiling nor anything
this branch introduced, and the honest description is a transient: the machine
was concurrently fetching 292 MiB of model weights when it happened. It is
recorded here rather than explained away, because a renderer that dies once at
this length is worth knowing about even when it cannot be reproduced.

Two passes cost 1,334 s against 668 s for one, which is very nearly linear, so
recycling at this length is a time cost far more than a memory one.

## Command

```sh
export AFWEBGPU_QUALIFICATION_ASSET_ROOT=/path/to/assets   # needs model-multimer/
AFWEBGPU_CAPACITY=1 AFWEBGPU_COMPLEX_CHAINS=24 AFWEBGPU_COMPLEX_RECYCLES=0 \
  npx playwright test capacity --browser=chromium
```
