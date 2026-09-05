# Brief: verify `f16-projection` on the GB10

You are on the Linux workstation with the NVIDIA GB10 and the reference
fixtures. Branch `f16-projection` was measured entirely on an M4 Pro MacBook,
which has neither, and it changes behaviour on your machine in ways nobody has
run there.

```sh
git fetch origin && git checkout f16-projection
```

## The one thing only you can do

That MacBook has no `test/fixtures/evoformer/model1-*-stack` or
`model1-query-59-block0`, so 32 GPU tests skip-by-failure there — every test
that compares a kernel against official AlphaFold intermediates. They were
never run against these changes.

```sh
AFWEBGPU_GPU_TESTS=1 npx vitest run
```

On `main` that suite should be green on your machine. If it is green on this
branch too, the kernels are right against the official references, which is a
stronger statement than anything measured on the Mac. **This is the highest
value check in this document.** If it is red, stop and report which test;
everything below is secondary.

### What that suite does and does not reach

Worth knowing before leaning on a green run. Only five of the thirty-nine GPU
test files build their device with `requestAlphaFoldDevice`, which is what runs
the projection calibration and installs the variant: `a3m-model`,
`a3m-raw-input`, `binding-limit-model`, `evoformer-attention` and
`evoformer-stack-a3m`. The other thirty-four request a device directly, so the
module default stands and they exercise the f32 projection kernel rather than
whichever one your device selects.

The attention work is not affected by this — the width of the keys and the
query count are decided inside `encodeAttention` from whatever device is
handed to it, so every test exercises them. That is why packing the values
showed up where it did.

So a green suite means: the attention changes are right against the
references everywhere, and the selected projection kernel is right in the five
tests that run the whole model or the whole stack. It does not mean the
f16-chunked or matrix projection has been compared against a reference at the
level of a single kernel. If that is wanted, `forceGemmVariant` at the top of
those thirty-four tests would get it, and only a machine with the fixtures can
run it.

## What changed for you specifically

Three things, none of which was measured on your hardware.

**1. How many queries an attention invocation carries.**
`attentionFlashKernelForShape` takes two above 128 queries, from a threshold
whose comment records "measured on GB10 at 1.17x-1.42x for 128 to 1024
queries, and 0.89x at 59". On the M4 Pro two is 2.2x *slower* than one and four
is 10.9x slower, so it is now measured per device in
`src/runtime/attention-queries.ts` rather than written down.

Your machine is the one that constant was right for. The probe should choose
two and your timings should not move.

```sh
AFWEBGPU_ATTENTION_PROBE=1 npx playwright test attention-shape-probe --browser=chromium
```

Expect `2q f32` or `2q f16`. If it reports `1q`, the probe shape — 512 queries,
4 batches, one head — does not represent your hardware, and it is choosing
badly rather than choosing. Say so; the probe shape is the thing to change.

Note the shape rule still applies below its threshold: a shape with fewer than
128 queries keeps one query per invocation whatever the probe found, because
the probe has nothing to say about that case and your own note records 0.89x
there.

**2. Attention keys and values as packed half words.**
Worth 1.29x on the Mac, where that kernel waits on memory. Yours may not: it
costs an unpack per load and saves half the traffic. The probe above measures
`f32` against `f16` and should pick whichever wins on your device. `pack2x16float`
is core WGSL, so nothing here needs a feature you lack.

**3. `ATTENTION_PROJECT_SHADER` is now built on demand.**
It was a module-scope constant, evaluated before any device exists, so the
projection calibration never reached it — on both machines it has been running
f32 regardless of what was selected. It is now a function, which means your
largest projection will start using whatever your device's probe picks. That is
the intended fix, but it is a change to your behaviour and worth watching.

## The gates, in order

All need a model bundle. Point `AFWEBGPU_QUALIFICATION_ASSET_ROOT` at a
directory holding `model/`, `model-multimer/` and `acceptance/test.a3m`; the
deployed browser bundles work
(`https://martin-steinegger.github.io/alphafold2-webgpu/model/manifest.json`).

```sh
AFWEBGPU_GPU_TESTS=1 npx vitest run                      # the fixtures, above
npx vitest run                                           # 191 CPU tests
AFWEBGPU_GEMM_DIFFERENTIAL=1  npx playwright test gemm-differential --browser=chromium
AFWEBGPU_ATTENTION_DIFFERENTIAL=1 npx playwright test attention-differential --browser=chromium
AFWEBGPU_BROWSER_MONOMER_Q8_REGRESSION=1 npx playwright test monomer-q8-metal-regression --browser=chromium
```

Both differentials predict one input with and without a change and hold the
result to 0.05 mean pLDDT and 0.005 pTM at every recycle. They also assert the
two sides *differ*, which is what catches a toggle that never reached the
kernel — that has happened twice on this branch.

## Does it still go faster, or at all?

```sh
AFWEBGPU_CAPACITY=1 AFWEBGPU_COMPLEX_CHAINS=24 \
  npx playwright test capacity --browser=chromium
```

1,416 residues, one pass. On `main` the Mac takes 746 s and 775 s; on this
branch 288 s. Run it on `main` and on the branch and compare — the absolute
numbers mean nothing on your hardware, only the ratio does. **A ratio below
1.0 is a regression and the thing worth reporting.**

The other change likely to move your numbers is
`triangleBlockRows`: the contraction dispatches whole 64-row GEMM tiles and
masks what the block does not cover, so the budget now holds at least one
whole tile. On the Mac that took 407 s to 288 and cost 47 MiB of peak. It
should help you too, but it does raise peak memory — check
`Estimated peak GPU allocations` in the run log against what your device
allows, especially at lengths past 1,416 where a block is 44 MiB and grows
linearly.

## Then merge it

If the fixture suite is green and no gate fails, merge to `main`. As of this
writing `main` has not moved since the branch was cut, so it is a
fast-forward.

Merging is the point of the exercise rather than an afterthought, and the
order matters. The branch changes 232 lines of `attention.ts` and `block.ts`,
which is exactly where the binding-layout refactor — the hand-computed shader
binding indices and the duplicated slot guards — is going to land. Whichever of
the two goes second pays for the rebase, and this one is written and verified
while that one is not.

You have the authority not to merge. The fixture suite is the gate that was
never run, and a failure there outranks every number in these documents.

## What Metal says after your fix

Re-measured on the M4 Pro at your commit, 24 chains and 1,416 residues, one
pass: **358 s against `main`'s 746 and 775, so 2.1x**, down from 288 s while
the values were packed too. Every gate green — both differentials, the q8
acceptance test, 191 CPU tests, and the same 32 fixture failures this machine
has always had for want of your fixtures. Estimated peak 2,487 MiB against a
5,734 MiB Apple budget, so 3,247 MiB of headroom here; the narrower margin you
saw is your configuration's, not Metal's.

**One regression in the fix, now repaired.** The probe reads
`process.env.AFWEBGPU_PROBE_DEBUG`, and there is no `process` in a browser: it
threw, the `catch` turned that into "no measurement", and every browser device
silently reverted to the shape rule and single-precision operands. It failed
only in the browser, which is the only place the model runs, and passed in the
test process, which is where it was checked. The access is guarded now and the
catch warns instead of swallowing.

Two things follow for your numbers. Your 75.06 → 70.13 s was measured through
`capacity.spec.ts`, which runs in a browser, so it was probably taken with the
probe dead and none of the attention work active — worth re-running. And your
observation that the probe picks one query per invocation there may be an
artifact of the same path rather than a real preference; it is worth
re-checking before concluding the probe shape is unrepresentative. On Metal it
now picks `1q f16-key` in 200 ms, which is correct for this device.

Noted on the device cost you raised: 480 ms here against your 830. It is the
GEMM calibration measuring six arrangements across two shapes, once per
device — 0.1% of a long prediction and 11% of a short one. One waste is gone
(each candidate was compiling its shader twice); the rest is what measuring
instead of assuming costs, and it is now stated rather than hidden.

## If something regresses

Every choice is measured and cached per device, and each has an override used
only by tests: `forceGemmVariant`, `forceAttentionKeyValueStorage`,
`forceAttentionQueriesPerThread`. Pinning one isolates which change is
responsible without unpicking the branch. `capacity.spec.ts` takes
`AFWEBGPU_ATTENTION_QUERIES=1|2` directly.

Report the ratio, which gate failed if any, and what the shape probe chose.
The three documents `ATTENTION_APPLE_METAL_3.md`,
`LONG_COMPLEX_APPLE_METAL_3.md` and `GEMM_CALIBRATION_APPLE_METAL_3.md` have
the Mac numbers to compare against, including what was measured and rejected.
