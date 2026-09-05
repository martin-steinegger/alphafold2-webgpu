# Triangle attention on Apple Metal 3

Measured on an M4 Pro MacBook, Chromium, Apple Metal 3, on branch
`f16-projection`.

## The result

A 24-chain, 1,416-residue complex, one pass, whole prediction:

| | elapsed | vs `main` |
|---|---:|---:|
| `main` | 746 s | 1.00x |
| the projection work in `GEMM_CALIBRATION_APPLE_METAL_3.md` | 668 s | 1.12x |
| **plus the two attention changes below** | **407 s** | **1.83x** |

Nothing here is a cleverer kernel. One is a constant that was measured on
different hardware, and the other is the width of two operands.

## How the projections turned out to be the wrong place to look

The brief scoped this work to the dense projections. They are not where the
time is, and a profile says so immediately — `ExecutionContext` has timestamped
and labelled every dispatch all along, and `runInference` already accepts a
`profile` selector. Nothing was using either to ask this question.

One Evoformer block, by component, as the chain grows:

| component | 236 res | 472 res | 708 res |
|---|---:|---:|---:|
| **triangle attention (flash)** | 36.1% | 52.1% | **57.3%** |
| **triangle multiply (contract)** | 6.7% | 9.3% | **14.0%** |
| triangle attention projections | 17.4% | 12.0% | 8.8% |
| triangle multiply projections | 13.2% | 9.3% | 7.0% |
| outer product mean | 13.0% | 9.2% | 7.0% |
| transitions | 10.5% | 6.5% | 4.7% |
| MSA attention | 3.1% | 1.7% | 1.3% |
| block total | 143 ms | 824 ms | 2,517 ms |

The dense projections are 54% of a block at 236 residues and 27% at 708, and
falling; the two triangle terms are 43% and then 71%, and rising. At 1,416
residues the main Evoformer stack is 92% of the whole run — 689 seconds out of
746 — and the structure module and confidence heads together are about one
percent.

So a 1.2x on the projections is a 1.05x on the prediction. That was arithmetic,
not misfortune, and measuring first would have said so in ten minutes.

## The kernel is not compute-bound

Triangle attention runs at about 1.0 TFLOP/s where the projection kernel
reaches 2.8 on the same device. Half-precision *arithmetic* would therefore
have bought nothing: what it waits for is operands. Each invocation owns one
query and walks every key, reading a key and a value for each.

Three changes measured at a real shape, batch 64, 708 queries, 4 heads, head
dimension 32:

| variant | time | GFLOP/s | vs shipped |
|---|---:|---:|---:|
| shipped: 1 query per invocation, f32 keys and values | 15.7 ms | 1,046 | 1.00x |
| **1 query, f16 keys and values** | **12.2 ms** | **1,346** | **1.29x** |
| 2 queries per invocation, f32 | 34.9 ms | 471 | 0.45x |
| 4 queries per invocation, f32 | 170.5 ms | 96 | 0.09x |
| transposed pair bias | 16.1 ms | 1,020 | 0.98x |

## Half-precision keys and values: 1.29x

Packing the two operands read in the key loop as half words is worth 1.29x on
the kernel and 1.19x on a whole 59-residue prediction, against 1.046x for all
the projection work put together. The query and the gate stay single precision:
they are read once per invocation rather than once per key, so narrowing them
would round something for nothing.

`pack2x16float` is core WGSL. This needs no device feature and no browser loses
anything. Only the register kernel reads them packed; every other flash variant
keeps the single-precision pair it was written against.

It was gated on `variant === "register"` at first, which is wrong in a way
worth recording: above 128 queries the shape rule picks `register-2q`, and a
long chain is entirely above 128 queries, so the packing was switched off for
exactly the predictions it was built for. It showed 1.15x on a 59-residue
monomer and 1.3% on a 1,416-residue complex, and that gap is what exposed the
next thing.

## The queries-per-invocation threshold: the largest single change

`attentionFlashKernelForShape` gives an invocation two queries once a shape has
128 or more, so that one key and value load serves both. Its own comment
records where that came from:

> measured on GB10 at 1.17x-1.42x for 128 to 1024 queries, and 0.89x at 59

On this device it is the wrong way round. Two queries measure 2.2x slower than
one and four measure 10.9x slower, and every long prediction takes that path
because every long prediction is above 128 queries. Carrying one query took the
1,416-residue complex from 698 seconds to 400.

The fix is not to write 1 down instead of 2. The ratio is a property of a
driver's register budget, the same kind of fact as which flash kernel to use or
which arithmetic the projections want, and this codebase measures both of those
already. `src/runtime/attention-queries.ts` measures it too, once per device
and head dimension, and caches the winner; a probe that cannot run leaves the
shape rule alone. Left to itself the calibration chooses one query and
reproduces the forced run to within noise, 407 seconds against 400.

## What it costs the prediction

The gate the projections had to pass: two inputs, recycle by recycle, 0.05
pLDDT and 0.005 pTM.

| | f32 keys and values | f16 | shift |
|---|---:|---:|---:|
| query-only, recycle 0 | 56.7188 | 56.7160 | 0.0028 |
| query-only, recycle 1 | 59.9107 | 59.9084 | 0.0023 |
| query-only, recycle 2 | 61.3724 | 61.3712 | 0.0012 |
| query-only, recycle 3 | 63.2353 | 63.2274 | **0.0079** |
| deep MSA, 508 + 1024 rows | 96.8029 | 96.8029 | 0.0000 |

Worst pTM shift 0.00016. Per-residue pLDDT MAE 0.0095, PAE MAE 0.0051, atom37
RMS 0.0038 Å. The 1,416-residue complex returns 32.3419 mean pLDDT against
`main`'s 32.3469, and the project's own `monomer-q8-metal-regression`
acceptance test passes unchanged.

`test/browser/attention-differential.spec.ts` holds this, and asserts the two
sides differ as well as agree: a toggle that never reached the kernel would
report perfect agreement, which is what the first version of the packing did.

## Measured and rejected

- **Transposing the pair bias.** It is indexed `(head * queries + query) *
  queries + key`, so for one key index the lanes of a workgroup read addresses
  `queries * 4` bytes apart — sixty-four separate cache lines, none coalesced.
  Making them contiguous is worth nothing at all, which is a genuine surprise
  and worth not trying twice.
- **Two and four queries per invocation**, above.

## What is still on the table

The triangle multiply's contraction was the other term that grows with length,
6.7% of a block at 236 residues and 14.0% at 708, and it turned out not to be a
slow kernel at all: at the production channel count it reaches 2,161 GFLOP/s
against the projection kernel's 2,800. It was being fed eleven-row blocks of a
sixty-four-row tile. Filling the tile took the 1,416-residue complex from 407
seconds to 288 and is described in `LONG_COMPLEX_APPLE_METAL_3.md`.

Two things measured and rejected while looking at it: the contraction's second
operand is addressed `column * L + k`, so its staging never coalesces, and
making it row-major is worth only 1.06x; and packing the attention pair bias as
half words, which is 1.20x against 1.30x for packing the keys and values alone —
it costs more in unpacking than it saves in traffic. The transposed-bias result
had already said that operand was not the constraint, and testing the same
operand a second way was the wrong call.

The first three attempts at the microbenchmark measured nothing: the uniform
buffer was sixteen bytes short of the `Parameters` struct, so every bind group
was invalid and every dispatch was skipped, which looks exactly like a very
fast kernel. The check that counts how many outputs came back non-zero is what
caught it, and it stays in the spec.

## Commands

```sh
export AFWEBGPU_QUALIFICATION_ASSET_ROOT=/path/to/assets
AFWEBGPU_BLOCK_PROFILE=1 AFWEBGPU_COMPLEX_CHAINS=12 npx playwright test block-profile --browser=chromium
AFWEBGPU_ATTENTION_BENCH=1 npx playwright test attention-microbenchmark --browser=chromium
AFWEBGPU_ATTENTION_DIFFERENTIAL=1 npx playwright test attention-differential --browser=chromium
AFWEBGPU_CAPACITY=1 AFWEBGPU_COMPLEX_CHAINS=24 npx playwright test capacity --browser=chromium
```
