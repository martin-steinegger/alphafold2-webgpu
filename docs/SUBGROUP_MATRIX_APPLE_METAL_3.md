# The matrix units are worth 1.7x, exactly, and need one thing from the callers

Measured on an M4 Pro MacBook, Chromium, Apple Metal 3, branch
`f16-projection`, via `AFWEBGPU_GEMM_CALIBRATION=1 npx playwright test
gemm-calibration`.

## The result

The half-precision work in `GEMM_CALIBRATION_APPLE_METAL_3.md` ships
`f16-chunked` at 1.11x to 1.25x, having rejected pure f16 as unsafe. A
subgroup-matrix kernel with proper reuse beats all of it at every shape, and it
is not approximate:

Median of three runs, with the spread across them, because a single pass is
not trustworthy on this machine — see the stability section of
`GEMM_CALIBRATION_APPLE_METAL_3.md`.

| shape | production | **matrix f32** | spread | f16-chunked (shipped) |
|---|---:|---:|---:|---:|
| opm-contract | 2.79 TFLOP/s | **1.69x** | 1.64-1.69x | 1.26x |
| opm-out | 2.04 | **2.15x** | 2.02-2.15x | 1.13x |
| opm-out2 | 2.23 | **1.76x** | 1.74-1.76x | 1.21x |
| project | 2.85 | **1.67x** | 1.56-1.68x | 1.16x |
| output | 2.77 | **1.69x** | 1.61-1.70x | 1.15x |
| trans1 | 2.83 | **1.67x** | 1.56-1.68x | 1.15x |
| trans2 | 2.95 | **1.71x** | 1.58-1.73x | 1.18x |
| extra1 | 2.32 | **1.46x** | 1.43-1.55x | 1.12x |

So 1.46x to 2.15x, against 1.12x to 1.26x for the kernel this branch ships and
1.27x to 1.50x for the pure f16 kernel it rejected. It reaches 4.7 to 5.1
TFLOP/s against production's 2.8, and its worst error is
2.62e-6 at K=256 and 1.00e-5 at K=1024 — the same as production, to the digit,
because it accumulates in f32. There is no accuracy gate to pass: it computes
what the f32 kernel computes.

So the honest answer to "should the projections compute in half precision" is,
on this device, **no** — they should use the matrix units in single precision,
where the exact kernel is faster than the approximate one. Half precision
remains the right answer for any device with `shader-f16` and no matrix units,
which is what ships today.

An f16 matrix variant measured a further 6% (2.29x at opm-out) and is not
pursued: it reintroduces the 0.31% to 0.87% error that the exact kernel makes
unnecessary.

## Why the earlier prototype said 0.72x

The brief's judgement was right. `matrix-f32-8x8x8` gives one subgroup one 8x8
output tile and walks all of K from storage: one multiply-accumulate per two
tile loads, memory-bound by construction. It measures 0.60x to 0.72x and says
nothing about the hardware.

`matrix-bounded-f32` gives one subgroup a 32x32 region. Four left tiles and
four right tiles feed sixteen multiply-accumulates — 2.0 per load rather than
0.5. That single change, register reuse with no workgroup staging at all, is
the whole difference between 0.72x and 1.7x.

## Two things that were not obvious

**`subgroupMatrixStore` requires a uniform offset.** WGSL's uniformity
analysis works at workgroup scope, so an offset derived from *which subgroup
you are* cannot be proven uniform even though it is. A workgroup of eight
subgroups each storing its own region does not compile:

```
error: 'subgroupMatrixStore' requires argument 1 to be uniform
```

The kernel therefore runs one subgroup per workgroup and derives every offset
from the workgroup id. That costs nothing, because the reuse is in registers.

**The ragged edge must not be a scalar fallback.** The model's row counts are
products of a sequence count and a residue count — 29,972, 60,416, 3,000 —
so a kernel needing M and N divisible by 32 cannot serve them. A scalar path
for partial regions is correct and catastrophic: its inner loop reads the
weights with a stride of a whole row, uncoalesced, and took opm-out2 to
**0.09x**, with `project` and `trans2` falling to 1.33x and 0.73x.

Running the matrix path everywhere and checking bounds *in the store* is both
correct and free. Loads past the end of a tensor are clamped by WGSL's
robustness rules, so a partial region computes garbage exactly in the rows and
columns that do not exist, and those are the ones never written. The bounded
kernel measures within noise of the unbounded one on aligned shapes and keeps
1.46x to 2.15x on the real ones.

## What it needs from the callers, and why it stopped here

`createTiledGemmShader` takes its operands as WGSL *expressions*:

```ts
sourceElement: "source[row * parameters.inner + k]",
weightElement: "weights[parameters.weight_offset + k * parameters.columns + column]",
```

which is what lets a caller read an activation stored as packed f16 through
`unpack2x16float`, or window a tensor past a binding limit. `subgroupMatrixLoad`
cannot consume an expression. It needs a typed array binding, a base offset and
a stride:

```wgsl
subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(
  &source, (row_origin + 8u) * parameters.inner + k0, false, parameters.inner)
```

So the matrix path cannot be made invisible to the call sites the way the
half-precision path was. It needs each caller that wants it to declare that its
operand *is* a plain array with a known stride — an additional, optional field
on `TiledGemmShader`, which only opted-in callers would fill in. That is a
change to `attention.ts`, `block.ts`, `outer-product-mean.ts` and
`triangle/shaders.ts`, which belong to the other branch, so this reports the
measurement rather than making the change.

Two further constraints for whoever picks it up:

- **`chromium-experimental-subgroup-matrix` is not standards-compliant WGSL**,
  and `AGENTS.md` asks for standards compliance. It has to be a
  capability-gated fast path, which is what `src/runtime/gemm-selection.ts`
  already is: add the variant, let the probe measure it, and every device
  without the extension keeps the kernel it has. The prediction gate in
  `gemm-differential.spec.ts` already runs whatever the selection can choose.
- **The `epilogue` hook cannot be served as it stands.** Its fragments are
  written against `acc{n}: vec4<f32>` in a 64x128 tile with eight rows and four
  columns per invocation; a matrix kernel has a different thread-to-output
  mapping. `triangle/shaders.ts` is the only caller using it. The matrix kernel
  here stages its result through workgroup memory and then runs a
  per-invocation epilogue over it, which is the same shape as the existing
  `gemm_stage` contract, so a common form is probably reachable — but it is a
  design question for the owner of that file, not a mechanical change.

## The grid is a second obstacle, and it costs about a fifth of the win

`matrix-bounded-f32` covers a 32x32 region per workgroup, which is a different
dispatch grid from the hand-tiled kernel. `gemmGrid` derives the grid from the
output tile and does not know which shader is asking, so if some callers opt
into a matrix path and others do not, one of them gets the wrong grid and
silently leaves output unwritten.

Keeping the shipped 64x128 tile fixes that — one subgroup walks eight 32x32
sub-regions instead of eight times as many workgroups each doing one — and
`matrix-wide-f32` measures what it costs:

| shape | 64x128 tile | 32x32 tile |
|---|---:|---:|
| opm-contract | 1.28x | 1.68x |
| opm-out | **0.50x** | 2.16x |
| opm-out2 | **0.40x** | 1.75x |
| project | 1.66x | 1.67x |
| output | 1.56x | 1.71x |
| trans1 | 1.57x | 1.67x |
| trans2 | 1.45x | 1.70x |
| extra1 | 1.34x | 1.39x |

It holds on the wide shapes and collapses on the narrow ones. `opm-out` and
`opm-out2` have N=128, so a 128-wide tile makes the grid one workgroup across:
64 workgroups of 32 lanes, which does not fill the machine. The 32x32 tile gets
four times the column blocks.

Weighted by the time each shape takes in a block, that is about **1.39x** for
the grid-compatible geometry against **1.69x** for the 32x32 one, and 1.17x for
the half-precision kernel that ships today.

Both still need the operand declaration below. The grid only decides how much
the change is worth, not whether it is needed.

## What is in the tree

`tools/gemm-candidates.ts` carries `matrix-tiled-f32` (requires M and N
divisible by 32), `matrix-bounded-f32` (any shape, 32x32 tile) and
`matrix-wide-f32` (any shape, the shipped 64x128 tile), all verified against
the harness's own reference at both depths, both applying bias and the
activation so they are checked rather than throughput-only. Four 32-aligned
shapes were added alongside the real ones so the aligned and bounded kernels
can be compared directly.

Nothing in `src/` uses them. The shipped kernel is unchanged.

## Command

```sh
AFWEBGPU_GEMM_CALIBRATION=1 npx playwright test gemm-calibration --browser=chromium
```
