# Verifying the memory work on Apple Silicon

This is a self-contained brief for an agent or a person with a Mac. It checks
that the inference memory reductions and the optional half-precision storage
committed on 2 September 2026 behave on Apple Silicon the way they do on the
Linux GB10 they were developed on. Nothing here changes the repository; it
only runs and reports.

## What changed, in one paragraph

Peak GPU memory of monomer inference was cut to the floor of the exact design:
the pair, the MSA, one whole triangle projection and three 8 MiB blocks. The
embedder writes the pair in place; the clustered MSA is embedded after the
extra stack; the template update is uploaded per recycle; every operation's
scratch is an 8 or 16 MiB chunk backed by whole mebibytes so the pool can hand
one operation's chunks to the next; the triangle multiplication runs on the
shared GEMM with its LayerNorms fused into operand loads, the incoming
direction blocked by output columns. Two inexact options exist, off by
default: `triangleWholeStorage: "f16"` and `msaStorage: "f16"`, both packed
half precision via `pack2x16float` (no device feature needed).

## Prerequisites

- macOS on Apple Silicon, Node 22 or newer, Chrome (stable) installed.
- The repository at the branch tip that contains commit `e9b3cb4` ("Offer
  packed half-precision storage for the MSA activations").
- The private fixture captures under `test/fixtures/evoformer/` (they are
  excluded from Git; the model weights the benchmarks load come from
  `test/fixtures/evoformer/model1-query-59-stack/manifest.json`). Without
  them, only steps 1 and 5a run.

```bash
npm ci
npx playwright install chromium
```

## 1. Unit checks (no GPU)

```bash
npm run check
```

Expected: build, web build and the unit tests all pass. On the GB10 this is
"127 passed, 58 skipped". Any failure here is a portability bug, not a GPU one;
report the failing test name and message.

## 2. GPU differential suite through dawn-node

```bash
npm run test:gpu
```

Expected: 28 files pass, 3 skipped (they need `shader-f16`, which Apple
exposes, so those may run and pass here instead of skipping). One file,
`test/global-attention-large-grid.gpu.test.ts`, segfaults on the Linux machine
on the plain `requestDevice()` path for reasons unrelated to this work; if it
crashes here too, exclude it and say so:

```bash
AFWEBGPU_GPU_TESTS=1 npx vitest run $(ls test/*.gpu.test.ts | grep -v global-attention-large-grid)
```

dawn-node occasionally kills a worker ("Worker exited unexpectedly") without a
failing test. That is known flakiness; rerun once. A test that fails with an
assertion is a real finding: report the file, the test name, and the numbers.

Tests that matter most for this work:

- `test/triangle-multiplication-outgoing.gpu.test.ts` and
  `test/evoformer-triangle-incoming.gpu.test.ts`: whole, 16-row and 2-row
  blocks in both directions against the reference, and the f16 option's error.
- `test/evoformer-block.gpu.test.ts`: the full block against the reference,
  forced tiny scratch windows against the default run, and the f16 MSA
  storage error.
- `test/input-embedder.gpu.test.ts`: the in-place embedder bit-exact against
  the separate-buffer path.
- `test/allocator.test.ts`, `test/storage.test.ts`, `test/device.test.ts`:
  pool rules, packing, and the memory estimate pinned to measurements.

## 3. Memory and speed at realistic lengths

```bash
npm run bench:shape -- 128 256 512 2
npm run bench:shape -- 384 256 512 2
npm run bench:shape -- 512 256 512 1
```

Each prints one JSON line. Compare `peakConcurrentMiB` (live working set) and
`peakResidentMiB` (live plus pooled) with the GB10 values. The allocation
pattern is deterministic and reads no device property, so these two numbers
should match **exactly**:

| args | peakConcurrentMiB | peakResidentMiB | meanPlddt | GB10 ms/recycle |
|---|---|---|---|---|
| 128 256 512 2 | 87 | 117 | 47.362 | 1314 |
| 384 256 512 2 | 272 | 351 | 41.418 | 7782 |
| 512 256 512 1 | 417 | 504 | 42.14 | 13895 |

`meanPlddt` should match to three decimals (synthetic alignment, deterministic
seed). `millisecondsPerRecycle` will differ; report it, and note the chip and
the memory of the Mac.

Then the inexact options, same shapes:

```bash
AFWEBGPU_TRIANGLE_F16=1 npm run bench:shape -- 384 256 512 2
AFWEBGPU_MSA_F16=1 npm run bench:shape -- 384 256 512 2
AFWEBGPU_TRIANGLE_F16=1 AFWEBGPU_MSA_F16=1 npm run bench:shape -- 384 256 512 2
```

GB10 values for 384: live 236 / 224 / 191 MiB, resident 331 / 307 / 307, mean
pLDDT 41.419 / 41.382 / 41.372. Live and resident should again match exactly.
pLDDT may differ in the last decimals here: `pack2x16float` rounding is
implementation-defined, so Metal may round differently from Vulkan. A
difference beyond the second decimal is worth reporting.

## 4. Real alignments (optional but the best quality check)

`tools/predict-a3m.ts` runs the model on any A3M with explicit row caps. Use an
A3M you already have (a ColabFold A3M works), or generate one through the page
in MMseqs2 mode and download it. Then, for example:

```bash
npx tsx tools/predict-a3m.ts my.a3m 256 512 4
AFWEBGPU_TRIANGLE_F16=1 AFWEBGPU_MSA_F16=1 npx tsx tools/predict-a3m.ts my.a3m 256 512 4
npx tsx tools/predict-a3m.ts my.a3m 128 256 4
```

Compare `meanPlddt`, `ptm` and `recyclePlddt` between the exact run and the
f16 run. On four alignments of 164–396 residues the GB10 saw identical values
to two decimals. Report any protein where they differ by more than 0.1.

## 5. The browser

### 5a. Preflight and UI specs (no model needed)

```bash
AFWEBGPU_BROWSER_CHANNEL=chrome npm run test:browser -- test/browser/webgpu.spec.ts
```

Expected: the preflight reports "WebGPU ready" with the Apple GPU. Without
`AFWEBGPU_BROWSER_CHANNEL=chrome` Playwright's bundled Chromium is used, which
also works on macOS.

### 5b. A real prediction through the page

```bash
npm run build:web
npm run export:web-model -- test/fixtures/evoformer/model1-query-59-stack/manifest.json dist/web/model
npx vite preview --config vite.browser.config.ts --port 4173 --host 127.0.0.1
```

Open `http://127.0.0.1:4173/`, set Alignment to "Single sequence", keep the
default 59-residue sequence and 3 recycles, and predict. Expected on the GB10:
about 1.4 s of inference, mean pLDDT 63.5, pTM 0.421, and a run log line
"Measured allocator peak: 46 MiB". The pLDDT and pTM should match; the time
will not. Then paste a 300–400 residue sequence and predict again; the run log
should report an allocator peak near 237 MiB for 384 residues in single-sequence
mode. Watch the Chrome GPU process in Activity Monitor during that run and note
its peak memory: it will sit above the allocator's number by whatever Dawn's
Metal backend and its staging buffers hold, and that gap is the one number
this work could not measure on Linux.

The page still applies the Apple safety budget of 70% of
`navigator.deviceMemory` against the memory estimate. The estimate is lower
now, so inputs that were refused this morning may be accepted; if one is
accepted and then fails with an out-of-memory error, report the sequence
length, the row settings and the machine's memory.

## What to report

1. The three JSON lines from step 3 and whether live/resident matched exactly.
2. The f16 lines and their pLDDT deltas.
3. Any test failure with file, name and numbers; any worker crash and whether
   a rerun cleared it.
4. Browser: preflight text, the 59-residue pLDDT/pTM/time, the long-sequence
   allocator peak, and the Chrome GPU process peak from Activity Monitor.
5. The chip (M1/M2/M3/M4, Pro/Max) and its memory.
