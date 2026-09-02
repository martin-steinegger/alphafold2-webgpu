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
half precision via `pack2x16float` (no device feature needed). The page exposes
them together as an explicit reduced-memory monomer option; exact f32 remains
the default and Multimer remains f32.

## Prerequisites

- macOS on Apple Silicon, Node 22 or newer, Chrome (stable) installed.
- The repository at the branch tip under review. Record `git rev-parse HEAD`
  with the results so validation stays attached to the exact shader revision.
- The private fixture captures under `test/fixtures/evoformer/` (they are
  excluded from Git; the model weights the benchmarks load come from
  `test/fixtures/evoformer/model1-query-59-stack/manifest.json`). Without
  them, steps 1, 5a and the q8 Chrome regression in 5b still run.

```bash
npm ci
npx playwright install chromium
```

## 1. Unit checks (no GPU)

```bash
npm run check
```

Expected: build, web build and the unit tests all pass. On the GB10 this is
"132 passed, 58 skipped". Any failure here is a portability bug, not a GPU one;
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

| args | admission estimate MiB | peakConcurrentMiB | peakResidentMiB | meanPlddt | GB10 ms/recycle |
|---|---:|---:|---:|---:|---:|
| 128 256 512 2 | 156 | 87 | 117 | 47.362 | 1314 |
| 384 256 512 2 | 386 | 272 | 351 | 41.418 | 7782 |
| 512 256 512 1 | 578 | 417 | 504 | 42.14 | 13895 |

`meanPlddt` should match to three decimals (synthetic alignment, deterministic
seed). `millisecondsPerRecycle` will differ; report it, and note the chip and
the memory of the Mac.

Then the inexact options, same shapes:

```bash
AFWEBGPU_TRIANGLE_F16=1 npm run bench:shape -- 384 256 512 2
AFWEBGPU_MSA_F16=1 npm run bench:shape -- 384 256 512 2
AFWEBGPU_TRIANGLE_F16=1 AFWEBGPU_MSA_F16=1 npm run bench:shape -- 384 256 512 2
```

GB10 values for 384: admission estimate 365 / 338 / 324 MiB, live
236 / 224 / 191 MiB, resident 331 / 307 / 307, mean pLDDT
41.419 / 41.382 / 41.372. Live and resident should again match exactly.
pLDDT may differ in the last decimals here: `pack2x16float` rounding is
implementation-defined, so Metal may round differently from Vulkan. A
difference beyond the second decimal is worth reporting.

## 4. Real alignments (optional but the best quality check)

`tools/predict-a3m.ts` runs the model on any A3M with explicit row caps. Start
with the tracked MMseqs2 acceptance alignment; this also proves that its
`#59\t1` header is ignored when sizing the WebGPU device:

```bash
npx tsx tools/predict-a3m.ts test.a3m 508 1024 4
AFWEBGPU_TRIANGLE_F16=1 AFWEBGPU_MSA_F16=1 npx tsx tools/predict-a3m.ts test.a3m 508 1024 4
```

Both lines must report `length: 59`, `depth: 8076`, mean pLDDT about 96.56 and
pTM about 0.761. The GB10 exact run reports 73 MiB live / 117 MiB resident; the
combined packed run 59 / 105 MiB. A low-confidence result around pLDDT 58-62
is the Metal correctness regression this check is meant to catch.

Then use another real A3M if available, for example:

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

### 5b. The exact q8 deep-MSA regression through Chrome/Metal

Keep the release model outside the repository:

```bash
QUAL_ROOT="$(mktemp -d)"
mkdir -p "$QUAL_ROOT/model"
gh release download model1-ptm-q8-v1 \
  --repo martin-steinegger/alphafold2-webgpu \
  --pattern afwebgpu-model1-ptm-q8-v1.tar.gz \
  --dir "$QUAL_ROOT"
tar -xzf "$QUAL_ROOT/afwebgpu-model1-ptm-q8-v1.tar.gz" -C "$QUAL_ROOT/model"
mkdir -p "$QUAL_ROOT/acceptance"
cp test.a3m "$QUAL_ROOT/acceptance/test.a3m"
AFWEBGPU_QUALIFICATION_ASSET_ROOT="$QUAL_ROOT" \
AFWEBGPU_BROWSER_CHANNEL=chrome \
AFWEBGPU_BROWSER_MONOMER_Q8_REGRESSION=1 \
npx playwright test test/browser/monomer-q8-metal-regression.spec.ts --workers=1
```

The automated test runs both exact and combined packed storage. Exact must
remain above pLDDT 90 / pTM 0.65, and packed must remain within 0.25 pLDDT and
0.005 pTM. Its console output prints both confidence pairs. For an interactive
run, start `AFWEBGPU_QUALIFICATION_ASSET_ROOT="$QUAL_ROOT" npm run dev` after
the test.

Open `http://127.0.0.1:4173/` in stable Chrome, then:

1. Choose Custom A3M and upload the tracked `test.a3m`.
2. Under Advanced settings set the monomer manifest to
   `/qualification-assets/model/manifest.json`.
3. Keep `Exact f32`, 508 clustered rows, 1024 extra rows and 3 recycles.
4. Predict and confirm all four passes stay above pLDDT 90 and pTM 0.65.
5. Record the four pLDDT/pTM values, runtime, allocator peak, Chrome version,
   macOS version, chip and memory.

Only after the exact run passes, select `Reduced-memory f16 (approximate)` and
repeat. Report its confidence deltas and memory/runtime changes. The page log
must explicitly say which activation storage was used.

### 5c. A longer prediction through the page

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

This work reduces aggregate residency but does not yet split the persistent
pair activation across buffers. It is therefore not an L1000 acceptance test:
at 1000 residues that one f32 tensor is about 488 MiB, above the 128 MiB
storage-binding limit reported by current Apple Chrome. On that adapter L512
is the contiguous-pair ceiling. L1000 needs a separate pair-sharding design.

## What to report

1. The three JSON lines from step 3 and whether live/resident matched exactly.
2. The f16 lines and their pLDDT deltas.
3. Any test failure with file, name and numbers; any worker crash and whether
   a rerun cleared it.
4. Browser: preflight text, the 59-residue pLDDT/pTM/time, the long-sequence
   allocator peak, and the Chrome GPU process peak from Activity Monitor.
5. The chip (M1/M2/M3/M4, Pro/Max) and its memory.
