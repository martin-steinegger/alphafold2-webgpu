# Verifying the memory work on Apple Silicon

This is a self-contained brief for an agent or a person with a Mac. It checks
that the inference memory reductions and the optional half-precision storage
committed on 2 September 2026 behave on Apple Silicon the way they do on the
Linux GB10 they were developed on. Nothing here changes the repository; it
only runs and reports.

## What changed, in one paragraph

There is one model now. Activations are stored as packed half precision
everywhere, monomer and Multimer alike: the MSA, the pair and the triangle
multiplication's whole projection, packed two to a word with `pack2x16float`,
which needs no device feature. The page has no storage control and no exact
mode. The exact f32 storages remain reachable in code as `EXACT_STORAGE`, and
in the tools as `AFWEBGPU_EXACT=1`, because the differential tests compare the
kernels against AlphaFold's own f32 tensors; nothing a user runs takes that
path.

That decision rests on measurement: on real alignments of 164 to 396 residues
the packed model returned mean pLDDT and pTM identical to the exact path to two
decimals, per recycle as well as at the end, and the pair's magnitudes reach
about 1050 against half precision's 65504, so what it costs is rounding rather
than range. What it buys is roughly half the memory, which is what decides
whether a chain runs in a browser at all.

Around that, the allocator now records what was live when the peak was reached,
and that list drove the rest of the work. It found two tensors that were
nothing but written-out LayerNorms, a copy of the pair in the structure module
and a copy of the extra MSA in the global column attention, 128 MiB each at 512
residues; both now normalize while loading from row statistics. Several
pair-sized host copies went too: the recycle buffers are cleared on the device
instead of being uploaded as zeros, the final pair is read back only when a
caller asks, and the clustered MSA features carry 27 channels instead of 49.
Finally the query-only template module no longer runs during a prediction, its
update being one constant vector folded into the pair projection's bias.

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
| 128 256 512 2 | 126 | 67 | 97 | 47.193 | 1314 |
| 384 256 512 2 | 246 | 152 | 231 | 41.426 | 7782 |
| 512 256 512 1 | 328 | 225 | 312 | 42.094 | 13895 |

Live and resident should match exactly. `meanPlddt` may differ in the last
decimals: `pack2x16float` rounding is implementation-defined, so Metal may
round differently from Vulkan. A difference beyond the second decimal is worth
reporting. `millisecondsPerRecycle` will differ; report it, and note the chip
and the memory of the Mac. The timings above predate the packed model and are
only a rough scale.

The exact storages, which no prediction uses, are one environment variable
away and are the reference for how much the packing saves:

```bash
AFWEBGPU_EXACT=1 npm run bench:shape -- 384 256 512 2
```

GB10: admission estimate 386 MiB, live 272, resident 351, mean pLDDT 41.418,
against 246 / 152 / 231 / 41.426 for the model as it ships.

## 4. Real alignments (optional but the best quality check)

`tools/predict-a3m.ts` runs the model on any A3M with explicit row caps. Start
with the tracked MMseqs2 acceptance alignment; this also proves that its
`#59\t1` header is ignored when sizing the WebGPU device:

```bash
npx tsx tools/predict-a3m.ts test.a3m 508 1024 4
```

It must report `length: 59`, `depth: 8076`, mean pLDDT about 96.56 and pTM
about 0.761, at 58 MiB live and 102 MiB resident on the GB10. A low-confidence
result around pLDDT 58-62 is the Metal correctness regression this check is
meant to catch.

Then use another real A3M if available, for example:

```bash
npx tsx tools/predict-a3m.ts my.a3m 256 512 4
AFWEBGPU_EXACT=1 npx tsx tools/predict-a3m.ts my.a3m 256 512 4
npx tsx tools/predict-a3m.ts my.a3m 128 256 4
```

Compare `meanPlddt`, `ptm` and `recyclePlddt` between the model and the exact
path. On four alignments of 164–396 residues the GB10 saw identical values to
two decimals. Report any protein where they differ by more than 0.1.

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
gh release download model1-ptm-q8-v1 \
  --repo martin-steinegger/alphafold2-webgpu \
  --pattern afwebgpu-model1-ptm-q8-v1.tar.gz \
  --dir "$QUAL_ROOT"
# The archive already contains its top-level model/ directory.
tar -xzf "$QUAL_ROOT/afwebgpu-model1-ptm-q8-v1.tar.gz" -C "$QUAL_ROOT"
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

Known pass for commit `9f0252a` on an Apple M4 Pro (20-core GPU), macOS
15.4.1 and Chrome 152.0.7977.65: the exact result was pLDDT 96.802248 / pTM
0.754544, the packed result was 96.802616 / 0.754548, and the test took 14.1
seconds. Chrome reported the non-fallback `apple metal-3` adapter with 4096
MiB maximum buffer and storage-binding limits.

Open `http://127.0.0.1:4173/` in stable Chrome, then:

1. Choose Custom A3M and upload the tracked `test.a3m`.
2. Under Advanced settings set the monomer manifest to
   `/qualification-assets/model/manifest.json`.
3. Keep 508 clustered rows, 1024 extra rows and 3 recycles.
4. Predict and confirm all four passes stay above pLDDT 90 and pTM 0.65.
5. Record the four pLDDT/pTM values, runtime, allocator peak, Chrome version,
   macOS version, chip and memory.

There is no storage choice to make: the page runs the packed model, and its
log says so. If the confidence here falls short of the Node run in step 4,
that is a Metal difference worth reporting, not a setting.

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

## 6. Multimer on a notebook

A ten-copy 59-mer (`PIAQ…ELASK` entered as ten chains separated by colons) at
the page's multimer defaults froze a notebook in Chrome and errored in
Firefox. Two things changed for it: the page's Apple budget is now 35% of the
memory the browser reports (2.8 GiB at the 8 GB Chromium cap) and applies to a
multimer-aware estimate, and every GPU allocation is checked against that
budget so an overrun fails with a `GpuMemoryBudgetError` message naming the
tensor instead of paging the machine. The multimer path also embeds the
clustered MSA only after the extra stack now, like the monomer.

Run the same ten-copy input again in Chrome. Expected: either the gate refuses
it before the device is created with a row suggestion, or the prediction runs
and the run log shows an allocator peak below the budget. Report which, the
budget line from the run log, and the machine's memory. The multimer reference
tests (`AFWEBGPU_MULTIMER_F32_MANIFEST` and `AFWEBGPU_MULTIMER_COMPRESSED_MANIFEST`
pointing at your multimer model manifests) must also pass:

```bash
AFWEBGPU_GPU_TESTS=1 AFWEBGPU_MULTIMER_F32_MANIFEST=<path> AFWEBGPU_MULTIMER_COMPRESSED_MANIFEST=<path> \
  npx vitest run test/multimer-model-official.gpu.test.ts
```

They could not run on the Linux machine, which has no multimer weights, so the
deferred MSA embedding for multimer is verified only by review until they do.

## 7. Worker, host memory and Stop

The page now runs the pipeline in a dedicated worker when the browser exposes
WebGPU there (Chrome does), and falls back to the main thread otherwise; the
run log's first lines say which. During a long prediction the page should
keep painting progress and the Stop button should end the run within a second
or two, discarding the worker and its GPU device (the next run reloads the
model). Also watch the renderer process in Activity Monitor: the Evoformer
weights are now decoded from the int8 shards on demand, so a q8 model should
hold about 100 MB on the host instead of 355, and for chains of about 256
residues and more the monomer template update is recomputed on the GPU each
recycle instead of being kept on the host (shorter chains keep the one-time
computation, where the copy is small and the stack a sizeable share of a
recycle).

## 7b. The packed model against the exact path

Every prediction runs packed. To see what that costs, run the same real
alignment both ways, at a length where the pair matters:

```bash
npx tsx tools/predict-a3m.ts alignment.a3m 508 1024 3
AFWEBGPU_EXACT=1 npx tsx tools/predict-a3m.ts alignment.a3m 508 1024 3
```

On the Linux GB10 the two agreed on mean pLDDT and pTM to two decimals on four
proteins of 164 to 396 residues, while the peak fell by 37 to 43 per cent: a
260-residue protein from 226 to 143 MiB and a 396-residue one from 382 to 216.
A complex behaves the same way where its confidence is real: on the acceptance
homodimer with a live ColabFold alignment, exact gave pLDDT 91.56, pTM 0.842
and ipTM 0.8359, packed 91.58, 0.843 and 0.8364, for 99 against 79 MiB live.
Query-only complexes, whose structures never converge, move further; that is
the input, not the storage.

Report both numbers per protein: if Metal rounds `pack2x16float` differently,
it shows as a pLDDT difference here first. Packing also halves the largest
storage binding, which is what an adapter limits: this Chrome reports 128 MiB,
which the exact pair reaches at 512 residues and the packed one at about 724,
so please report the limit Chrome gives you and the longest chain that runs.

`AFWEBGPU_MEMORY=1` on `tools/benchmark-shape.ts` prints what was live at the
peak, largest first. Send that list for one long shape; it is the fastest way
to see whether Metal holds anything the Linux run does not.

## 8. Reading a complex

Predict a homodimer (paste the default 59-residue chain twice, separated by a
colon). The complex views should all agree on the same two chains:

- the structure opens coloured by chain, with a key under it giving each
  chain's length and mean pLDDT, and the colour control switches to pLDDT or
  rainbow without reloading the model;
- the confidence, alignment-error and coverage plots carry a black rule at
  each chain boundary and a coloured chain letter;
- the "Chains" card lists per-chain pLDDT and the mean alignment error of
  every chain pair, whose off-diagonal figure is the interface;
- clicking an off-diagonal block of the alignment-error matrix fades every
  other chain in the structure and says which pair is shown; clicking the same
  block again brings the whole complex back.

The browser specs for this are opt-in because they need a served model:

```bash
AFWEBGPU_QUALIFICATION_ASSET_ROOT=$HOME/models \
AFWEBGPU_BROWSER_MULTIMER_MANIFEST=/qualification-assets/model-multimer/manifest.json \
AFWEBGPU_BROWSER_MONOMER_MANIFEST=/qualification-assets/model-q8/manifest.json \
  npx playwright test multimer-visualization
```

## 9. The result archive

After any prediction finishes, the results heading carries a "Download results
ZIP" button. It should produce `<job>.result.zip` within a second or two, and
unzipping it should leave one folder named after the job holding the structure,
the scores, the alignment error in AlphaFold-DB's format, three PNG plots, the
alignment, the settings, the run log and the citations:

```bash
unzip -l ~/Downloads/test.result.zip
unzip -p ~/Downloads/test.result.zip 'test/config.json'
```

Safari writes the archive with its own compression path, so check there too
that `unzip -t` reports no errors.

## What to report

1. The three JSON lines from step 3 and whether live/resident matched exactly.
2. The exact-path line from step 3 and how far the model sits below it.
3. Any test failure with file, name and numbers; any worker crash and whether
   a rerun cleared it.
4. Browser: preflight text, the 59-residue pLDDT/pTM/time, the long-sequence
   allocator peak, and the Chrome GPU process peak from Activity Monitor.
5. The model and exact-path pLDDT/pTM pairs from step 7b, and the live-at-peak
   list for one long shape.
6. Whether the complex views agreed on the chains, and whether clicking the
   alignment-error matrix isolated the interface.
7. Whether the result archive unzipped cleanly in each browser you tried.
8. The chip (M1/M2/M3/M4, Pro/Max) and its memory.
