# Hardware qualification

AlphaFold2 WebGPU treats numerical correctness and performance as separate release gates. A device is qualified only when both the automatic and bounded-transition paths reproduce the official four-recycle `model_1_ptm` reference.

## Full-model qualification

The full reference captures are intentionally excluded from Git history. Run this command from a development workspace that contains:

```text
test/fixtures/evoformer/model1-a3m-59-stack/
test/fixtures/evoformer/model1-query-59-stack/
```

Then run:

```bash
npm ci
npm run check
npm run test:browser
npm run qualify:hardware | tee hardware-qualification.json
```

The qualification command runs four recycles twice: once with the automatically selected fast path and once with bounded transition scratch. It checks every recycle against the official AlphaFold reference and checks that the two WebGPU paths agree. The JSON report records the adapter, exposed limits and features, host platform, time, and measured allocator peaks.

An optional performance ceiling makes a known adapter fail on regression:

```bash
AFWEBGPU_MAX_MS=15000 npm run qualify:hardware
```

Use a ceiling based on a warm median from the same hardware and software stack. Do not compare a cold first run on one browser or driver with a warm run on another.

## Apple Silicon release check

Run the qualification in current stable Chrome on the target Mac as well as through the native command above. Record:

- Mac model and Apple chip.
- Installed unified memory.
- macOS and Chrome versions.
- Automatic and compact-path timings.
- Peak resident allocator memory.
- A 59-residue reference run and manual browser runs near 180 and 291 residues.

For the longer browser checks, use the sequences documented in the project issue or release checklist, MMseqs2 MSA mode, 508 clustered rows, 2,048 extra rows for Multimer (1,024 for monomer), and three recycles. Save the scores JSON and the browser run log. A release must not claim support for a Mac generation that has not completed these checks.

The run log must say `compact memory policy`, request bounded WebGPU limits, and report `Transition memory mode: chunked`. If macOS identification is unavailable in an experimental browser, append `?compact=1` to the URL.

## GB10 regression check

The NVIDIA GB10 remains the optimized-path regression host. Its release job should run:

```bash
AFWEBGPU_MAX_MS=15000 npm run qualify:hardware
```

The numerical thresholds are fixed in the qualification harness. Performance thresholds may be tightened after collecting repeated warm-run medians, but must not be relaxed merely to make a regression pass.
