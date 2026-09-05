# Custom templates

How someone uploads a structure and has it used as an AlphaFold template. This
began as a plan and is now a record: everything below was built on 2026-09-06,
and the numbers are measured rather than projected.

## What it does

Ubiquitin, from its sequence alone with no alignment at all:

|                | pLDDT | pTM   | RMSD to 1UBQ |
|----------------|-------|-------|--------------|
| no template    | 49.35 | 0.398 | 11.69 A      |
| with 1UBQ      | 94.26 | 0.800 | 0.38 A       |

ACE2, 597 residues, against chain A of 6M0J: pLDDT 97.03, 0.20 A RMSD, and the
template costs 55 MiB of live memory on top of the 241 the run needs without
one.

Checked against AlphaFold itself at every layer, not only end to end:

| layer | against | worst |
|---|---|---|
| structure reading | Biopython, on the same file | exact |
| every template feature | `templates.py`, `all_atom.py` | 6.1e-06 |
| the MSA row | `template_single_embedding` + `template_projection` | 7.4e-05 |
| the whole GPU module | official `TemplateEmbedding` with real weights | 1.9e-04 |

`tools/dump-template-features.ts` writes the features out and
`tools/capture_alphafold_template_reference.py` recomputes each one with the
official implementation, reading the structure a second time through Biopython
so the coordinates it compares against did not come from the port.

Two things came out of doing it that way and are worth keeping in mind:

- **The collapse is real, and measured.** With one template the pointwise
  attention softmaxes over a single key, so the pair update cannot depend on the
  query pair. Running the official module against a random query moves the
  output by exactly 0.0. The module is therefore a constant for the prediction,
  and its two matrices are composed on the host into one.
- **Undefined torsion angles are noise.** See the deviation note at the end.

## What changed from the plan

- The pair update is **not** held between recycles. Holding it would cost a
  pair-sized tensor for the whole trunk; recomputing costs a two-block stack at
  64 channels, well under a percent of a recycle.
- The output norm, the pointwise attention and the residual are **one kernel**
  writing straight into the model's pair. Written separately they were three
  pair-sized tensors nothing else reads: 91, 91 and 182 MiB at 597 residues.
- The memory estimate needed the template's *two* pair-shaped tensors, its own
  64-channel pair and the whole projection its triangle multiplication keeps
  beside it. With one of the two it under-predicted a templated run.
- Multimer is untouched and refuses a template rather than ignoring one.

Everything below is the reasoning the implementation followed, kept because it
explains why the module is shaped the way it is. Claims about what exists were
checked against this tree and against the AlphaFold source in the ColabFold
environment; the weight shapes come from the manifests the site serves.

## The surprise: most of the machinery is already here

`src/evoformer/template.ts` is not a stub. `QueryOnlyTemplateGpu.run` is the
real monomer template embedding with one piece missing:

| AlphaFold step | here |
|---|---|
| `embedding2d`: Linear(88 → 64) over the template pair features | `INIT_SHADER`, which fills with **the bias alone** — the features are all zero |
| `TemplatePairStack`, 2 blocks | `encodeTemplatePairBlock`, already running |
| `output_layer_norm` | `ATTENTION_NORMALIZE_SHADER` |
| `TemplatePointwiseAttention` | `VALUE_SHADER` + `OUTPUT_SHADER` |

So the module already runs the whole stack; it runs it on a zero template. Turn
the features on and it becomes the real thing.

The weights are already shipped too. `model/manifest.json` carries all 67
`template_embedding/` tensors, including `single_template_embedding/embedding2d`
at **[88, 64] with bias [64]** — the projection the zero features never use.

### Why the pointwise attention collapses to two matrices

`SingleTemplateEmbedding` takes `query_embedding` only for its shape and dtype;
its output does not depend on the pair representation. The pointwise attention
does — but it has no gating (`attention` in the manifest is `query_w, key_w,
value_w, output_w, output_b`, 4 heads of 16), and a softmax over a single
unmasked key is 1. AlphaFold pads to `max_templates = 4` and gives the padded
slots a `-1e9` bias, so with one real template the output is
`output_w · value_w · template_rep + output_b`, independent of the query and of
the recycle.

That is exactly the collapse `valueWeight`/`outputWeight` already implement. So
**one template costs one pass, before the trunk, not one per recycle.** Two or
more templates need the real attention and the query, and then it is per
recycle. Ship single-template first.

`reduce_msa_clusters_by_max_templates` is already why this port uses 508
clustered rows rather than 512, so the MSA budget needs no change.

## What is actually missing

1. **Two weight tensors.** `template_single_embedding` and `template_projection`
   are siblings of `template_embedding/` under `evoformer/`, and the monomer
   exporter's `embedding` section does not collect them. `model_1_ptm` sets
   `embed_torsion_angles: True`, so they are required, not optional. They are
   already in the *multimer* bundle, which is a useful cross-check of names.
2. **A structure parser**, PDB and ideally mmCIF, producing atom37 positions and
   masks plus the template's own sequence.
3. **A query-to-template alignment**, to map query residue i to template residue
   j (see below).
4. **The 88-channel pair feature**, fused into `embedding2d`.
5. **Torsion angle features**, the 57-channel per-residue input to
   `template_single_embedding`, and the extra MSA rows it produces.
6. **UI and packaging**: upload, chain choice, what the results ZIP records.

## Same idea as ColabFold, without the detour

ColabFold's `--custom-template-path` reaches its features the long way round,
because it is driving AlphaFold's own template pipeline from outside and that
pipeline expects a search. `mk_hhsearch_db` writes a fake pdb70 ffindex database
out of the user's files, `mk_template` runs hhsearch against it to "find" the
template the user has just handed it by name, and `HhsearchHitFeaturizer` then
shells out to kalign to realign the PDB's SEQRES to the hit hhsearch produced.
Everything upstream of the features is search infrastructure being used to
re-derive something already known.

The features on the far side are the ones we want. The route to them is not.

| ColabFold | here |
|---|---|
| builds a pdb70 database, runs hhsearch to find a named file | one pairwise alignment, query against template |
| mmCIF only; PDB converted through BioPython and patched for missing `_pdbx_audit_revision_history.revision_date` | parse the uploaded file directly, no round trip |
| kalign subprocess, to realign SEQRES to the hhsearch hit | no hit exists to realign to |
| `max_template_date`, release dates, obsolete-PDB mapping | dropped: they are PDB-provenance rules, meaningless for a file someone just made |
| `template_sum_probs` from hhsearch scores | 0, which is what their own `mk_mock_template` uses anyway |
| a template that fails a check is dropped with a log line | coverage and identity shown in the page, before the run |

Same idea, same features, a tenth of the moving parts, and nothing that needs a
filesystem — which matters, because all of this has to run in a browser tab.

## Phase 1 — features on the CPU (`src/input/template-features.ts`)

Produce AlphaFold's template features, indexed by *query* position:

- `template_aatype` [L] — the **template's** residue at the aligned position,
  20/unknown coded as AlphaFold codes it, zero where unaligned.
- `template_all_atom_positions` [L, 37, 3] and `template_all_atom_masks` [L, 37],
  gathered through the alignment, zero where unaligned.
- `template_pseudo_beta` [L, 3] and its mask — CB, or CA for glycine. The input
  embedder already computes pseudo-beta for the recycled positions
  (`pseudo_beta_coordinate` in `input-embedder.ts`); lift that.
- `template_sum_probs` — 0 for a custom template, as ColabFold's
  `mk_mock_template` does.

**The alignment.** kalign is the tool AlphaFold reaches for, but it uses it to
realign a PDB's SEQRES to an HHsearch *hit*. A custom template has no hit: the
problem is a single pairwise alignment of query against template sequence.
Recommendation: **write Gotoh (affine-gap Needleman–Wunsch) with BLOSUM62 in
TypeScript**, about 150 lines. 1500 × 1500 is 2.25M cells, which is nothing, it
is deterministic and unit-testable against known pairs, and it adds no wasm
payload to a page that already downloads 180 MiB of weights. Keep kalign-wasm
(biowasm ships v3) as a later option only if multi-template or MSA-style
alignment is wanted; it does not make the single-template case better.

Report the alignment back to the user — coverage and percent identity are what
tells someone whether their template is worth using.

## Phase 2 — the pair side (`src/evoformer/template.ts`)

Replace `INIT_SHADER` with a kernel that computes the 88 channels for a pair and
projects them in one pass. The channel layout, from
`modules.py:SingleTemplateEmbedding`, is:

```
39  dgram of pseudo-beta distances, 39 bins, 3.25 to 50.75 Å
 1  pseudo-beta mask outer product
22  template aatype one-hot, tiled along j
22  template aatype one-hot, tiled along i
 3  unit vector — zero, because model_1_ptm sets use_template_unit_vector: False
 1  backbone frame mask outer product (N, CA, C all present)
```

then `act *= mask_2d` and `Linear(88 → 64)`.

Never materialise the 88 channels: at 1500 residues that is 792 MiB in f32. Every
input is O(L) — positions, masks, aatype — so a fused kernel reads O(L) and
writes L² × 64. The recycle distogram kernel already in `input-embedder.ts` is
the same shape of computation with different bins; parameterise it rather than
writing a second one.

The rest of the module needs no change. `OUTPUT_SHADER` should gain an
add-into-the-pair variant so the L² × 128 result never exists as its own tensor.

**Memory.** Packed f16, the 64-channel activation is L² × 128 bytes: 34 MiB at
512, 128 MiB at 1000, 288 MiB at 1500, and the stack's whole-operand scratch is
the same again. It runs once, before the trunk, so at 1500 it lands against
pair + MSA (≈921 MiB) rather than the 1527 MiB trunk peak — probably free, but
this repo's own rule applies: measure `peakComposition` with `AFWEBGPU_MEMORY=1`
before promising it. At 512 it is 67 MiB against a 320 MiB peak.

## Phase 3 — the MSA side

`template_angle_feat` is 57 channels per residue: aatype one-hot (22), torsion
sin/cos (14), alternative torsion sin/cos (14), torsion mask (7). Then
`template_single_embedding` → relu → `template_projection`, appended to the MSA
as one extra row per template.

Computing torsions from coordinates (AlphaFold's `atom37_to_torsion_angles`)
needs three residue-constant tables this repo does not yet carry: chi atom
indices [21, 4, 4], chi mask [21, 4], and chi pi-periodic [21]. Add them to
`tools/export_residue_geometry.py` alongside the atom14/atom37 tables that are
already exported. The computation itself is O(L) on the CPU.

The multimer mock-template module already returns `msaRows` and `msaMask`
(`MultimerMockTemplateWeights.msaInputWeight`, `msaOutputWeight`,
`templateRows`), so the shape of this plumbing exists to copy from.

## Phase 4 — the page

- Upload a `.pdb`/`.cif` next to the alignment input; chain picker when the file
  has more than one; show coverage and identity after aligning.
- Templates apply to the MMseqs2 and custom-A3M paths alike; they are orthogonal
  to where the MSA came from.
- Put the template, the alignment and the coverage figures in the results ZIP —
  a prediction that used a template is not reproducible without them.
- The run log should say which template was used and how much of the query it
  covered.

## Phase 5 — multimer, later and separately

The multimer bundle already ships its template weights, including the 1D path.
But v3's `SingleTemplateEmbedding` is a different module: nine summed linears
(`template_pair_embedding_0..8`) instead of one `embedding2d`, and it takes the
**normalised query pair** as one of its inputs (`query_embedding_norm`). So the
multimer template embedding is *not* recycle-independent and cannot use the
collapse that makes the monomer case cheap. Treat it as its own project.

## Verification

Non-negotiable, per AGENTS.md: no reference tensor or tolerance was touched.
None was.

1. Capture an official reference with a real template through
   `tools/capture_alphafold_reference*.py`, one small case with one template.
2. Differential-test the feature builder against AlphaFold's own
   `template_pair_feat` and `template_angle_feat` for that case, before any GPU
   work — a wrong feature is much easier to see here than three modules later.
3. Differential-test `QueryOnlyTemplateGpu` (renamed) against the captured
   template pair representation.
4. Whole-model: pLDDT and pTM against the JAX run for the same template. The
   storage-flag lesson in the memory notes applies — a kernel test cannot see a
   plumbing gap, so the whole-prediction test is the one that matters.
5. A no-template run must produce bit-identical output to today's, since the
   constant-collapse path must still be taken when no template is given.

## What is still open

- Multiple templates. The pointwise attention would have to run for real, per
  recycle, with the query pair as its query. One template is the common case for
  an uploaded structure and it is the cheap one.
- Multimer, as its own project (see above).
- The template's own 64-channel pair is float32 while the trunk's activations
  are packed. Packing it would halve 87 MiB at 597 residues; the pair stack
  blocks already take a storage option, so it is a matter of threading it
  through and re-running the differential.
- The published bundle is `model1-ptm-q8-v2`. A bundle without
  `template_single_embedding` refuses a template with a message saying so
  rather than folding without one.

## Order of work

1. Export the two missing weight tensors; assert their shapes on load.
2. Structure parser + aligner + feature builder, with the differential test.
3. Residue-constant tables for torsions.
4. Fused `embedding2d` kernel; turn the pair path on for one template.
5. Torsion rows into the MSA.
6. Whole-model differential; memory measurement at 512 and 1500.
7. UI, ZIP contents, README.
8. Multiple templates (the real pointwise attention, per recycle) only if wanted.
9. Multimer, as its own project.
