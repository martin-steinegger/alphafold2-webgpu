/**
 * Arithmetic a second for every Evoformer kernel, against ColabFold's Pallas.
 *
 * Usage:
 *   AFWEBGPU_PROFILE=1 AFWEBGPU_PROFILE_JSON=run.json tsx tools/predict-a3m.ts <a3m> 512 5120 1
 *   tsx tools/kernel-tflops.ts run.json [baselines.json]
 *
 * The fold is the slow half and the arithmetic is the cheap half, so they are
 * separate: one profile can be read again against a changed baseline table, and
 * two profiles can be compared against each other without folding twice.
 *
 * A rate, not a millisecond, is what carries across a change of length or of
 * how a kernel is blocked. Where a kernel moves data rather than contracting
 * it, this prints no rate: a FLOP count for a layer normalization says only
 * that it has few flops, and the Pallas row for that kernel says the same.
 */
import { readFileSync } from "node:fs";
import {
  type EvoformerCostShape, type KernelBaseline, formatKernelCostReport, kernelCostRows,
  kernelGroupComparisons, monomerCostShapes,
} from "../src/runtime/kernel-cost.js";

interface SavedBlock {
  readonly block: number;
  readonly method: string;
  readonly wallMilliseconds: number;
  readonly entries: readonly { readonly label: string; readonly nanoseconds: number }[];
}

interface SavedProfile {
  readonly file?: string;
  readonly length: number;
  readonly msaSequences: number;
  readonly extraSequences: number;
  readonly extraMsa: SavedBlock;
  readonly mainEvoformer: SavedBlock;
}

const [profilePath, baselinePath = "tools/pallas-baselines.json"] = process.argv.slice(2);
if (profilePath === undefined) {
  console.error("usage: tsx tools/kernel-tflops.ts <profile.json> [baselines.json]");
  console.error("write the profile with AFWEBGPU_PROFILE=1 AFWEBGPU_PROFILE_JSON=<file> on a fold");
  process.exit(2);
}

const profile = JSON.parse(readFileSync(profilePath, "utf8")) as SavedProfile;
const saved = JSON.parse(readFileSync(baselinePath, "utf8")) as {
  readonly device?: string; readonly baselines: readonly KernelBaseline[];
};
const shapes = monomerCostShapes(profile.length, profile.msaSequences, profile.extraSequences);

console.log(`${profile.file ?? profilePath}: ${profile.length} residues, `
  + `${profile.msaSequences} clustered rows, ${profile.extraSequences} extra`);
console.log(`baselines: ${baselinePath}${saved.device === undefined ? "" : ` on ${saved.device}`}`);

const report = (
  name: string, stack: "mainEvoformer" | "extraMsa", block: SavedBlock, shape: EvoformerCostShape,
): void => {
  const rows = kernelCostRows(block.entries, shape, saved.baselines);
  const groups = kernelGroupComparisons(rows, saved.baselines, stack);
  const gpuMilliseconds = block.entries.reduce((sum, entry) => sum + entry.nanoseconds, 0) / 1e6;
  const counted = rows.reduce((sum, row) => sum + (row.flops ?? 0), 0);
  console.log(`\n== ${name} block ${block.block} ==  ${block.entries.length} dispatches, `
    + `gpu ${gpuMilliseconds.toFixed(3)} ms, ${(counted / 1e9).toFixed(1)} GFLOP counted, `
    + `${(counted / gpuMilliseconds / 1e9).toFixed(1)} TFLOP/s over the block`);
  console.log(formatKernelCostReport(rows, groups, shape));
};

report("extra-MSA", "extraMsa", profile.extraMsa, shapes.extraMsa);
report("main-Evoformer", "mainEvoformer", profile.mainEvoformer, shapes.mainEvoformer);
