/**
 * How large every binding of a prediction is, without running one.
 *
 * A prediction's sizes are decided entirely by its shape, so the length past
 * which something no longer fits is a number this can compute rather than a
 * fold someone has to wait for. That matters because the ceiling is not
 * memory: a storage binding may cover 2 GiB whatever the card holds, so a
 * tensor that is bound whole caps the model at a length no amount of VRAM
 * lifts. A 3,300-residue tetramer met one such binding and failed in thirty
 * seconds after planning a 13 GB peak on a 97 GB card.
 *
 * Nothing here restates the windowing arithmetic. Every size comes from the
 * same exported helper the encoder calls, so a change to how a tensor is
 * blocked moves this report with it. What is written out by hand is only the
 * list of tensors, which is why coverage is stated rather than implied: this
 * enumerates the bindings that grow with the square of the length, because
 * those are the only ones that reach 2 GiB at a length anyone will run. Row
 * tensors grow with the alignment depth and are already held under a scratch
 * budget far below the limit.
 *
 * That claim was tested rather than assumed. probeBindings recorded all 229
 * binding labels a fold makes at two lengths, and of those only the fifteen
 * below grow with the square. The probe cannot replace this file, because a
 * pooled buffer keeps the label of whatever created it, so three of its rows
 * scaled impossibly; each formula here was read back out of the allocation
 * that makes it.
 */
import { ATTENTION_WINDOW_MAX_SCALE, ATTENTION_WINDOW_TARGET_BYTES, attentionBatchWindow, attentionPairBiasStride } from "../evoformer/attention.js";
import {
  GLOBAL_GATE_TARGET_BYTES, globalAttentionGateRows, TRIANGLE_BLOCK_TARGET_BYTES, triangleBlockRows,
} from "../evoformer/block.js";
import { OUTER_PRODUCT_BLOCK_LIMIT_BYTES, outerProductMeanRowBlock } from "../evoformer/outer-product-mean.js";
import { wholeProjectionStride } from "../triangle/shaders.js";
import { TRANSITION_CHUNK_TARGET_BYTES, transitionChunkRows } from "../evoformer/transition.js";
import { type ActivationStorage, storageWords } from "./storage.js";
import { planShards } from "./sharded.js";
import { scratchBudget } from "./scratch-budget.js";

/** The 2 GiB a Vulkan storage binding may cover, which is not ours to raise. */
export const STORAGE_BINDING_LIMIT_BYTES = 2_147_483_644;

export interface PredictionShape {
  readonly length: number;
  readonly msaSequences: number;
  readonly extraSequences: number;
  readonly multimer?: boolean;
  /** Whether the confidence head reduces the alignment error as it goes. */
  readonly reducedConfidence?: boolean;
  readonly pairStorage?: ActivationStorage;
  readonly msaStorage?: ActivationStorage;
  readonly triangleWholeStorage?: ActivationStorage;
  /** What one binding may cover. Defaults to the Vulkan ceiling. */
  readonly bindingLimitBytes?: number;
}

/** How a tensor reaches a shader, which is what decides whether it has a ceiling. */
export type BindingKind =
  /** Bound whole: its size is its binding, so it caps the model. */
  | "whole"
  /** Cut into bindings that each fit, so it has no ceiling of its own. */
  | "shards"
  /** Bound a window at a time, held under a budget rather than the limit. */
  | "window";

export interface BindingSize {
  readonly label: string;
  /** Bytes of the tensor entire, which is what a device has to hold. */
  readonly totalBytes: number;
  /** Bytes of the largest single binding, which is what the limit applies to. */
  readonly bindingBytes: number;
  readonly kind: BindingKind;
  /** Set where the binding is past the limit, saying what it would take. */
  readonly exceedsBy?: number;
}

const bytesOf = (storage: ActivationStorage): number => storage === "f16" ? 2 : 4;

/**
 * Every binding that grows with the square of the length.
 *
 * Sorted largest binding first, which is the order they become ceilings in.
 */
export function predictionBindingSizes(shape: PredictionShape): readonly BindingSize[] {
  const { length, msaSequences, extraSequences } = shape;
  if (![length, msaSequences, extraSequences].every((v) => Number.isSafeInteger(v) && v > 0)) {
    throw new RangeError("a prediction shape needs positive lengths and depths");
  }
  const limit = shape.bindingLimitBytes ?? STORAGE_BINDING_LIMIT_BYTES;
  const pairStorage = shape.pairStorage ?? "f16";
  const wholeStorage = shape.triangleWholeStorage ?? "f16";
  const pairBytes = bytesOf(pairStorage);
  const pairs = length * length;
  const cZ = 128;
  const cOuter = 32;
  const hidden = 128;
  const sizes: BindingSize[] = [];
  const add = (label: string, totalBytes: number, kind: BindingKind, bindingBytes = totalBytes): void => {
    const entry: BindingSize = { label, totalBytes, bindingBytes, kind,
      ...(bindingBytes > limit ? { exceedsBy: bindingBytes - limit } : {}) };
    sizes.push(entry);
  };
  const sharded = (label: string, elements: number, align: number, elementBytes: number): void => {
    const layout = planShards(elements, align, limit, elementBytes);
    add(label, elements * elementBytes, layout.count === 1 ? "whole" : "shards",
      layout.shardElements * elementBytes);
  };

  // The pair itself, and the triangle's whole projection, which matches it.
  sharded("pair", pairs * cZ, cZ, pairBytes);
  // Written a shard at a time by the projection, read a channel group at a
  // time by the contraction; the projection's shards are the larger of the two
  // and are what a device has to allow.
  const wholeStride = wholeProjectionStride(length);
  sharded("triangle.whole", wholeStride * hidden, 2, bytesOf(wholeStorage));
  // The triangle writes a pair-shaped output and a mean and variance a pair.
  sharded("triangle.output", pairs * cZ, cZ, pairBytes);
  add("triangle.statistics", pairs * 2 * 4, "whole");

  // Blocked over the first residue axis, so a budget bounds them, not the length.
  const triangleBlock = triangleBlockRows(length, cZ, hidden, scratchBudget(TRIANGLE_BLOCK_TARGET_BYTES));
  add("triangle.blocked", triangleBlock * length * hidden * 4, "window");
  add("triangle.contracted", triangleBlock * length * hidden * 4, "window");
  add("triangle.gate", triangleBlock * length * cZ * 4, "window");

  // The outer product's contraction, packed where the matrix units want it.
  const outerStorage: ActivationStorage = pairStorage === "f32" ? "f32" : "f16";
  const rowBlock = outerProductMeanRowBlock(length, cOuter,
    scratchBudget(OUTER_PRODUCT_BLOCK_LIMIT_BYTES), outerStorage);
  add("opm.outer", storageWords(rowBlock * length * cOuter * cOuter, outerStorage) * 4, "window");
  add("opm.pair-count", pairs * 4, "whole");

  // Attention over the pair: a bias of one value a head and query pair, and
  // the batch window the flash kernels read.
  add("attention.pair-bias", 8 * length * attentionPairBiasStride(length) * 4, "whole");
  const pairAttentionWindow = attentionBatchWindow(length, length, cZ,
    Math.min(scratchBudget(ATTENTION_WINDOW_TARGET_BYTES, ATTENTION_WINDOW_MAX_SCALE), limit));
  add("attention.pair-window", pairAttentionWindow * length * cZ * 4, "window");

  // The pair transition, chunked against the binding limit itself.
  const pairChunk = transitionChunkRows(pairs, cZ, 4 * cZ, limit);
  add("pair-transition.chunk", pairChunk * 4 * cZ * 4, "window");

  // The residual add over the pair, windowed since 01c4f40. Whole before it,
  // which is what stopped a tetramer.
  const residualWords = storageWords(pairs * cZ, pairStorage);
  add("pair.residual-add", residualWords * 4, "window", Math.min(residualWords * 4, limit));

  // The mask over residue pairs, uploaded whole and read whole.
  add("monomer.pair-mask", pairs * 4, "whole");

  if (shape.multimer === true) {
    // The multimer template holds pair-shaped tensors of its own, read a
    // window of rows at a time. Its statistics are one mean and variance a
    // pair, which is small enough to bind whole.
    const templateChannels = 64;
    add("multimer-template.pair", pairs * templateChannels * 4, "window",
      Math.min(pairs * templateChannels * 4, limit));
    add("multimer-template.query-statistics", pairs * 2 * 4, "whole");
    // The update it writes back is pair-shaped, and it walks that in windows
    // too, so what it costs is memory rather than a ceiling.
    const updateWords = storageWords(pairs * cZ, pairStorage);
    add("multimer-template.pair-update", updateWords * 4, "window",
      Math.min(updateWords * 4, limit));
    // Its own blocks run the pair machinery at 64 channels rather than 128, so
    // every tensor of theirs is half of one already listed, except the
    // attention bias: that one has four heads where the trunk's has eight.
    add("multimer-template.attention.pair-bias",
      4 * length * attentionPairBiasStride(length) * 4, "whole");
  }

  // The confidence head projects the pair to alignment-error bins. Reduced, it
  // walks windows sized against the binding limit; whole, the tensor is the
  // binding and it is the largest one a prediction makes.
  const paeBins = 64;
  if (shape.reducedConfidence === false) {
    add("confidence.pae-logits", pairs * paeBins * 4, "whole");
  } else {
    add("confidence.pae-logits", pairs * paeBins * 4, "window",
      Math.min(pairs * paeBins * 4, limit));
  }

  // Reduced, the head still keeps one alignment error and one TM term a pair,
  // and writes both from every window, so each is bound whole.
  if (shape.reducedConfidence !== false) {
    add("confidence.predicted-aligned-error", pairs * 4, "whole");
    add("confidence.tm-score-terms", pairs * 4, "whole");
  }

  // Invariant point attention biases every head by the pair, scores every head
  // over the same pairs, and keeps a mean and variance a pair. The bias and the
  // logits are the same size, so the model has two tensors at its lowest
  // ceiling rather than the one that was first reported.
  const ipaHeads = 12;
  add("ipa.pair-bias", ipaHeads * pairs * 4, "whole");
  add("ipa.logits", ipaHeads * pairs * 4, "whole");
  add("ipa.pair-statistics", pairs * 2 * 4, "whole");

  return [...sizes].sort((left, right) => right.bindingBytes - left.bindingBytes);
}

/** The bindings a shape cannot make, which is what caps it. */
export function oversizedBindings(shape: PredictionShape): readonly BindingSize[] {
  return predictionBindingSizes(shape).filter((size) => size.exceedsBy !== undefined);
}

/**
 * The longest prediction of this shape whose every binding fits.
 *
 * Found by bisection rather than by algebra: the sizes come from the same
 * blocking helpers the encoder uses, and those round and align, so the length
 * where one crosses is not worth deriving in closed form.
 */
export function maximumPredictionLength(
  shape: Omit<PredictionShape, "length">, ceiling = 16_384,
): number {
  const fits = (length: number): boolean => oversizedBindings({ ...shape, length }).length === 0;
  if (!fits(1)) return 0;
  if (fits(ceiling)) return ceiling;
  let low = 1;
  let high = ceiling;
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2);
    if (fits(middle)) low = middle; else high = middle;
  }
  return low;
}

export function formatBindingSizes(sizes: readonly BindingSize[]): string {
  const mib = (bytes: number): string => `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  return sizes.map((size) => `${mib(size.bindingBytes).padStart(11)} ${size.kind.padEnd(7)}`
    + ` ${size.label}${size.exceedsBy === undefined ? "" : `  OVER by ${mib(size.exceedsBy)}`}`
    + (size.totalBytes === size.bindingBytes ? "" : ` (of ${mib(size.totalBytes)})`)).join("\n");
}
