import { describe, expect, it } from "vitest";
import {
  formatKernelCostReport, kernelCostKey, kernelCostRows, kernelFlops, kernelGroupComparisons,
  monomerCostShapes, type KernelBaseline,
} from "../src/runtime/kernel-cost.js";

const shapes = monomerCostShapes(8, 4, 16);

describe("what a kernel has to compute", () => {
  it("reads one label whatever stack and dispatch wrote it", () => {
    expect(kernelCostKey("extra.msa-row-attention.flash-3")).toBe("msa-row-attention.flash");
    expect(kernelCostKey("triangle.incoming.contract")).toBe("triangle.contract");
    expect(kernelCostKey("triangle.outgoing.project-block-64")).toBe("triangle.project-block");
    expect(kernelCostKey("opm.contract")).toBe("opm.contract");
  });

  it("counts a block's whole contraction, not one dispatch of it", () => {
    // The triangle contraction is the model's only cubic kernel: every pair of
    // residues against every residue, once a hidden channel.
    const main = shapes.mainEvoformer;
    expect(kernelFlops("triangle.incoming.contract", main))
      .toBe(2 * main.length ** 3 * main.triangleHidden);
    // The outer product's contraction is the alignment depth contracted away
    // between two projections of the residues.
    expect(kernelFlops("opm.contract", shapes.extraMsa))
      .toBe(2 * (8 * 32) * 16 * (8 * 32));
  });

  it("says nothing for a kernel that moves data rather than contracting it", () => {
    for (const label of ["opm.normalize", "opm.pair-count", "triangle.incoming.input-statistics",
      "msa-row-attention.normalize", "extra.msa-column-global-attention.statistics"]) {
      expect(kernelFlops(label, shapes.extraMsa)).toBeUndefined();
    }
  });

  it("does not count column attention the extra stack never runs", () => {
    // The extra stack runs global column attention instead, and a stray label
    // must not be given the main stack's shape.
    expect(kernelFlops("msa-column-attention.flash", shapes.extraMsa)).toBeUndefined();
    expect(kernelFlops("msa-column-attention.flash", shapes.mainEvoformer)).toBeGreaterThan(0);
  });
});

describe("a profile read as a rate", () => {
  const entries = [
    { label: "triangle.incoming.contract-0", nanoseconds: 1_000_000 },
    { label: "triangle.incoming.contract-4", nanoseconds: 1_000_000 },
    { label: "opm.normalize-0", nanoseconds: 2_000_000 },
  ];

  it("groups a blocked kernel's dispatches and divides by the total", () => {
    const rows = kernelCostRows(entries, shapes.mainEvoformer);
    const contract = rows.find((row) => row.label === "triangle.incoming.contract")!;
    expect(contract.count).toBe(2);
    expect(contract.milliseconds).toBeCloseTo(2, 6);
    // Two milliseconds is two million nanoseconds, so the rate is the count
    // over 2e9: operations a second, in units of a million million.
    expect(contract.tflops).toBeCloseTo(contract.flops! / 2e9, 6);
    // Half the block's time is the normalization, which gets no rate.
    expect(contract.share).toBeCloseTo(0.5, 6);
    expect(rows.find((row) => row.label === "opm.normalize")!.tflops).toBeUndefined();
  });

  it("holds a reference kernel against the group of ours that does its work", () => {
    const baselines: readonly KernelBaseline[] = [{
      name: "gated dual projection", kernels: ["triangle.contract"],
      millisecondsPerBlock: 1, tflops: 50, stack: "mainEvoformer", length: 8, source: "a test",
    }];
    const rows = kernelCostRows(entries, shapes.mainEvoformer, baselines);
    const [group] = kernelGroupComparisons(rows, baselines, "mainEvoformer");
    expect(group?.labels).toEqual(["triangle.incoming.contract"]);
    expect(group?.milliseconds).toBeCloseTo(2, 6);
    expect(group?.millisecondRatio).toBeCloseTo(2, 6);
    expect(group?.rateRatio).toBeCloseTo(50 / group!.tflops!, 6);
    expect(formatKernelCostReport(rows, [group!], shapes.mainEvoformer))
      .toContain("gated dual projection");
  });

  it("refuses a rate measured at another head width, and says which", () => {
    const baselines: readonly KernelBaseline[] = [{
      name: "tri_flash", kernels: ["msa-row-attention.flash"],
      tflops: 107, headDim: 32, length: 8, source: "a test",
    }];
    const flash = [{ label: "extra.msa-row-attention.flash-0", nanoseconds: 1_000_000 }];
    // The extra stack attends over eight channels, the reference over 32.
    const [extra] = kernelCostRows(flash, shapes.extraMsa, baselines);
    expect(extra?.baseline?.name).toBe("tri_flash");
    expect(extra?.ratio).toBeUndefined();
    expect(extra?.mismatch).toContain("head 8");
    // The main stack attends over 32 and does compare.
    const [main] = kernelCostRows(flash, shapes.mainEvoformer, baselines);
    expect(main?.ratio).toBeCloseTo(107 / main!.tflops!, 6);
    expect(main?.mismatch).toBe("");
  });

  it("holds a per-block figure only against the block it was measured on", () => {
    const baselines: readonly KernelBaseline[] = [{
      name: "pallas_layer_norm", kernels: ["triangle.contract"],
      millisecondsPerBlock: 1, stack: "mainEvoformer", length: 8, source: "a test",
    }];
    const rows = kernelCostRows(entries, shapes.mainEvoformer, baselines);
    expect(kernelGroupComparisons(rows, baselines, "mainEvoformer")[0]?.millisecondRatio)
      .toBeCloseTo(2, 6);
    // An extra-MSA block holds other channel counts and another depth, so the
    // same milliseconds are not the same work.
    expect(kernelGroupComparisons(rows, baselines, "extraMsa")[0]?.millisecondRatio)
      .toBeUndefined();
    // A row that never says which block it came from is never used as one.
    const { stack: _named, ...unstackedRow } = baselines[0]!;
    const unstacked = [unstackedRow];
    expect(kernelGroupComparisons(rows, unstacked, "mainEvoformer")[0]?.millisecondRatio)
      .toBeUndefined();
  });

  it("reports no comparison where no reference names the kernel", () => {
    const rows = kernelCostRows(entries, shapes.mainEvoformer, []);
    expect(rows.every((row) => row.baseline === undefined && row.ratio === undefined)).toBe(true);
    expect(kernelGroupComparisons(rows, [])).toHaveLength(0);
  });
});
