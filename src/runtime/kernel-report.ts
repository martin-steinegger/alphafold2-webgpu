/**
 * Which kernel each operator settled on, and where a faster one was refused.
 *
 * A selection gate that says no says nothing else. One wrong feature name once
 * turned off every matrix kernel on a card that has matrix units, cost a fold
 * twice its time, and no log line mentioned it: the model simply ran the
 * kernels it had. The distinction that matters is not which kernel won but
 * whether the faster one was ever measured.
 *
 * So each row carries a status. "slower" means the kernel was built, measured
 * and lost, which is the selection working. "refused" means this adapter
 * reports the hardware and a gate declined it anyway, which is nearly always a
 * bug here rather than a property of the device. "absent" means the hardware
 * is not there, which is the ordinary case on a software adapter.
 */
import {
  ATTENTION_MATRIX_QUERY_TILE, ATTENTION_MATRIX_SUBGROUP_SIZE, attentionMatrixShape,
  attentionMatrixStorageBytes,
} from "../evoformer/attention-matrix.js";
import { attentionFlashCandidates, calibrateAttentionFlashKernel } from "../evoformer/attention-calibration.js";
import { gemmVariant, MATRIX_LANES } from "./gemm.js";
import { gemmVariantCandidates, gemmVariantName, selectMatrixShape } from "./gemm-selection.js";
import { subgroupMatrixConfigs, supportsSubgroupSize } from "./subgroups.js";
import { dialect } from "./dialect.js";
import { rowNormalizeLayout } from "./reduction.js";

export type KernelStatus = "chosen" | "slower" | "refused" | "absent";

export interface KernelChoice {
  /** The operator, as a reader of a log would name it. */
  readonly operator: string;
  /** The kernel it runs. */
  readonly chosen: string;
  /** The faster kernel this row is about, and what became of it. */
  readonly fastPath: string;
  readonly status: KernelStatus;
  /** Why, where the status is not "chosen". Empty otherwise. */
  readonly reason: string;
}

/** The first gate that refuses, or undefined where none does. */
function refusal(gates: readonly (readonly [boolean, string])[]): string | undefined {
  return gates.find(([open]) => !open)?.[1];
}

function matrixGates(device: GPUDevice): readonly (readonly [boolean, string])[] {
  return [
    [dialect(device).matrix !== undefined,
      "the device accepts no cooperative-matrix directive, so it was opened without the feature"],
    [device.features.has("shader-f16" as GPUFeatureName), "the device has no shader-f16"],
    [supportsSubgroupSize(device, MATRIX_LANES),
      `the subgroup width cannot be held at ${MATRIX_LANES}`],
  ];
}

export function projectionChoice(device: GPUDevice): KernelChoice {
  const configs = subgroupMatrixConfigs(device);
  const chosen = gemmVariantName(gemmVariant());
  const row = { operator: "projection", chosen, fastPath: "matrix" } as const;
  if (chosen.startsWith("matrix")) return { ...row, status: "chosen", reason: "" };
  if (configs.length === 0) {
    return { ...row, status: "absent", reason: "the adapter reports no matrix configurations" };
  }
  const gate = refusal([...matrixGates(device),
    [selectMatrixShape(device, configs) !== undefined,
      "no configuration this kernel can use, of the ones reported"]]);
  if (gate !== undefined) return { ...row, status: "refused", reason: gate };
  const offered = gemmVariantCandidates(device, configs)
    .some((variant) => variant.precision === "matrix");
  return offered
    ? { ...row, status: "slower", reason: "measured against the others and lost" }
    : { ...row, status: "refused", reason: "the candidate list carries no matrix variant" };
}

export async function attentionChoice(
  device: GPUDevice, headDim: number,
): Promise<KernelChoice> {
  const chosen = (await calibrateAttentionFlashKernel(device, headDim)).variant;
  const row = { operator: `attention, head ${headDim}`, chosen, fastPath: "matrix" } as const;
  if (chosen === "matrix") return { ...row, status: "chosen", reason: "" };
  const configs = subgroupMatrixConfigs(device);
  if (configs.length === 0) {
    return { ...row, status: "absent", reason: "the adapter reports no matrix configurations" };
  }
  const storage = attentionMatrixStorageBytes(headDim);
  const gate = refusal([...matrixGates(device),
    [device.limits.maxComputeInvocationsPerWorkgroup >= ATTENTION_MATRIX_QUERY_TILE * 2,
      `the device grants ${device.limits.maxComputeInvocationsPerWorkgroup} invocations a workgroup,`
      + ` below the ${ATTENTION_MATRIX_QUERY_TILE * 2} this kernel lays out`],
    [device.limits.maxComputeWorkgroupStorageSize >= storage,
      `the device grants ${device.limits.maxComputeWorkgroupStorageSize} bytes of workgroup storage,`
      + ` below the ${storage} this kernel stages`],
    [attentionMatrixShape(device, headDim, configs) !== undefined,
      "no configuration this kernel can use, of the ones reported"]]);
  if (gate !== undefined) return { ...row, status: "refused", reason: gate };
  return attentionFlashCandidates(device, headDim).includes("matrix")
    ? { ...row, status: "slower", reason: "measured against the others and lost" }
    : { ...row, status: "refused", reason: "the candidate list carries no matrix variant" };
}

export function reductionChoice(device: GPUDevice): KernelChoice {
  const rows = rowNormalizeLayout(device).rowsPerWorkgroup;
  const subgroups = device.features.has("subgroups" as GPUFeatureName);
  const chosen = rows > 1 ? "subgroup rows" : subgroups ? "subgroup workgroup" : "tree";
  const row = { operator: "row normalize", chosen, fastPath: "subgroup rows" } as const;
  if (rows > 1) return { ...row, status: "chosen", reason: "" };
  if (!subgroups) {
    return { ...row, status: "absent", reason: "the device has no subgroups" };
  }
  return {
    ...row, status: "refused",
    reason: `the subgroup width cannot be held at ${ATTENTION_MATRIX_SUBGROUP_SIZE}`,
  };
}

/**
 * Every operator that has a fast path worth reporting.
 *
 * The attention rows measure where nothing has measured yet, and read a cached
 * answer where a fold already has. Head widths 8 and 32 are the two the model
 * builds.
 */
export async function kernelSelectionReport(device: GPUDevice): Promise<readonly KernelChoice[]> {
  const attention = await Promise.all([8, 32].map((headDim) => attentionChoice(device, headDim)));
  return [projectionChoice(device), ...attention, reductionChoice(device)];
}

/** The rows a caller should treat as a defect here rather than as this device. */
export function refusedFastPaths(rows: readonly KernelChoice[]): readonly KernelChoice[] {
  return rows.filter((row) => row.status === "refused");
}

export function formatKernelReport(rows: readonly KernelChoice[]): string {
  return rows.map((row) => {
    const note = row.status === "chosen" ? "" : `  (${row.fastPath}: ${row.status}, ${row.reason})`;
    return `${row.operator.padEnd(20)} ${row.chosen}${note}`;
  }).join("\n");
}
