import { describe, expect, it } from "vitest";
import { DAWN_MATRIX, presetDialect } from "../src/runtime/dialect.js";
import { GEMM_VARIANT_F32, setGemmVariant } from "../src/runtime/gemm.js";
import { projectionChoice, reductionChoice, refusedFastPaths } from "../src/runtime/kernel-report.js";
import { recordSubgroupMatrixConfigs, recordSubgroupRange } from "../src/runtime/subgroups.js";

const SIXTEEN = { componentType: "f16", resultComponentType: "f32", M: 16, N: 16, K: 16 };

/**
 * A device the way one arrives: the adapter carries the matrix configurations
 * and the subgroup range, and the dialect is whatever calibration settled.
 */
function stubDevice(options: {
  matrix?: boolean; features?: string[]; configs?: readonly unknown[]; width?: [number, number];
} = {}): GPUDevice {
  const device = {
    features: new Set((options.features ?? ["shader-f16", "subgroups"]) as GPUFeatureName[]),
    limits: { maxComputeWorkgroupStorageSize: 49152, maxComputeInvocationsPerWorkgroup: 1024 },
  } as unknown as GPUDevice;
  presetDialect(device, {
    subgroupEnable: "", subgroupSize: () => "",
    matrix: options.matrix === true ? DAWN_MATRIX : undefined,
  });
  recordSubgroupRange(device, { info: {
    subgroupMinSize: options.width?.[0] ?? 32, subgroupMaxSize: options.width?.[1] ?? 32,
  } } as unknown as GPUAdapter);
  recordSubgroupMatrixConfigs(device, {
    info: { subgroupMatrixConfigs: options.configs ?? [SIXTEEN] },
  } as unknown as GPUAdapter);
  return device;
}

describe("which kernel each operator settled on", () => {
  it("calls it a defect when the adapter has the units and a gate refuses them", () => {
    // The shape of a real bug: the device was opened without the feature this
    // implementation names it by, so the dialect probe found no matrix half
    // while the adapter went on reporting six configurations.
    setGemmVariant(GEMM_VARIANT_F32);
    const row = projectionChoice(stubDevice({ matrix: false }));
    expect(row.status).toBe("refused");
    expect(row.reason).toMatch(/opened without the feature/);
    expect(refusedFastPaths([row])).toHaveLength(1);
  });

  it("calls it this device when the adapter reports no units", () => {
    setGemmVariant(GEMM_VARIANT_F32);
    const row = projectionChoice(stubDevice({ matrix: false, configs: [] }));
    expect(row.status).toBe("absent");
    expect(refusedFastPaths([row])).toHaveLength(0);
  });

  it("names the gate that refused, not just that one did", () => {
    setGemmVariant(GEMM_VARIANT_F32);
    expect(projectionChoice(stubDevice({ matrix: true, features: ["subgroups"] })).reason)
      .toMatch(/shader-f16/);
    // SwiftShader fixes the width at four, which no matrix kernel here can use.
    expect(projectionChoice(stubDevice({ matrix: true, width: [4, 4] })).reason)
      .toMatch(/subgroup width/);
  });

  it("says nothing is wrong when the fast path won", () => {
    setGemmVariant({ precision: "matrix", inner: 8 });
    const row = projectionChoice(stubDevice({ matrix: true }));
    expect(row.status).toBe("chosen");
    expect(row.reason).toBe("");
    setGemmVariant(GEMM_VARIANT_F32);
  });

  it("separates a subgroup reduction this device cannot hold from one it lacks", () => {
    expect(reductionChoice(stubDevice({ width: [4, 4] })).status).toBe("refused");
    expect(reductionChoice(stubDevice({ features: ["shader-f16"] })).status).toBe("absent");
    expect(reductionChoice(stubDevice()).status).toBe("chosen");
  });
});
