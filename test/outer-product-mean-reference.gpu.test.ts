import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  OuterProductMeanGpu, type OuterProductMeanInput, type OuterProductMeanWeights,
} from "../src/evoformer/outer-product-mean.js";
import { errorMetrics } from "../src/triangle/types.js";
import { testGpu } from "./support/gpu-instance.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

const LAYER_NORM_EPSILON = 1e-5;
const NORMALIZATION_EPSILON = 1e-3;

/**
 * The outer product mean, written the way AlphaFold states it.
 *
 * Independent of every shader: the normalization, both projections, the
 * contraction over sequences and the output projection are each a plain loop
 * here, so a rewrite of any of them has something to fail against. The GPU
 * path instead blocks the contraction, packs its operands, and runs the
 * projections as tiled GEMMs, and none of that may change the answer.
 */
function reference(input: OuterProductMeanInput): Float32Array {
  const { activations, mask, sequences, length, cM, cOuter, cZ, weights } = input;
  const rows = sequences * length;
  const normalized = new Float32Array(rows * cM);
  for (let row = 0; row < rows; row += 1) {
    let sum = 0;
    for (let c = 0; c < cM; c += 1) sum += activations[row * cM + c]!;
    const mean = sum / cM;
    let squared = 0;
    for (let c = 0; c < cM; c += 1) squared += (activations[row * cM + c]! - mean) ** 2;
    const inverseStd = 1 / Math.sqrt(squared / cM + LAYER_NORM_EPSILON);
    for (let c = 0; c < cM; c += 1) {
      normalized[row * cM + c] = (activations[row * cM + c]! - mean) * inverseStd
        * weights.layerNormScale[c]! + weights.layerNormOffset[c]!;
    }
  }
  const project = (weight: Float32Array, bias: Float32Array): Float32Array => {
    const out = new Float32Array(rows * cOuter);
    for (let row = 0; row < rows; row += 1) {
      for (let outer = 0; outer < cOuter; outer += 1) {
        let total = bias[outer]!;
        for (let c = 0; c < cM; c += 1) {
          total += normalized[row * cM + c]! * weight[c * cOuter + outer]!;
        }
        out[row * cOuter + outer] = mask[row]! * total;
      }
    }
    return out;
  };
  const left = project(weights.leftWeight, weights.leftBias);
  const right = project(weights.rightWeight, weights.rightBias);
  const output = new Float32Array(length * length * cZ);
  const pair = new Float32Array(cOuter * cOuter);
  for (let i = 0; i < length; i += 1) {
    for (let j = 0; j < length; j += 1) {
      pair.fill(0);
      let count = 0;
      for (let sequence = 0; sequence < sequences; sequence += 1) {
        count += mask[sequence * length + i]! * mask[sequence * length + j]!;
        for (let a = 0; a < cOuter; a += 1) {
          const leftValue = left[(sequence * length + i) * cOuter + a]!;
          for (let b = 0; b < cOuter; b += 1) {
            pair[a * cOuter + b]! += leftValue * right[(sequence * length + j) * cOuter + b]!;
          }
        }
      }
      const scale = 1 / (NORMALIZATION_EPSILON + count);
      for (let z = 0; z < cZ; z += 1) {
        let total = weights.outputBias[z]!;
        for (let k = 0; k < cOuter * cOuter; k += 1) {
          total += pair[k]! * weights.outputWeight[k * cZ + z]!;
        }
        output[(i * length + j) * cZ + z] = total * scale;
      }
    }
  }
  return output;
}

/** Deterministic and not uniform: a zero-filled operand hides a layout error. */
function spread(count: number, seed: number): Float32Array {
  const values = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    values[index] = Math.sin((index + 1) * seed) * 0.7;
  }
  return values;
}

describe.skipIf(!enabled)("outer product mean against an independent reference", () => {
  let gpu: GPU;
  let device: GPUDevice;

  beforeAll(async () => {
    gpu = testGpu();
    const adapter = await gpu.requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter is available");
    device = await adapter.requestDevice();
  });

  afterAll(() => device?.destroy());

  it("matches whatever the projections and the contraction are spelled as", async () => {
    // The outer channel count is the released model's, so the projection's
    // column range is the whole tile it is built for. The length and depth are
    // small, and the contraction is blocked so that the block is not the whole.
    // A block's output window is bound at an offset, which WebGPU requires to
    // be a multiple of 256 bytes: rowBlock * length * cZ * 4 is 1,536 here.
    const sequences = 9; const length = 8; const cM = 12; const cOuter = 32; const cZ = 16;
    const mask = new Float32Array(sequences * length).fill(1);
    // A masked row must drop out of both the product and the pair count.
    for (let index = 0; index < sequences * length; index += 3) mask[index] = 0;
    const weights: OuterProductMeanWeights = {
      layerNormScale: spread(cM, 0.31), layerNormOffset: spread(cM, 0.17),
      leftWeight: spread(cM * cOuter, 0.11), leftBias: spread(cOuter, 0.23),
      rightWeight: spread(cM * cOuter, 0.13), rightBias: spread(cOuter, 0.29),
      outputWeight: spread(cOuter * cOuter * cZ, 0.07), outputBias: spread(cZ, 0.37),
    };
    const input: OuterProductMeanInput = {
      activations: spread(sequences * length * cM, 0.19), mask,
      sequences, length, cM, cOuter, cZ, weights,
      layerNormEpsilon: LAYER_NORM_EPSILON, normalizationEpsilon: NORMALIZATION_EPSILON,
      rowBlockResidues: 3,
    };
    const expected = reference(input);
    const actual = await new OuterProductMeanGpu(device).run(input);
    const error = errorMetrics(actual.output, expected);
    expect(error.meanAbsoluteError).toBeLessThan(2e-5);
    expect(error.maxAbsoluteError).toBeLessThan(2e-4);
  });
});
