import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import {
  adaptMultimerInvariantPointAttentionWeights,
  InvariantPointAttentionGpu,
  type MultimerInvariantPointAttentionWeights,
} from "../src/structure/ipa.js";
import { errorMetrics } from "../src/triangle/types.js";

const enabled = process.env.AFWEBGPU_GPU_TESTS === "1";

function values(length: number, scale: number, phase: number): Float32Array {
  return Float32Array.from({ length }, (_, index) => scale * Math.sin(index * 1.713 + phase));
}

function linear(source: Float32Array, rows: number, inner: number, columns: number,
  weight: Float32Array, bias?: Float32Array): Float32Array {
  const output = new Float32Array(rows * columns);
  for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) {
    let total = bias?.[column] ?? 0;
    for (let input = 0; input < inner; input += 1) {
      total += source[row * inner + input]! * weight[input * columns + column]!;
    }
    output[row * columns + column] = total;
  }
  return output;
}

function officialMultimerIpaReference(input: {
  readonly activations: Float32Array; readonly pair: Float32Array; readonly mask: Float32Array;
  readonly affine: Float32Array; readonly length: number; readonly channels: number; readonly pairChannels: number;
  readonly heads: number; readonly scalarQk: number; readonly scalarV: number;
  readonly pointQk: number; readonly pointV: number; readonly weights: MultimerInvariantPointAttentionWeights;
}): Float32Array {
  const { length: length, channels, pairChannels, heads, scalarQk, scalarV, pointQk, pointV, weights } = input;
  const pair = new Float32Array(input.pair.length);
  for (let row = 0; row < length * length; row += 1) {
    let mean = 0;
    for (let c = 0; c < pairChannels; c += 1) mean += input.pair[row * pairChannels + c]!;
    mean /= pairChannels;
    let variance = 0;
    for (let c = 0; c < pairChannels; c += 1) {
      const centered = input.pair[row * pairChannels + c]! - mean; variance += centered * centered;
    }
    const inverseStd = 1 / Math.sqrt(variance / pairChannels + 1e-5);
    for (let c = 0; c < pairChannels; c += 1) {
      pair[row * pairChannels + c] = (input.pair[row * pairChannels + c]! - mean) * inverseStd
        * weights.pairNormScale[c]! + weights.pairNormOffset[c]!;
    }
  }
  const qScalar = linear(input.activations, length, channels, heads * scalarQk, weights.queryScalarWeight);
  const kScalar = linear(input.activations, length, channels, heads * scalarQk, weights.keyScalarWeight);
  const vScalar = linear(input.activations, length, channels, heads * scalarV, weights.valueScalarWeight);
  const pointProjection = (weight: Float32Array, bias: Float32Array, points: number): Float32Array => {
    const local = linear(input.activations, length, channels, heads * 3 * points, weight, bias);
    const global = new Float32Array(length * heads * points * 3);
    for (let residue = 0; residue < length; residue += 1) for (let head = 0; head < heads; head += 1) {
      const affineBase = residue * 7; const tx = input.affine[affineBase + 4]!;
      const ty = input.affine[affineBase + 5]!; const tz = input.affine[affineBase + 6]!;
      for (let point = 0; point < points; point += 1) for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const localIndex = ((residue * heads + head) * 3 + coordinate) * points + point;
        const translation = coordinate === 0 ? tx : coordinate === 1 ? ty : tz;
        global[((residue * heads + head) * points + point) * 3 + coordinate] = local[localIndex]! + translation;
      }
    }
    return global;
  };
  const qPoint = pointProjection(weights.queryPointWeight, weights.queryPointBias, pointQk);
  const kPoint = pointProjection(weights.keyPointWeight, weights.keyPointBias, pointQk);
  const vPoint = pointProjection(weights.valuePointWeight, weights.valuePointBias, pointV);
  const attention = new Float32Array(heads * length * length);
  const logitScale = Math.sqrt(1 / 3);
  for (let head = 0; head < heads; head += 1) for (let query = 0; query < length; query += 1) {
    const row = new Float64Array(length); let maximum = -Infinity;
    for (let key = 0; key < length; key += 1) {
      let scalar = 0;
      for (let c = 0; c < scalarQk; c += 1) {
        scalar += qScalar[(query * heads + head) * scalarQk + c]!
          * kScalar[(key * heads + head) * scalarQk + c]! / Math.sqrt(scalarQk);
      }
      let distance = 0;
      for (let point = 0; point < pointQk; point += 1) for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        const q = qPoint[((query * heads + head) * pointQk + point) * 3 + coordinate]!;
        const k = kPoint[((key * heads + head) * pointQk + point) * 3 + coordinate]!;
        distance += (q - k) ** 2;
      }
      const pointWeight = Math.log1p(Math.exp(weights.trainablePointWeights[head]!))
        / Math.sqrt(Math.max(pointQk, 1) * 4.5);
      let pairBias = weights.attention2dBias[head]!;
      for (let c = 0; c < pairChannels; c += 1) {
        pairBias += pair[(query * length + key) * pairChannels + c]! * weights.attention2dWeight[c * heads + head]!;
      }
      row[key] = (scalar - 0.5 * pointWeight * distance + pairBias
        - 1e5 * (1 - input.mask[query]! * input.mask[key]!)) * logitScale;
      maximum = Math.max(maximum, row[key]!);
    }
    let sum = 0; for (let key = 0; key < length; key += 1) sum += Math.exp(row[key]! - maximum);
    for (let key = 0; key < length; key += 1) {
      attention[(head * length + query) * length + key] = Math.exp(row[key]! - maximum) / sum;
    }
  }
  const featureChannels = heads * (scalarV + 4 * pointV + pairChannels);
  const features = new Float32Array(length * featureChannels);
  for (let query = 0; query < length; query += 1) for (let head = 0; head < heads; head += 1) {
    for (let c = 0; c < scalarV; c += 1) {
      let total = 0; for (let key = 0; key < length; key += 1) {
        total += attention[(head * length + query) * length + key]!
          * vScalar[(key * heads + head) * scalarV + c]!;
      }
      features[query * featureChannels + head * scalarV + c] = total;
    }
    for (let point = 0; point < pointV; point += 1) {
      const local = [0, 0, 0];
      for (let key = 0; key < length; key += 1) for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        local[coordinate]! += attention[(head * length + query) * length + key]!
          * vPoint[((key * heads + head) * pointV + point) * 3 + coordinate]!;
      }
      local[0]! -= input.affine[query * 7 + 4]!; local[1]! -= input.affine[query * 7 + 5]!;
      local[2]! -= input.affine[query * 7 + 6]!;
      const scalarSize = heads * scalarV; const pointSize = heads * pointV; const index = head * pointV + point;
      for (let coordinate = 0; coordinate < 3; coordinate += 1) {
        features[query * featureChannels + scalarSize + coordinate * pointSize + index] = local[coordinate]!;
      }
      features[query * featureChannels + scalarSize + 3 * pointSize + index]
        = Math.sqrt(1e-8 + local[0]! ** 2 + local[1]! ** 2 + local[2]! ** 2);
    }
    for (let c = 0; c < pairChannels; c += 1) {
      let total = 0; for (let key = 0; key < length; key += 1) {
        total += attention[(head * length + query) * length + key]! * pair[(query * length + key) * pairChannels + c]!;
      }
      const offset = heads * scalarV + 4 * heads * pointV;
      features[query * featureChannels + offset + head * pairChannels + c] = total;
    }
  }
  return linear(features, length, featureChannels, channels, weights.outputWeight, weights.outputBias);
}

describe.skipIf(!enabled)("AlphaFold-Multimer invariant point attention WebGPU", () => {
  let device: GPUDevice;
  beforeAll(async () => {
    Object.assign(globalThis, globals); const adapter = await create([]).requestAdapter();
    if (adapter === null) throw new Error("no WebGPU adapter"); device = await adapter.requestDevice();
  });
  afterAll(() => device?.destroy());

  it("matches an independent native-layout Multimer-v3 reference", async () => {
    const length = 3; const channels = 4; const pairChannels = 3; const heads = 2;
    const scalarQk = 2; const scalarV = 2; const pointQk = 1; const pointV = 2;
    const featureChannels = heads * (scalarV + 4 * pointV + pairChannels);
    const weights: MultimerInvariantPointAttentionWeights = {
      pairNormScale: Float32Array.of(0.9, 1.1, 1.2), pairNormOffset: Float32Array.of(0.1, -0.2, 0.05),
      queryScalarWeight: values(channels * heads * scalarQk, 0.15, 0.1),
      keyScalarWeight: values(channels * heads * scalarQk, 0.12, 0.3),
      valueScalarWeight: values(channels * heads * scalarV, 0.17, 0.7),
      queryPointWeight: values(channels * heads * 3 * pointQk, 0.08, 0.2),
      queryPointBias: values(heads * 3 * pointQk, 0.04, 0.4),
      keyPointWeight: values(channels * heads * 3 * pointQk, 0.07, 0.6),
      keyPointBias: values(heads * 3 * pointQk, 0.03, 0.8),
      valuePointWeight: values(channels * heads * 3 * pointV, 0.09, 1.0),
      valuePointBias: values(heads * 3 * pointV, 0.02, 1.2),
      trainablePointWeights: Float32Array.of(0.3, -0.2),
      attention2dWeight: values(pairChannels * heads, 0.11, 1.4), attention2dBias: Float32Array.of(0.02, -0.03),
      outputWeight: values(featureChannels * channels, 0.13, 1.6), outputBias: values(channels, 0.04, 1.8),
    };
    const affine = Float32Array.of(1, 0, 0, 0, 0.1, -0.2, 0.3, 1, 0, 0, 0, -0.4, 0.2, 0.1,
      1, 0, 0, 0, 0.3, 0.4, -0.2);
    const nativeInput = {
      activations: values(length * channels, 0.7, 0.25), pair: values(length * length * pairChannels, 0.5, 0.5),
      mask: Float32Array.of(1, 1, 1), affine, length, channels, pairChannels, heads, scalarQk, scalarV,
      pointQk, pointV, weights,
    };
    const expected = officialMultimerIpaReference(nativeInput);
    const adapted = adaptMultimerInvariantPointAttentionWeights(
      weights, channels, pairChannels, heads, scalarQk, scalarV, pointQk, pointV,
    );
    const result = await new InvariantPointAttentionGpu(device).run({ ...nativeInput, weights: adapted, multimer: true });
    const metrics = errorMetrics(result.output, expected);
    expect(metrics.meanAbsoluteError).toBeLessThan(3e-6);
    expect(metrics.maxAbsoluteError).toBeLessThan(2e-5);
  });
});
