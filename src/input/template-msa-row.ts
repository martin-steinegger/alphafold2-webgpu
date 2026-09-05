import { TEMPLATE_ANGLE_CHANNELS } from "./template-torsions.js";

/**
 * `template_single_embedding` and `template_projection`, the two layers that
 * turn a template's torsion angles into an MSA row.
 *
 * These live outside the template embedding module in AlphaFold, as siblings of
 * it under the evoformer, which is why a bundle can carry the whole template
 * pair stack and still not have them.
 */
export interface TemplateMsaWeights {
  /** `[57, 256]`. */
  readonly inputWeight: Float32Array;
  readonly inputBias: Float32Array;
  /** `[256, 256]`. */
  readonly outputWeight: Float32Array;
  readonly outputBias: Float32Array;
}

/**
 * The MSA row a template contributes, `[length, channels]`.
 *
 * One matrix multiply of 57 by 256 and one of 256 by 256, per residue: about
 * 120 MFLOP for a 1500-residue chain, once for the whole prediction rather than
 * once per recycle, since a template does not change between them. It stays on
 * the CPU because its result is uploaded either way, and a kernel for it would
 * be a kernel to bind, dispatch and test for no measurable time.
 */
export function templateMsaRow(
  angleFeatures: Float32Array, length: number, weights: TemplateMsaWeights,
): Float32Array {
  const channels = weights.inputBias.length;
  if (weights.inputWeight.length !== TEMPLATE_ANGLE_CHANNELS * channels) {
    throw new Error(
      `template_single_embedding is ${weights.inputWeight.length} values, not `
      + `${TEMPLATE_ANGLE_CHANNELS} by ${channels}`,
    );
  }
  if (weights.outputWeight.length !== channels * channels) {
    throw new Error(`template_projection is ${weights.outputWeight.length} values, not ${channels} squared`);
  }
  const hidden = new Float32Array(channels);
  const row = new Float32Array(length * channels);
  for (let residue = 0; residue < length; residue += 1) {
    const source = residue * TEMPLATE_ANGLE_CHANNELS;
    hidden.set(weights.inputBias);
    for (let feature = 0; feature < TEMPLATE_ANGLE_CHANNELS; feature += 1) {
      const value = angleFeatures[source + feature]!;
      if (value === 0) continue;
      const weightRow = feature * channels;
      for (let channel = 0; channel < channels; channel += 1) {
        hidden[channel] = hidden[channel]! + value * weights.inputWeight[weightRow + channel]!;
      }
    }
    const target = residue * channels;
    row.set(weights.outputBias, target);
    for (let input = 0; input < channels; input += 1) {
      // The relu between the two layers, which is also why the second loop can
      // skip most of its work: a template's angle features leave a good share
      // of the hidden channels at zero.
      const value = hidden[input]! > 0 ? hidden[input]! : 0;
      if (value === 0) continue;
      const weightRow = input * channels;
      for (let channel = 0; channel < channels; channel += 1) {
        row[target + channel] = row[target + channel]! + value * weights.outputWeight[weightRow + channel]!;
      }
    }
  }
  return row;
}
