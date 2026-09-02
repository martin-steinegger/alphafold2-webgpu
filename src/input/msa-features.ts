/**
 * Compact clustered-MSA features.
 *
 * AlphaFold's clustered MSA carries 49 channels per row and position: a 23-way
 * one-hot of the cluster centre, two deletion scalars, the 23-way cluster
 * profile and the cluster's mean deletion. Twenty-two of those channels are
 * always zero, because the one-hot has a single entry, and at 508 rows of a
 * 1000-residue chain the dense form is 95 MiB that is rebuilt every recycle
 * and uploaded whole.
 *
 * Here the one-hot is a single code and the rest is stored as it is, giving 27
 * channels. The embedding shader reads the code as an index into the same
 * weight matrix, so it adds exactly the term the dense product would have
 * added, in the same order, and the embedding is unchanged to the last bit.
 */

/** Channels in the compact layout: code, two deletion scalars, profile, deletion mean. */
export const CLUSTERED_MSA_CHANNELS = 27;

/** Channels in AlphaFold's dense layout, which the reference tensors use. */
export const DENSE_CLUSTERED_MSA_CHANNELS = 49;

/** Code for a row the alignment masks out, whose one-hot is entirely zero. */
export const MSA_CODE_NONE = 23;

const PROFILE_CHANNELS = 23;

/** Packs AlphaFold's dense channels into the compact layout. */
export function compactClusteredMsaFeatures(dense: Float32Array, slots: number): Float32Array {
  if (dense.length !== slots * DENSE_CLUSTERED_MSA_CHANNELS) {
    throw new RangeError("dense clustered MSA features must have 49 channels per row and position");
  }
  const compact = new Float32Array(slots * CLUSTERED_MSA_CHANNELS);
  for (let slot = 0; slot < slots; slot += 1) {
    const source = slot * DENSE_CLUSTERED_MSA_CHANNELS; const target = slot * CLUSTERED_MSA_CHANNELS;
    let code = MSA_CODE_NONE;
    for (let category = 0; category < PROFILE_CHANNELS; category += 1) {
      if (dense[source + category] !== 0) {
        if (code !== MSA_CODE_NONE) throw new RangeError("dense clustered MSA one-hot has more than one entry");
        if (dense[source + category] !== 1) throw new RangeError("dense clustered MSA one-hot is not one");
        code = category;
      }
    }
    compact[target] = code;
    compact[target + 1] = dense[source + 23]!;
    compact[target + 2] = dense[source + 24]!;
    for (let category = 0; category < PROFILE_CHANNELS; category += 1) {
      compact[target + 3 + category] = dense[source + 25 + category]!;
    }
    compact[target + 26] = dense[source + 48]!;
  }
  return compact;
}

/** Expands the compact layout back to AlphaFold's dense channels. */
export function expandClusteredMsaFeatures(compact: Float32Array, slots: number): Float32Array {
  if (compact.length !== slots * CLUSTERED_MSA_CHANNELS) {
    throw new RangeError("compact clustered MSA features must have 27 channels per row and position");
  }
  const dense = new Float32Array(slots * DENSE_CLUSTERED_MSA_CHANNELS);
  for (let slot = 0; slot < slots; slot += 1) {
    const source = slot * CLUSTERED_MSA_CHANNELS; const target = slot * DENSE_CLUSTERED_MSA_CHANNELS;
    const code = compact[source]!;
    if (code < PROFILE_CHANNELS) dense[target + code] = 1;
    dense[target + 23] = compact[source + 1]!;
    dense[target + 24] = compact[source + 2]!;
    for (let category = 0; category < PROFILE_CHANNELS; category += 1) {
      dense[target + 25 + category] = compact[source + 3 + category]!;
    }
    dense[target + 48] = compact[source + 26]!;
  }
  return dense;
}
