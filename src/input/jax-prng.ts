/** Bit-compatible subset of JAX's partitionable Threefry2x32 PRNG. */
export type JaxKey = readonly [number, number];

const PARITY = 0x1bd11bda;
const ROTATIONS = [[13, 15, 26, 6], [17, 29, 16, 24]] as const;

const add = (left: number, right: number): number => (left + right) >>> 0;
const rotateLeft = (value: number, bits: number): number =>
  ((value << bits) | (value >>> (32 - bits))) >>> 0;

export function threefry2x32(key: JaxKey, count0: number, count1: number): JaxKey {
  const keys = [key[0] >>> 0, key[1] >>> 0, (key[0] ^ key[1] ^ PARITY) >>> 0];
  let first = add(count0 >>> 0, keys[0]!);
  let second = add(count1 >>> 0, keys[1]!);
  for (let injection = 1; injection <= 5; injection += 1) {
    const rotations = ROTATIONS[(injection - 1) % 2]!;
    for (const rotation of rotations) {
      first = add(first, second);
      second = (rotateLeft(second, rotation) ^ first) >>> 0;
    }
    first = add(first, keys[injection % 3]!);
    second = add(add(second, keys[(injection + 1) % 3]!), injection);
  }
  return [first, second];
}
/** JAX partitionable split and fold_in both hash the key with the scalar index. */
export function jaxFoldIn(key: JaxKey, value: number): JaxKey {
  if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) {
    throw new RangeError("JAX fold-in value must be a uint32");
  }
  return threefry2x32(key, 0, value);
}

export function jaxSplit(key: JaxKey, count: number): readonly JaxKey[] {
  if (!Number.isSafeInteger(count) || count < 1) throw new RangeError("JAX split count must be positive");
  return Array.from({ length: count }, (_, index) => jaxFoldIn(key, index));
}

/** Scalar float32 uniform used by AlphaFold's padding_consistent_rng wrapper. */
export function jaxUniform(key: JaxKey): number {
  const bits = threefry2x32(key, 0, 0);
  return ((bits[0] ^ bits[1]) >>> 9) / 0x80_0000;
}

export function jaxPaddingConsistentUniform(key: JaxKey, indices: readonly number[]): number {
  let elementKey = key;
  for (const index of indices) elementKey = jaxFoldIn(elementKey, index);
  return jaxUniform(elementKey);
}

export interface MultimerMsaKeys {
  readonly nextRoot: JaxKey;
  readonly sample: JaxKey;
  readonly maskPosition: JaxKey;
  readonly maskGumbel: JaxKey;
}

/** Keys consumed by ColabFold's Python recycle loop, Haiku, and Multimer embedder. */
export function multimerMsaKeys(root: JaxKey): MultimerMsaKeys {
  const [nextRoot, apply] = jaxSplit(root, 2);
  const haiku = jaxSplit(apply!, 2)[1]!;
  const ensemble = jaxSplit(haiku, 2)[1]!;
  const embedding = jaxSplit(ensemble, 3);
  const mask = jaxSplit(embedding[2]!, 3);
  return {
    nextRoot: nextRoot!, sample: embedding[1]!, maskPosition: mask[1]!, maskGumbel: mask[2]!,
  };
}
