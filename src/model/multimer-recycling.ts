/** Official Multimer-v3 recycling convergence metric over pairwise CA distances. */
export function multimerRecycleDistanceRms(
  previousAtom37: Float32Array,
  nextAtom37: Float32Array,
  sequenceMask: Float32Array,
): number {
  const length = sequenceMask.length;
  if (length === 0 || previousAtom37.length !== length * 37 * 3 || nextAtom37.length !== length * 37 * 3) {
    throw new RangeError("multimer recycle atom37 tensors must match a non-empty sequence mask");
  }
  if (sequenceMask.some((value) => value !== 0 && value !== 1)) {
    throw new RangeError("multimer recycle sequence mask must be binary");
  }
  const distance = (atoms: Float32Array, left: number, right: number): number => {
    const leftBase = (left * 37 + 1) * 3; const rightBase = (right * 37 + 1) * 3;
    const dx = atoms[leftBase]! - atoms[rightBase]!;
    const dy = atoms[leftBase + 1]! - atoms[rightBase + 1]!;
    const dz = atoms[leftBase + 2]! - atoms[rightBase + 2]!;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  };
  let squaredDifference = 0; let weight = 0;
  for (let left = 0; left < length; left += 1) for (let right = 0; right < length; right += 1) {
    const pairWeight = sequenceMask[left]! * sequenceMask[right]!;
    if (pairWeight === 0) continue;
    const difference = distance(previousAtom37, left, right) - distance(nextAtom37, left, right);
    squaredDifference += pairWeight * difference * difference; weight += pairWeight;
  }
  return Math.sqrt(squaredDifference / Math.max(weight, 1) + 1e-8);
}
