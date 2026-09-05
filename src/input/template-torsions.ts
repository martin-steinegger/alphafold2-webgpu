import { ATOM_TYPE_COUNT, RESTYPE_UNKNOWN } from "./residue-atoms.js";
import type { TemplateFeatures } from "./template-features.js";
import {
  CHI_ANGLES_MASK, CHI_ATOM_INDICES, CHI_COUNT, CHI_PI_PERIODIC, TORSION_COUNT,
} from "./torsion-tables.js";

/**
 * The seven torsion angles AlphaFold reads off a template's backbone and side
 * chains, in its order: pre-omega, phi, psi, chi 1 to 4.
 *
 * This is `all_atom.atom37_to_torsion_angles` with `placeholder_for_undefined`
 * on, which needs saying because inference runs it off.
 *
 * An angle whose four atoms are not all present is computed from a degenerate
 * frame, and what comes out is floating-point noise: perturbing a template's
 * coordinates by one part in a million swings the first residue's pre-omega
 * from (0.052, 0.999) to (0.408, 0.913), while a real angle moves in the
 * seventh digit. AlphaFold feeds that noise to the network — the sine and
 * cosine channels are concatenated raw, and only the seven mask channels say
 * which are meaningless — so no reimplementation can reproduce it, on any
 * platform, in any precision. AlphaFold's own answer to this is the placeholder
 * branch, which pins an undefined angle to (1, 0): unit length, like every
 * defined angle, and the same on every machine. That is the branch taken here.
 * It differs from an official run only where the mask is already zero.
 */
export interface TemplateTorsions {
  /** `[residues, 7, 2]`, sine then cosine. */
  readonly sinCos: Float32Array;
  /** The same angles with the ambiguously named chis turned by pi. */
  readonly alternativeSinCos: Float32Array;
  /** `[residues, 7]`, 1 where all four defining atoms were present. */
  readonly mask: Float32Array;
}

const EPSILON = 1e-8;

function normalize(x: number, y: number, z: number): readonly [number, number, number] {
  const length = Math.sqrt(x * x + y * y + z * z + EPSILON);
  return [x / length, y / length, z / length];
}

/**
 * The fourth atom's position in the frame the first three define.
 *
 * AlphaFold builds a frame from three of the four atoms — the second on the
 * negative x axis, the third at the origin, the first in the x-y plane — and
 * reads the torsion off the fourth atom's y and z in that frame. Returns the
 * sine and cosine, already normalised.
 */
function torsionSinCos(
  positions: Float32Array, a0: number, a1: number, a2: number, a3: number,
): readonly [number, number] {
  const e0 = normalize(
    positions[a2 * 3]! - positions[a1 * 3]!,
    positions[a2 * 3 + 1]! - positions[a1 * 3 + 1]!,
    positions[a2 * 3 + 2]! - positions[a1 * 3 + 2]!,
  );
  const rawX = positions[a0 * 3]! - positions[a2 * 3]!;
  const rawY = positions[a0 * 3 + 1]! - positions[a2 * 3 + 1]!;
  const rawZ = positions[a0 * 3 + 2]! - positions[a2 * 3 + 2]!;
  const projection = rawX * e0[0] + rawY * e0[1] + rawZ * e0[2];
  const e1 = normalize(
    rawX - projection * e0[0], rawY - projection * e0[1], rawZ - projection * e0[2],
  );
  const e2 = [
    e0[1] * e1[2] - e0[2] * e1[1],
    e0[2] * e1[0] - e0[0] * e1[2],
    e0[0] * e1[1] - e0[1] * e1[0],
  ] as const;
  const dx = positions[a3 * 3]! - positions[a2 * 3]!;
  const dy = positions[a3 * 3 + 1]! - positions[a2 * 3 + 1]!;
  const dz = positions[a3 * 3 + 2]! - positions[a2 * 3 + 2]!;
  const y = dx * e1[0] + dy * e1[1] + dz * e1[2];
  const z = dx * e2[0] + dy * e2[1] + dz * e2[2];
  const scale = Math.sqrt(z * z + y * y + EPSILON);
  return [z / scale, y / scale];
}

/**
 * Computes a template's torsion angles from its atom37 coordinates.
 *
 * Positions and masks are indexed by query position, so "the previous residue"
 * is the previous query position: where the template did not cover it, its
 * atoms are absent and the angles that need them are masked out, which is the
 * same answer AlphaFold reaches for a template with a gap there.
 */
export function templateTorsions(features: TemplateFeatures): TemplateTorsions {
  const { length, aatype, atomPositions, atomMask } = features;
  const sinCos = new Float32Array(length * TORSION_COUNT * 2);
  const alternativeSinCos = new Float32Array(length * TORSION_COUNT * 2);
  const mask = new Float32Array(length * TORSION_COUNT);

  // A working buffer holding this residue's 37 atoms after the previous
  // residue's, so a torsion spanning the two indexes one array.
  const window = new Float32Array(ATOM_TYPE_COUNT * 2 * 3);
  const previous = (atom: number): number => atom;
  const current = (atom: number): number => ATOM_TYPE_COUNT + atom;

  for (let residue = 0; residue < length; residue += 1) {
    window.fill(0);
    if (residue > 0) {
      window.set(
        atomPositions.subarray((residue - 1) * ATOM_TYPE_COUNT * 3, residue * ATOM_TYPE_COUNT * 3), 0,
      );
    }
    window.set(
      atomPositions.subarray(residue * ATOM_TYPE_COUNT * 3, (residue + 1) * ATOM_TYPE_COUNT * 3),
      ATOM_TYPE_COUNT * 3,
    );
    const previousMask = (atom: number): number =>
      residue === 0 ? 0 : atomMask[(residue - 1) * ATOM_TYPE_COUNT + atom]!;
    const currentMask = (atom: number): number => atomMask[residue * ATOM_TYPE_COUNT + atom]!;

    // Anything above the unknown residue, which is what a gap is, has no
    // torsions at all; AlphaFold clamps the type to 20 and the tables give it
    // no chis, and its atoms are absent so the backbone angles mask out too.
    const type = Math.min(aatype[residue]!, RESTYPE_UNKNOWN);
    const base = residue * TORSION_COUNT * 2;

    const angles: (readonly [number, number])[] = [
      // pre-omega: previous CA, previous C, this N, this CA.
      torsionSinCos(window, previous(1), previous(2), current(0), current(1)),
      // phi: previous C, this N, CA, C.
      torsionSinCos(window, previous(2), current(0), current(1), current(2)),
      // psi: this N, CA, C, O.
      torsionSinCos(window, current(0), current(1), current(2), current(4)),
    ];
    const masks = [
      previousMask(1) * previousMask(2) * currentMask(0) * currentMask(1),
      previousMask(2) * currentMask(0) * currentMask(1) * currentMask(2),
      currentMask(0) * currentMask(1) * currentMask(2) * currentMask(4),
    ];

    for (let chi = 0; chi < CHI_COUNT; chi += 1) {
      const table = (type * CHI_COUNT + chi) * 4;
      const atoms = [
        CHI_ATOM_INDICES[table]!, CHI_ATOM_INDICES[table + 1]!,
        CHI_ATOM_INDICES[table + 2]!, CHI_ATOM_INDICES[table + 3]!,
      ];
      angles.push(torsionSinCos(window, current(atoms[0]!), current(atoms[1]!), current(atoms[2]!), current(atoms[3]!)));
      const present = atoms.reduce((product, atom) => product * currentMask(atom), 1);
      masks.push(CHI_ANGLES_MASK[type * CHI_COUNT + chi]! * present);
    }

    for (let torsion = 0; torsion < TORSION_COUNT; torsion += 1) {
      const defined = masks[torsion]!;
      mask[residue * TORSION_COUNT + torsion] = defined;
      if (defined === 0) {
        sinCos[base + torsion * 2] = 1;
        alternativeSinCos[base + torsion * 2] = 1;
        continue;
      }
      // Psi is measured from the oxygen, so its angle comes out mirrored.
      const sign = torsion === 2 ? -1 : 1;
      const [sine, cosine] = angles[torsion]!;
      sinCos[base + torsion * 2] = sign * sine;
      sinCos[base + torsion * 2 + 1] = sign * cosine;
      // The chis whose terminal atoms are named ambiguously have a second,
      // equally correct angle half a turn away.
      const ambiguous = torsion < 3 ? 0 : CHI_PI_PERIODIC[type * CHI_COUNT + torsion - 3]!;
      const mirror = 1 - 2 * ambiguous;
      alternativeSinCos[base + torsion * 2] = mirror * sign * sine;
      alternativeSinCos[base + torsion * 2 + 1] = mirror * sign * cosine;
    }
  }
  return { sinCos, alternativeSinCos, mask };
}

/** Channels in `template_angle_feat`: 22 aatype, 14 torsions, 14 alternatives, 7 masks. */
export const TEMPLATE_ANGLE_CHANNELS = 22 + 14 + 14 + 7;

/**
 * The per-residue template feature the MSA row is embedded from.
 *
 * Returns the `[residues, 57]` feature and, separately, the psi mask, which is
 * what AlphaFold uses as the row's MSA mask: it depends only on one residue's
 * own backbone, so it says whether this position was covered at all.
 */
export function templateAngleFeatures(
  features: TemplateFeatures,
): { readonly features: Float32Array; readonly rowMask: Float32Array } {
  const torsions = templateTorsions(features);
  const { length, aatype } = features;
  const angles = new Float32Array(length * TEMPLATE_ANGLE_CHANNELS);
  const rowMask = new Float32Array(length);
  for (let residue = 0; residue < length; residue += 1) {
    const row = residue * TEMPLATE_ANGLE_CHANNELS;
    angles[row + aatype[residue]!] = 1;
    for (let value = 0; value < TORSION_COUNT * 2; value += 1) {
      angles[row + 22 + value] = torsions.sinCos[residue * TORSION_COUNT * 2 + value]!;
      angles[row + 36 + value] = torsions.alternativeSinCos[residue * TORSION_COUNT * 2 + value]!;
    }
    for (let torsion = 0; torsion < TORSION_COUNT; torsion += 1) {
      angles[row + 50 + torsion] = torsions.mask[residue * TORSION_COUNT + torsion]!;
    }
    rowMask[residue] = torsions.mask[residue * TORSION_COUNT + 2]!;
  }
  return { features: angles, rowMask };
}
