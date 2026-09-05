/**
 * The chi-angle tables AlphaFold's torsion features are computed from.
 *
 * Generated from `alphafold.model.all_atom.get_chi_atom_indices()` and
 * `residue_constants.chi_angles_mask` / `chi_pi_periodic`, in AlphaFold's
 * residue order with row 20 for the unknown residue. Hand-transcribing atom
 * indices is exactly the kind of thing that fails silently, so these are
 * copied out of the official implementation and checked by a test.
 */

/** Atom37 indices of the four atoms defining each chi angle, `[21, 4, 4]`. */
export const CHI_ATOM_INDICES = Int32Array.from([
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 11,  3,  5, 11, 23,  5, 11, 23, 32,
   0,  1,  3,  5,  1,  3,  5, 16,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 16,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 11,  3,  5, 11, 26,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 11,  3,  5, 11, 26,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 14,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  6,  1,  3,  6, 12,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 12,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 11,  3,  5, 11, 19,  5, 11, 19, 35,
   0,  1,  3,  5,  1,  3,  5, 18,  3,  5, 18, 19,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 12,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 11,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  8,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  9,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 12,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  5,  1,  3,  5, 12,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  1,  3,  6,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
   0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
]);

/** Which chi angles a residue has, `[21, 4]`. */
export const CHI_ANGLES_MASK = Float32Array.from([
  0, 0, 0, 0,
  1, 1, 1, 1,
  1, 1, 0, 0,
  1, 1, 0, 0,
  1, 0, 0, 0,
  1, 1, 1, 0,
  1, 1, 1, 0,
  0, 0, 0, 0,
  1, 1, 0, 0,
  1, 1, 0, 0,
  1, 1, 0, 0,
  1, 1, 1, 1,
  1, 1, 1, 0,
  1, 1, 0, 0,
  1, 1, 0, 0,
  1, 0, 0, 0,
  1, 0, 0, 0,
  1, 1, 0, 0,
  1, 1, 0, 0,
  1, 0, 0, 0,
  0, 0, 0, 0,
]);

/** Chi angles whose terminal atoms are named ambiguously, `[21, 4]`. */
export const CHI_PI_PERIODIC = Float32Array.from([
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 1, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
  0, 1, 0, 0,
  0, 0, 0, 0,
  0, 0, 0, 0,
]);

export const TORSION_COUNT = 7;
export const CHI_COUNT = 4;
