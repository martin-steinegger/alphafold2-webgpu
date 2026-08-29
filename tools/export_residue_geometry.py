#!/usr/bin/env python3
"""Export AlphaFold residue geometry lookup tables into a fixture manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from alphafold.common import residue_constants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    atom37_to_atom14 = np.zeros((21, 37), dtype=np.float32)
    atom37_mask = np.zeros((21, 37), dtype=np.float32)
    for restype_index, restype in enumerate(residue_constants.restypes + ["X"]):
        restype_name = residue_constants.restype_1to3.get(restype, "UNK")
        atom14_names = residue_constants.restype_name_to_atom14_names[restype_name]
        for atom14_index, atom_name in enumerate(atom14_names):
            if atom_name:
                atom37_index = residue_constants.atom_order[atom_name]
                atom37_to_atom14[restype_index, atom37_index] = atom14_index
                atom37_mask[restype_index, atom37_index] = 1
    values = {
        "geometryDefaultFrames": residue_constants.restype_rigid_group_default_frame,
        "geometryAtom14ToGroup": residue_constants.restype_atom14_to_rigid_group,
        "geometryAtom14Positions": residue_constants.restype_atom14_rigid_group_positions,
        "geometryAtom14Mask": residue_constants.restype_atom14_mask,
        "geometryAtom37ToAtom14": atom37_to_atom14,
        "geometryAtom37Mask": atom37_mask,
    }
    for name, value in values.items():
        array = np.asarray(value, dtype="<f4", order="C")
        filename = f"{name}.f32.bin"
        (args.manifest.parent / filename).write_bytes(array.tobytes())
        manifest["tensors"][name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
    manifest["residueGeometry"] = {"source": "alphafold.common.residue_constants", "tensors": list(values)}
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported residue geometry to {args.manifest}")


if __name__ == "__main__":
    main()
