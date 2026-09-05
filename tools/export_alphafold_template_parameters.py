#!/usr/bin/env python3
"""Add official model template parameters to an existing reference manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("params", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    parameters = np.load(args.params, allow_pickle=False)
    evoformer = "alphafold/alphafold_iteration/evoformer/"
    prefix = f"{evoformer}template_embedding/"
    # The torsion angles a template contributes to the MSA are embedded outside
    # the template_embedding module, by two siblings of it. model_1_ptm sets
    # embed_torsion_angles, so they are as required as the rest.
    siblings = ("template_single_embedding", "template_projection")

    def selected() -> list[tuple[str, str, str]]:
        chosen: list[tuple[str, str, str]] = []
        for key in sorted(parameters.files):
            if key.startswith(prefix):
                module, name = key.removeprefix(prefix).split("//")
                chosen.append((key, module, name))
        # Appended after them, so the tensors already in a published bundle keep
        # the names they have.
        for key in sorted(parameters.files):
            for sibling in siblings:
                if key.startswith(f"{evoformer}{sibling}//"):
                    chosen.append((key, sibling, key.split("//")[1]))
        return chosen

    records: dict[str, dict[str, str]] = {}
    index = 0
    for key, module, name in selected():
        tensor_name = f"template_haiku_{index:04d}"
        index += 1
        value = np.asarray(parameters[key], dtype="<f4", order="C")
        filename = f"{tensor_name}.f32.bin"
        (args.manifest.parent / filename).write_bytes(value.tobytes())
        manifest["tensors"][tensor_name] = {
            "file": filename,
            "shape": list(value.shape),
            "dtype": "float32",
        }
        records.setdefault(module, {})[name] = tensor_name
    for sibling in siblings:
        if sibling not in records:
            raise SystemExit(f"parameters contain no {sibling}; templates need it for the MSA rows")
    manifest["templateEmbedding"] = {
        "parameterFormat": "haiku",
        "parameters": records,
        "queryOnlySimplification": "all mock-template atom masks are zero; one template_mask entry is valid",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {index} template tensors into {args.manifest}")


if __name__ == "__main__":
    main()
