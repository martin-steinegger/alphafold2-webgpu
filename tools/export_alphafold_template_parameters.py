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
    prefix = "alphafold/alphafold_iteration/evoformer/template_embedding/"
    records: dict[str, dict[str, str]] = {}
    index = 0
    for key in sorted(parameters.files):
        if not key.startswith(prefix):
            continue
        module, name = key.removeprefix(prefix).split("//")
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
    manifest["templateEmbedding"] = {
        "parameterFormat": "haiku",
        "parameters": records,
        "queryOnlySimplification": "all mock-template atom masks are zero; one template_mask entry is valid",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported {index} template tensors into {args.manifest}")


if __name__ == "__main__":
    main()
