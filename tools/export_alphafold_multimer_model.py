#!/usr/bin/env python3
"""Export one official AlphaFold-Multimer-v3 checkpoint for AFWebGPU.

Run this inside a ColabFold environment. The loader's ``use_fuse=False`` mode
performs ColabFold's deterministic fused-triangle split, producing the exact
parameter layout consumed by AFWebGPU's independently tested kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from types import GeneratorType
from typing import Any

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "colabfold")
    parser.add_argument("--model-number", type=int, choices=(1,), default=1)
    return parser.parse_args()


def geometry_tables(residue_constants: Any) -> dict[str, np.ndarray]:
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
    return {
        "geometryDefaultFrames": residue_constants.restype_rigid_group_default_frame,
        "geometryAtom14ToGroup": residue_constants.restype_atom14_to_rigid_group,
        "geometryAtom14Positions": residue_constants.restype_atom14_rigid_group_positions,
        "geometryAtom14Mask": residue_constants.restype_atom14_mask,
        "geometryAtom37ToAtom14": atom37_to_atom14,
        "geometryAtom37Mask": atom37_mask,
        "confidencePaeBreaks": np.linspace(0.0, 31.0, 63, dtype=np.float32),
    }


def main() -> None:
    args = arguments()
    # ColabFold releases that still call np.sum(generator) need this NumPy 2
    # compatibility shim while loading configuration. It does not touch tensors.
    original_sum = np.sum

    def compatible_sum(value: Any, *sum_args: Any, **sum_kwargs: Any) -> Any:
        if isinstance(value, GeneratorType):
            value = list(value)
        return original_sum(value, *sum_args, **sum_kwargs)

    np.sum = compatible_sum  # type: ignore[assignment]
    try:
        from alphafold.common import residue_constants
        from colabfold.alphafold.models import load_models_and_params
    except ImportError as error:
        raise SystemExit("Use a Python environment containing ColabFold, AlphaFold, JAX, and Haiku.") from error

    model_name, _, params = load_models_and_params(
        num_models=1,
        use_templates=False,
        num_recycles=0,
        recycle_early_stop_tolerance=-1,
        num_ensemble=1,
        model_order=[args.model_number],
        model_type="alphafold2_multimer_v3",
        data_dir=args.data_dir,
        max_seq=1,
        max_extra_seq=1,
        use_fuse=False,
        use_bfloat16=False,
        use_dropout=False,
        save_all=False,
    )[0]

    args.output.mkdir(parents=True, exist_ok=True)
    license_source = args.data_dir / "params" / "LICENSE"
    if not license_source.is_file():
        raise RuntimeError(f"official parameter license is missing: {license_source}")
    shutil.copyfile(license_source, args.output / "WEIGHTS_LICENSE.txt")
    tensors: dict[str, dict[str, Any]] = {}
    tensor_index = 0

    def store(value: Any, prefix: str) -> str:
        nonlocal tensor_index
        name = f"{prefix}_{tensor_index:04d}"
        tensor_index += 1
        array = np.asarray(value, dtype="<f4", order="C")
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"non-finite parameter {name}")
        filename = f"{name}.f32.bin"
        (args.output / filename).write_bytes(array.tobytes())
        tensors[name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
        return name

    def parameter_map(prefix: str, *, exclude: tuple[str, ...] = ()) -> dict[str, dict[str, str]]:
        records: dict[str, dict[str, str]] = {}
        for path in sorted(params):
            if not path.startswith(prefix):
                continue
            relative = path.removeprefix(prefix)
            if relative.startswith(exclude):
                continue
            records[relative] = {
                name: store(value, "parameter") for name, value in sorted(params[path].items())
            }
        return records

    evoformer_prefix = "alphafold/alphafold_iteration/evoformer/"
    main_prefix = f"{evoformer_prefix}evoformer_iteration/"
    extra_prefix = f"{evoformer_prefix}extra_msa_stack/"
    template_prefix = f"{evoformer_prefix}template_embedding/"
    embedding_names = {
        "extra_msa_activations", "left_single", "preprocess_1d", "preprocess_msa",
        "prev_msa_first_row_norm", "prev_pair_norm", "prev_pos_linear", "right_single",
        "single_activations", "~_relative_encoding/position_activations",
    }
    embedding: dict[str, dict[str, str]] = {}
    for path in sorted(params):
        if not path.startswith(evoformer_prefix):
            continue
        relative = path.removeprefix(evoformer_prefix)
        if relative not in embedding_names:
            continue
        embedding[relative] = {
            name: store(value, "embedding") for name, value in sorted(params[path].items())
        }

    structure_prefix = "alphafold/alphafold_iteration/structure_module/"
    lddt_prefix = "alphafold/alphafold_iteration/predicted_lddt_head/"
    pae_prefix = "alphafold/alphafold_iteration/predicted_aligned_error_head/"
    geometry_names: list[str] = []
    for name, value in geometry_tables(residue_constants).items():
        array = np.asarray(value, dtype="<f4", order="C")
        filename = f"{name}.f32.bin"
        (args.output / filename).write_bytes(array.tobytes())
        tensors[name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
        geometry_names.append(name)

    template_parameters = parameter_map(template_prefix)
    for name in ("template_single_embedding", "template_projection"):
        path = f"{evoformer_prefix}{name}"
        template_parameters[name] = {
            parameter_name: store(value, "parameter")
            for parameter_name, value in sorted(params[path].items())
        }

    manifest = {
        "formatVersion": 1,
        "source": "official AlphaFold-Multimer-v3 parameters loaded by ColabFold with use_fuse=False",
        "model": {"name": model_name, "type": "alphafold2_multimer_v3", "number": args.model_number},
        "weightsLicense": {
            "spdx": "CC-BY-4.0",
            "file": "WEIGHTS_LICENSE.txt",
            "source": "AlphaFold model parameters",
            "url": "https://github.com/google-deepmind/alphafold",
            "modified": True,
            "modifications": ["repacked into versioned browser shards"],
        },
        "bundle": {
            "purpose": "browser-inference",
            "model": f"model_{args.model_number}_multimer_v3",
            "encoding": "float32-le",
        },
        "evoformerStack": {
            "blocks": 48, "parameterFormat": "stacked-haiku", "parameters": parameter_map(main_prefix),
        },
        "extraMsaStack": {
            "blocks": 4, "parameterFormat": "stacked-haiku", "parameters": parameter_map(extra_prefix),
        },
        "embedding": {"parameterFormat": "haiku", "parameters": embedding},
        "multimerTemplate": {
            "implementation": "ColabFold mock-template pair and torsion-row embedding",
            "templates": 4,
            "parameters": template_parameters,
        },
        "structureModule": {
            "implementation": "official AlphaFold folding_multimer.StructureModule",
            "dtype": "float32", "iterations": 8, "positionScale": 20,
            "parameters": parameter_map(structure_prefix),
        },
        "confidenceHeads": {"parameters": {
            "predictedLddt": parameter_map(lddt_prefix),
            "predictedAlignedError": parameter_map(pae_prefix),
        }},
        "residueGeometry": {"source": "alphafold.common.residue_constants", "tensors": geometry_names},
        "tensors": tensors,
    }
    shard_count = 8
    shards: list[dict[str, Any]] = [
        {"index": index, "bytes": 0, "names": []} for index in range(shard_count)
    ]
    for name, record in sorted(
        tensors.items(),
        key=lambda item: (-int(np.prod(item[1]["shape"], dtype=np.int64)), item[0]),
    ):
        shard = min(shards, key=lambda value: (value["bytes"], value["index"]))
        size = int(np.prod(record["shape"], dtype=np.int64)) * 4
        shard["names"].append(name)
        shard["bytes"] += size
    files: list[dict[str, Any]] = []
    source_files: list[Path] = []
    for shard in shards:
        filename = f"weights-{shard['index']:02d}.v2.f32.bin"
        position = 0
        with (args.output / filename).open("wb") as output_file:
            for name in shard["names"]:
                record = tensors[name]
                source_path = args.output / record["file"]
                value = source_path.read_bytes()
                output_file.write(value)
                source_files.append(source_path)
                tensors[name] = {**record, "file": filename, "byteOffset": position}
                position += len(value)
        digest = hashlib.sha256((args.output / filename).read_bytes()).hexdigest()
        files.append({"file": filename, "bytes": position, "sha256": digest})
    for source_path in source_files:
        source_path.unlink()
    bytes_written = sum(
        int(np.prod(record["shape"], dtype=np.int64)) * 4 for record in tensors.values()
    )
    manifest["bundle"].update({
        "version": 1,
        "id": f"model_{args.model_number}_multimer_v3-f32-v2",
        "tensors": len(tensors),
        "bytes": bytes_written,
        "shards": shard_count,
        "files": files,
    })
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"Exported {model_name}: {len(tensors)} tensors, {bytes_written / 2**20:.1f} MiB, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
