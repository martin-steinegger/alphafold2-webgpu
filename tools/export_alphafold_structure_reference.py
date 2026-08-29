#!/usr/bin/env python3
"""Replay and export the official AF2 structure module from a trunk fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("params", type=Path)
    args = parser.parse_args()

    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphafold.model import config, folding, r3

    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.parent

    def tensor(name: str) -> np.ndarray:
        record = manifest["tensors"][name]
        return np.fromfile(directory / record["file"], dtype="<f4").reshape(record["shape"])

    source_params = np.load(args.params, allow_pickle=False)
    embedding_prefix = "alphafold/alphafold_iteration/evoformer/single_activations//"
    msa = tensor("stackRecycle3ExpectedMsa")[0]
    pair = tensor("stackRecycle3ExpectedPair")
    single = (
        msa @ source_params[f"{embedding_prefix}weights"]
        + source_params[f"{embedding_prefix}bias"]
    ).astype(np.float32)
    batch = {
        "aatype": tensor("feature_aatype_recycle3").astype(np.int32),
        "seq_mask": tensor("feature_seq_mask_recycle3"),
        "atom14_atom_exists": tensor("feature_atom14_atom_exists_recycle3"),
        "atom37_atom_exists": tensor("feature_atom37_atom_exists_recycle3"),
        "residx_atom37_to_atom14": tensor("feature_residx_atom37_to_atom14_recycle3").astype(np.int32),
    }

    model_config = config.model_config("model_1_ptm")
    structure_config = model_config.model.heads.structure_module
    global_config = model_config.model.global_config
    global_config.bfloat16 = False
    captures: dict[str, list[np.ndarray]] = {
        "ipa": [], "act_input": [], "affine_input": [], "act_output": [], "affine_output": [],
        "initial_act": [], "angles": [], "unnormalized_angles": [], "atom14": [],
    }
    original_ipa = folding.InvariantPointAttention.__call__
    original_fold = folding.FoldIteration.__call__

    def ipa_call(self, *call_args, **call_kwargs):
        output = original_ipa(self, *call_args, **call_kwargs)
        jax.debug.callback(lambda value: captures["ipa"].append(np.asarray(value, dtype=np.float32).copy()), output,
                           ordered=True)
        return output

    def fold_call(self, activations, *call_args, **call_kwargs):
        output_activations, output = original_fold(self, activations, *call_args, **call_kwargs)

        initial_act = call_kwargs["initial_act"]

        def receive(act_in, affine_in, act_out, affine_out, initial, angles, unnormalized, atom14):
            captures["act_input"].append(np.asarray(act_in, dtype=np.float32).copy())
            captures["affine_input"].append(np.asarray(affine_in, dtype=np.float32).copy())
            captures["act_output"].append(np.asarray(act_out, dtype=np.float32).copy())
            captures["affine_output"].append(np.asarray(affine_out, dtype=np.float32).copy())
            captures["initial_act"].append(np.asarray(initial, dtype=np.float32).copy())
            captures["angles"].append(np.asarray(angles, dtype=np.float32).copy())
            captures["unnormalized_angles"].append(np.asarray(unnormalized, dtype=np.float32).copy())
            captures["atom14"].append(np.asarray(atom14, dtype=np.float32).copy())

        jax.debug.callback(
            receive, activations["act"], activations["affine"],
            output_activations["act"], output_activations["affine"], initial_act,
            output["sc"]["angles_sin_cos"], output["sc"]["unnormalized_angles_sin_cos"],
            r3.vecs_to_tensor(output["sc"]["atom_pos"]), ordered=True,
        )
        return output_activations, output

    folding.InvariantPointAttention.__call__ = ipa_call
    folding.FoldIteration.__call__ = fold_call

    def forward(single_value, pair_value, batch_value):
        module = folding.StructureModule(
            structure_config, global_config, compute_loss=False, name="structure_module",
        )
        return module({"single": single_value, "pair": pair_value}, batch_value, is_training=False)

    transformed = hk.transform(forward)
    parameter_prefix = "alphafold/alphafold_iteration/structure_module/"
    parameters: dict[str, dict[str, Any]] = {}
    parameter_records: dict[str, dict[str, str]] = {}
    parameter_tensors: dict[str, np.ndarray] = {}
    parameter_index = 0
    for key in sorted(source_params.files):
        if not key.startswith(parameter_prefix):
            continue
        module, name = key.removeprefix(parameter_prefix).split("//")
        parameters.setdefault(f"structure_module/{module}", {})[name] = jnp.asarray(source_params[key])
        tensor_name = f"structure_haiku_{parameter_index:04d}"
        parameter_index += 1
        parameter_tensors[tensor_name] = np.asarray(source_params[key], dtype=np.float32)
        parameter_records.setdefault(module, {})[name] = tensor_name

    cpu = jax.devices("cpu")[0]
    parameters = jax.tree.map(lambda value: jax.device_put(value, cpu), parameters)
    inputs = jax.tree.map(lambda value: jax.device_put(jnp.asarray(value), cpu), (single, pair, batch))
    try:
        with jax.default_matmul_precision("highest"):
            output = transformed.apply(parameters, jax.random.PRNGKey(0), *inputs)
        jax.effects_barrier()
    finally:
        folding.InvariantPointAttention.__call__ = original_ipa
        folding.FoldIteration.__call__ = original_fold

    if not all(len(values) == 8 for values in captures.values()):
        raise RuntimeError({name: len(values) for name, values in captures.items()})
    recycle_outputs = []
    for recycle in range(4):
        recycle_msa = tensor(f"stackRecycle{recycle}ExpectedMsa")[0]
        recycle_pair = tensor(f"stackRecycle{recycle}ExpectedPair")
        recycle_single = (
            recycle_msa @ source_params[f"{embedding_prefix}weights"]
            + source_params[f"{embedding_prefix}bias"]
        ).astype(np.float32)
        recycle_batch = {
            "aatype": tensor(f"feature_aatype_recycle{recycle}").astype(np.int32),
            "seq_mask": tensor(f"feature_seq_mask_recycle{recycle}"),
            "atom14_atom_exists": tensor(f"feature_atom14_atom_exists_recycle{recycle}"),
            "atom37_atom_exists": tensor(f"feature_atom37_atom_exists_recycle{recycle}"),
            "residx_atom37_to_atom14": tensor(f"feature_residx_atom37_to_atom14_recycle{recycle}").astype(np.int32),
        }
        recycle_inputs = jax.tree.map(
            lambda value: jax.device_put(jnp.asarray(value), cpu),
            (recycle_single, recycle_pair, recycle_batch),
        )
        with jax.default_matmul_precision("highest"):
            recycle_output = transformed.apply(parameters, jax.random.PRNGKey(0), *recycle_inputs)
        recycle_outputs.append(recycle_output)
    jax.effects_barrier()
    exported = {
        "structureInputSingle": single,
        "structureInputPair": pair,
        "structureFinalAtomPositions": np.asarray(output["final_atom_positions"], dtype=np.float32),
        "structureFinalAtomMask": np.asarray(output["final_atom_mask"], dtype=np.float32),
        "structureFinalRepresentation": np.asarray(output["representations"]["structure_module"], dtype=np.float32),
        **parameter_tensors,
    }
    for recycle, recycle_output in enumerate(recycle_outputs):
        exported[f"structureRecycle{recycle}FinalAtomPositions"] = np.asarray(
            recycle_output["final_atom_positions"], dtype=np.float32,
        )
        exported[f"structureRecycle{recycle}FinalRepresentation"] = np.asarray(
            recycle_output["representations"]["structure_module"], dtype=np.float32,
        )
    for name, values in captures.items():
        exported[f"structureStage_{name}"] = np.stack(values)

    for name, value in exported.items():
        array = np.asarray(value, dtype="<f4", order="C")
        filename = f"{name}.f32.bin"
        (directory / filename).write_bytes(array.tobytes())
        manifest["tensors"][name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
    manifest["structureModule"] = {
        "implementation": "official AlphaFold folding.StructureModule",
        "dtype": "float32",
        "iterations": 8,
        "parameters": parameter_records,
        "referenceStages": {
            "ipa": "structureStage_ipa",
            "actInput": "structureStage_act_input",
            "affineInput": "structureStage_affine_input",
            "actOutput": "structureStage_act_output",
            "affineOutput": "structureStage_affine_output",
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported structure reference with {parameter_index} tensors to {args.manifest}")


if __name__ == "__main__":
    main()
