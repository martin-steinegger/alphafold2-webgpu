#!/usr/bin/env python3
"""Replay and stage-capture one official AlphaFold Evoformer block on CPU."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--extra", action="store_true", help="replay an extra-MSA block instead of the selected main block")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--recycle", type=int, default=0)
    args = parser.parse_args()

    try:
        import haiku as hk
        import jax
        import jax.numpy as jnp
        import numpy as np
        from alphafold.model import config, modules
    except ImportError as error:
        raise SystemExit("Use a Python environment containing AlphaFold, JAX, and Haiku.") from error

    manifest = json.loads(args.manifest.read_text())
    block_record = manifest.get("evoformerBlock")
    if not args.extra and (not block_record or not block_record.get("captured")):
        raise SystemExit("manifest does not contain a captured Evoformer block")
    directory = args.manifest.parent

    def tensor(name: str) -> np.ndarray:
        record = manifest["tensors"][name]
        return np.fromfile(directory / record["file"], dtype="<f4").reshape(record["shape"])

    model_config = config.model_config("model_1_ptm")
    model_config.model.global_config.bfloat16 = False
    evoformer_config = model_config.model.embeddings_and_evoformer.evoformer
    evoformer_config.triangle_multiplication_outgoing.fuse_projection_weights = False
    evoformer_config.triangle_multiplication_incoming.fuse_projection_weights = False
    global_config = model_config.model.global_config

    parameters: dict[str, dict[str, Any]] = {}
    if args.extra:
        stack_record = manifest["extraMsaStack"]
        if not 0 <= args.block < stack_record["blocks"]:
            raise SystemExit("extra-MSA block is out of range")
        for relative_path, values in stack_record["parameters"].items():
            parameters[f"evoformer_iteration/{relative_path}"] = {
                parameter_name: tensor(tensor_name)[args.block]
                for parameter_name, tensor_name in values.items()
            }
    else:
        for relative_path, values in block_record["parameters"].items():
            parameters[f"evoformer_iteration/{relative_path}"] = {
                parameter_name: tensor(tensor_name)
                for parameter_name, tensor_name in values.items()
            }

    captures: dict[str, dict[str, np.ndarray]] = {}
    originals: dict[type, Any] = {}
    capture_classes = (
        modules.MSARowAttentionWithPairBias,
        modules.MSAColumnAttention,
        modules.MSAColumnGlobalAttention,
        modules.Transition,
        modules.OuterProductMean,
        modules.TriangleMultiplication,
        modules.TriangleAttention,
    )

    def install_capture(cls: type) -> None:
        original = cls.__call__
        originals[cls] = original

        def wrapped(self, *call_args, **call_kwargs):
            output = original(self, *call_args, **call_kwargs)
            name = self.module_name.removeprefix("evoformer_iteration/")
            record = {
                "input": np.asarray(call_args[0], dtype=np.float32).copy(),
                "output": np.asarray(output, dtype=np.float32).copy(),
            }
            if "pair_act" in call_kwargs:
                record["pair"] = np.asarray(call_kwargs["pair_act"], dtype=np.float32).copy()
            captures[name] = record
            return output

        cls.__call__ = wrapped

    for cls in capture_classes:
        install_capture(cls)

    def forward(msa, pair, msa_mask, pair_mask):
        block = modules.EvoformerIteration(
            evoformer_config,
            global_config,
            is_extra_msa=args.extra,
            name="evoformer_iteration",
        )
        return block(
            {"msa": msa, "pair": pair},
            {"msa": msa_mask, "pair": pair_mask},
            is_training=False,
        )

    transformed = hk.transform(forward)
    cpu = jax.devices("cpu")[0]
    parameters = jax.tree.map(lambda value: jax.device_put(jnp.asarray(value), cpu), parameters)
    if args.extra:
        input_names = (
            f"extraStackRecycle{args.recycle}InputMsa", f"extraStackRecycle{args.recycle}InputPair",
            f"feature_extra_msa_mask_recycle{args.recycle}", "extraStackPairMask",
        )
    else:
        input_names = ("blockInputMsa", "blockInputPair", "blockMsaMask", "blockPairMask")
    inputs = [jax.device_put(jnp.asarray(tensor(name)), cpu) for name in input_names]
    try:
        with jax.default_matmul_precision("highest"):
            result = transformed.apply(parameters, jax.random.PRNGKey(0), *inputs)
        jax.effects_barrier()
    finally:
        for cls, original in originals.items():
            cls.__call__ = original

    output_prefix = "extraReplay" if args.extra else "blockExpected"
    stable_outputs = {
        f"{output_prefix}Msa": np.asarray(result["msa"], dtype=np.float32),
        f"{output_prefix}Pair": np.asarray(result["pair"], dtype=np.float32),
    }
    for name, array in stable_outputs.items():
        if name in manifest["tensors"]:
            array.astype("<f4", copy=False).tofile(directory / manifest["tensors"][name]["file"])
        else:
            filename = f"{name}.f32.bin"
            value = np.asarray(array, dtype="<f4", order="C")
            (directory / filename).write_bytes(value.tobytes())
            manifest["tensors"][name] = {"file": filename, "shape": list(value.shape), "dtype": "float32"}

    reference_stages: dict[str, dict[str, str]] = {}
    next_index = 0
    for module_name, arrays in captures.items():
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", module_name).strip("_")
        stage: dict[str, str] = {}
        for kind, array in arrays.items():
            tensor_name = f"stage_{next_index:03d}_{kind}"
            next_index += 1
            filename = f"{tensor_name}.f32.bin"
            value = np.asarray(array, dtype="<f4", order="C")
            (directory / filename).write_bytes(value.tobytes())
            manifest["tensors"][tensor_name] = {
                "file": filename,
                "shape": list(value.shape),
                "dtype": "float32",
                "stage": slug,
            }
            stage[kind] = tensor_name
        reference_stages[module_name] = stage

    target_record = manifest["extraMsaStack"] if args.extra else block_record
    target_record["referenceExecution"] = {
        "implementation": "official AlphaFold Haiku EvoformerIteration",
        "platform": "CPU",
        "dtype": "float32",
        "matmulPrecision": "highest",
    }
    target_record["referenceStages"] = reference_stages
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"replayed {args.manifest}: modules={len(reference_stages)}, "
        f"msa={next(iter(stable_outputs.values())).shape}, pair={list(stable_outputs.values())[1].shape}"
    )


if __name__ == "__main__":
    main()
