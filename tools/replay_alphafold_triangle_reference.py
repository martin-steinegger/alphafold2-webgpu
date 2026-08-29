#!/usr/bin/env python3
"""Replay a captured bundle through the official AF2 Haiku operator on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
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
    directory = args.manifest.parent

    def tensor(name: str) -> np.ndarray:
        record = manifest["tensors"][name]
        return np.fromfile(directory / record["file"], dtype="<f4").reshape(record["shape"])

    model_config = config.model_config("model_3_ptm")
    model_config.model.global_config.bfloat16 = False
    triangle_config = (
        model_config.model.embeddings_and_evoformer.evoformer
        .triangle_multiplication_outgoing
    )
    triangle_config.fuse_projection_weights = False
    global_config = model_config.model.global_config

    def forward(z, mask):
        return modules.TriangleMultiplication(
            triangle_config,
            global_config,
            name="triangle_multiplication_outgoing",
        )(z, mask, is_training=False)

    transformed = hk.without_apply_rng(hk.transform(forward))
    root = "triangle_multiplication_outgoing"
    params = {
        f"{root}/layer_norm_input": {
            "scale": tensor("layerNormInWeight"),
            "offset": tensor("layerNormInBias"),
        },
        f"{root}/left_projection": {
            "weights": tensor("linearAPWeight").T,
            "bias": tensor("linearAPBias"),
        },
        f"{root}/left_gate": {
            "weights": tensor("linearAGWeight").T,
            "bias": tensor("linearAGBias"),
        },
        f"{root}/right_projection": {
            "weights": tensor("linearBPWeight").T,
            "bias": tensor("linearBPBias"),
        },
        f"{root}/right_gate": {
            "weights": tensor("linearBGWeight").T,
            "bias": tensor("linearBGBias"),
        },
        f"{root}/center_layer_norm": {
            "scale": tensor("layerNormOutWeight"),
            "offset": tensor("layerNormOutBias"),
        },
        f"{root}/output_projection": {
            "weights": tensor("linearZWeight").T,
            "bias": tensor("linearZBias"),
        },
        f"{root}/gating_linear": {
            "weights": tensor("linearGWeight").T,
            "bias": tensor("linearGBias"),
        },
    }
    cpu = jax.devices("cpu")[0]
    params = jax.tree.map(lambda value: jax.device_put(jnp.asarray(value), cpu), params)
    z = jax.device_put(jnp.asarray(tensor("z")), cpu)
    mask = jax.device_put(jnp.asarray(tensor("mask")), cpu)
    with jax.default_matmul_precision("highest"):
        expected = np.asarray(transformed.apply(params, z, mask), dtype="<f4")
    expected.tofile(directory / manifest["tensors"]["expected"]["file"])
    manifest["referenceExecution"] = {
        "implementation": "official AlphaFold Haiku TriangleMultiplication",
        "platform": "CPU",
        "dtype": "float32",
        "matmulPrecision": "highest",
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"replayed {args.manifest}: shape={expected.shape}")


if __name__ == "__main__":
    main()
