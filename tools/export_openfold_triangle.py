#!/usr/bin/env python3
"""Export an OpenFold TriangleMultiplicationOutgoing differential-test bundle.

OpenFold and PyTorch are deliberately optional project dependencies. Run this
script inside an environment where both are already installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--cz", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--state-dict", type=Path)
    parser.add_argument(
        "--prefix",
        default="",
        help="prefix of the triangle module inside a larger OpenFold state dict",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import torch
        # OpenFold's primitives module imports its optional fused attention CUDA
        # extension eagerly, even though triangle multiplication does not use it.
        # A placeholder lets the unrelated pure-PyTorch operator load on CPU.
        if importlib.util.find_spec("attn_core_inplace_cuda") is None:
            sys.modules["attn_core_inplace_cuda"] = types.ModuleType(
                "attn_core_inplace_cuda"
            )
        from openfold.model.triangular_multiplicative_update import (
            TriangleMultiplicationOutgoing,
        )
    except ImportError as error:
        raise SystemExit(
            "This exporter requires PyTorch and OpenFold in the active Python environment."
        ) from error

    if min(args.length, args.cz, args.hidden) <= 0:
        raise SystemExit("length, cz, and hidden must be positive")
    torch.manual_seed(args.seed)
    module = TriangleMultiplicationOutgoing(args.cz, args.hidden).eval()

    source = f"OpenFold deterministic test weights, seed={args.seed}"
    if args.state_dict is not None:
        checkpoint: Any = torch.load(args.state_dict, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise SystemExit("checkpoint must contain a PyTorch state-dict mapping")
        prefix = args.prefix
        selected = {
            key[len(prefix) :]: value
            for key, value in checkpoint.items()
            if key.startswith(prefix)
        }
        missing, unexpected = module.load_state_dict(selected, strict=False)
        if missing or unexpected:
            raise SystemExit(f"state-dict mismatch: missing={missing}, unexpected={unexpected}")
        source = f"OpenFold state dict {args.state_dict.name}, prefix={prefix!r}"
    else:
        # OpenFold's production "final" initializer makes linear_z exactly zero,
        # which would turn a random fixture into a trivial all-zero test. Use
        # deterministic nonzero test parameters while retaining OpenFold itself
        # as the executing oracle.
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if "layer_norm" in name and name.endswith("weight"):
                    parameter.uniform_(0.8, 1.2)
                elif name.endswith("bias"):
                    parameter.uniform_(-0.1, 0.1)
                else:
                    parameter.uniform_(-0.15, 0.15)

    z = torch.randn(1, args.length, args.length, args.cz, dtype=torch.float32)
    mask = (torch.rand(1, args.length, args.length) > 0.1).to(torch.float32)
    with torch.no_grad():
        expected = module(z, mask)

    tensors = {
        "z": z.squeeze(0),
        "mask": mask.squeeze(0),
        "expected": expected.squeeze(0),
        "layerNormInWeight": module.layer_norm_in.weight,
        "layerNormInBias": module.layer_norm_in.bias,
        "linearAPWeight": module.linear_a_p.weight,
        "linearAPBias": module.linear_a_p.bias,
        "linearAGWeight": module.linear_a_g.weight,
        "linearAGBias": module.linear_a_g.bias,
        "linearBPWeight": module.linear_b_p.weight,
        "linearBPBias": module.linear_b_p.bias,
        "linearBGWeight": module.linear_b_g.weight,
        "linearBGBias": module.linear_b_g.bias,
        "layerNormOutWeight": module.layer_norm_out.weight,
        "layerNormOutBias": module.layer_norm_out.bias,
        "linearZWeight": module.linear_z.weight,
        "linearZBias": module.linear_z.bias,
        "linearGWeight": module.linear_g.weight,
        "linearGBias": module.linear_g.bias,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        array = tensor.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
        filename = f"{name}.f32.bin"
        (args.output / filename).write_bytes(array.tobytes())
        records[name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}

    manifest = {
        "formatVersion": 1,
        "operator": "TriangleMultiplicationOutgoing",
        "source": source,
        "shape": {"length": args.length, "cZ": args.cz, "cHidden": args.hidden},
        "epsilon": 1e-5,
        "tensors": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
