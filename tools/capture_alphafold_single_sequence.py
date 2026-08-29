#!/usr/bin/env python3
"""Capture a real single-sequence AF2 triangle-outgoing invocation.

This script must run in a ColabFold/AlphaFold JAX environment. It constructs a
query-only MSA, executes an official monomer model, and intercepts the actual
input and output of one main-Evoformer TriangleMultiplicationOutgoing block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--sequence")
    input_group.add_argument("--a3m", type=Path)
    parser.add_argument("--name", default="query")
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--max-seq", type=int)
    parser.add_argument("--max-extra-seq", type=int)
    parser.add_argument(
        "--capture-evoformer",
        action="store_true",
        help="also export the selected main Evoformer block inputs, outputs, and parameters",
    )
    parser.add_argument(
        "--capture-evoformer-stack",
        action="store_true",
        help="export one complete main-Evoformer stack input/output and all 48 blocks of parameters",
    )
    parser.add_argument(
        "--capture-recycle",
        type=int,
        help="recycle to capture; defaults to the final configured recycle",
    )
    parser.add_argument("--model-number", choices=range(1, 6), type=int, default=3)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "colabfold",
        help="directory containing params/params_model_N_ptm.npz",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.block < 48:
        raise SystemExit("block must be between 0 and 47")
    capture_recycle = args.recycles if args.capture_recycle is None else args.capture_recycle
    if args.recycles < 0 or not 0 <= capture_recycle <= args.recycles:
        raise SystemExit("recycles and capture-recycle must define a valid non-negative iteration")
    target_invocation = capture_recycle * 48 + args.block

    try:
        import haiku as hk
        import jax
        import numpy as np
        from alphafold.data import parsers
        from alphafold.model import modules
        from colabfold.alphafold.models import load_models_and_params
        from colabfold.batch import build_monomer_feature, mk_mock_template
    except ImportError as error:
        raise SystemExit(
            "Use a Python environment containing ColabFold, AlphaFold, JAX, and Haiku."
        ) from error

    if args.a3m is None:
        sequence = args.sequence.strip().upper()
        a3m_text = f">{args.name}\n{sequence}\n"
        input_kind = "query-only"
    else:
        # ColabFold serialized A3M files may begin with a '#length cardinality'
        # metadata row. AlphaFold's low-level FASTA parser does not consume it.
        a3m_text = "\n".join(
            line for line in args.a3m.read_text().splitlines()
            if not line.lstrip().startswith("#")
        ) + "\n"
        parsed_a3m = parsers.parse_a3m(a3m_text)
        if not parsed_a3m.sequences:
            raise SystemExit("A3M contains no sequences")
        sequence = parsed_a3m.sequences[0].replace("-", "").upper()
        input_kind = "a3m"
    parsed_a3m = parsers.parse_a3m(a3m_text)
    msa_depth = len(parsed_a3m.sequences)
    default_max_seq = 512 if args.a3m is not None else (5 if args.model_number in (1, 2) else 1)
    max_seq = default_max_seq if args.max_seq is None else args.max_seq
    max_extra_seq = (1024 if args.a3m is not None else 1) if args.max_extra_seq is None else args.max_extra_seq
    if max_seq <= 0 or max_extra_seq <= 0:
        raise SystemExit("max-seq and max-extra-seq must be positive")
    allowed = set("ACDEFGHIKLMNPQRSTVWYX")
    if not sequence or set(sequence) - allowed:
        raise SystemExit("query must contain only the 20 standard amino acids or X")

    captured: dict[str, np.ndarray] = {}
    captured_evoformer: dict[str, np.ndarray] = {}
    captured_evoformer_stack: dict[str, np.ndarray] = {}
    captured_extra_stack: dict[str, np.ndarray] = {}
    captured_features: dict[str, np.ndarray] = {}
    recycle_metrics: list[dict[str, float | int]] = []
    invocation = 0
    evoformer_invocation = 0
    extra_evoformer_invocation = 0
    original_call = modules.TriangleMultiplication.__call__
    original_evoformer_call = modules.EvoformerIteration.__call__
    original_template_call = modules.TemplateEmbedding.__call__
    template_invocation = 0

    def intercepted_call(self, left_act, left_mask, is_training=True):
        nonlocal invocation
        output = original_call(self, left_act, left_mask, is_training)
        module_name = self.module_name
        is_main_outgoing = (
            module_name.endswith("evoformer_iteration/triangle_multiplication_outgoing")
            and "extra_msa" not in module_name
            and "template" not in module_name
        )
        if is_main_outgoing:
            def receive(z_value, mask_value, output_value):
                nonlocal invocation
                if invocation == target_invocation:
                    captured["z"] = np.asarray(z_value, dtype=np.float32).copy()
                    captured["mask"] = np.asarray(mask_value, dtype=np.float32).copy()
                    captured["expected"] = np.asarray(output_value, dtype=np.float32).copy()
                invocation += 1

            jax.debug.callback(receive, left_act, left_mask, output, ordered=True)
        return output

    modules.TriangleMultiplication.__call__ = intercepted_call
    if args.capture_evoformer or args.capture_evoformer_stack:
        if args.capture_evoformer_stack:
            def intercepted_template_call(self, *call_args, **call_kwargs):
                nonlocal template_invocation
                output = original_template_call(self, *call_args, **call_kwargs)

                def receive_template(value):
                    nonlocal template_invocation
                    captured_features[f"templatePairUpdateRecycle{template_invocation}"] = np.asarray(
                        value, dtype=np.float32
                    ).copy()
                    template_invocation += 1

                jax.debug.callback(receive_template, output, ordered=True)
                return output

            modules.TemplateEmbedding.__call__ = intercepted_template_call

        def intercepted_evoformer_call(
            self,
            activations,
            masks,
            is_training=True,
            safe_key=None,
        ):
            nonlocal evoformer_invocation, extra_evoformer_invocation
            output = original_evoformer_call(
                self,
                activations,
                masks,
                is_training,
                safe_key,
            )
            if not self.is_extra_msa:
                def receive(msa_in, pair_in, msa_mask, pair_mask, msa_out, pair_out):
                    nonlocal evoformer_invocation
                    if evoformer_invocation == target_invocation:
                        captured_evoformer["blockInputMsa"] = np.asarray(msa_in, dtype=np.float32).copy()
                        captured_evoformer["blockInputPair"] = np.asarray(pair_in, dtype=np.float32).copy()
                        captured_evoformer["blockMsaMask"] = np.asarray(msa_mask, dtype=np.float32).copy()
                        captured_evoformer["blockPairMask"] = np.asarray(pair_mask, dtype=np.float32).copy()
                        captured_evoformer["blockExpectedMsa"] = np.asarray(msa_out, dtype=np.float32).copy()
                        captured_evoformer["blockExpectedPair"] = np.asarray(pair_out, dtype=np.float32).copy()
                    stack_first = capture_recycle * 48
                    stack_last = stack_first + 47
                    recycle = evoformer_invocation // 48
                    block_in_stack = evoformer_invocation % 48
                    if args.capture_evoformer_stack and block_in_stack == 0:
                        captured_evoformer_stack[f"stackRecycle{recycle}InputMsa"] = np.asarray(msa_in, dtype=np.float32).copy()
                        captured_evoformer_stack[f"stackRecycle{recycle}InputPair"] = np.asarray(pair_in, dtype=np.float32).copy()
                    if args.capture_evoformer_stack and block_in_stack == 47:
                        captured_evoformer_stack[f"stackRecycle{recycle}ExpectedMsa"] = np.asarray(msa_out, dtype=np.float32).copy()
                        captured_evoformer_stack[f"stackRecycle{recycle}ExpectedPair"] = np.asarray(pair_out, dtype=np.float32).copy()
                    if args.capture_evoformer_stack and evoformer_invocation == stack_first:
                        captured_evoformer_stack["stackInputMsa"] = np.asarray(msa_in, dtype=np.float32).copy()
                        captured_evoformer_stack["stackInputPair"] = np.asarray(pair_in, dtype=np.float32).copy()
                        captured_evoformer_stack["stackMsaMask"] = np.asarray(msa_mask, dtype=np.float32).copy()
                        captured_evoformer_stack["stackPairMask"] = np.asarray(pair_mask, dtype=np.float32).copy()
                    if args.capture_evoformer_stack and evoformer_invocation == stack_last:
                        captured_evoformer_stack["stackExpectedMsa"] = np.asarray(msa_out, dtype=np.float32).copy()
                        captured_evoformer_stack["stackExpectedPair"] = np.asarray(pair_out, dtype=np.float32).copy()
                    evoformer_invocation += 1

                jax.debug.callback(
                    receive,
                    activations["msa"],
                    activations["pair"],
                    masks["msa"],
                    masks["pair"],
                    output["msa"],
                    output["pair"],
                    ordered=True,
                )
            elif args.capture_evoformer_stack:
                def receive_extra(msa_in, pair_in, msa_mask, pair_mask, msa_out, pair_out):
                    nonlocal extra_evoformer_invocation
                    stack_first = capture_recycle * 4
                    stack_last = stack_first + 3
                    recycle = extra_evoformer_invocation // 4
                    block_in_stack = extra_evoformer_invocation % 4
                    if block_in_stack == 0:
                        captured_extra_stack[f"extraStackRecycle{recycle}InputMsa"] = np.asarray(msa_in, dtype=np.float32).copy()
                        captured_extra_stack[f"extraStackRecycle{recycle}InputPair"] = np.asarray(pair_in, dtype=np.float32).copy()
                    if block_in_stack == 3:
                        captured_extra_stack[f"extraStackRecycle{recycle}ExpectedMsa"] = np.asarray(msa_out, dtype=np.float32).copy()
                        captured_extra_stack[f"extraStackRecycle{recycle}ExpectedPair"] = np.asarray(pair_out, dtype=np.float32).copy()
                    if extra_evoformer_invocation == stack_first:
                        captured_extra_stack["extraStackInputMsa"] = np.asarray(msa_in, dtype=np.float32).copy()
                        captured_extra_stack["extraStackInputPair"] = np.asarray(pair_in, dtype=np.float32).copy()
                        captured_extra_stack["extraStackMsaMask"] = np.asarray(msa_mask, dtype=np.float32).copy()
                        captured_extra_stack["extraStackPairMask"] = np.asarray(pair_mask, dtype=np.float32).copy()
                    if extra_evoformer_invocation == stack_last:
                        captured_extra_stack["extraStackExpectedMsa"] = np.asarray(msa_out, dtype=np.float32).copy()
                        captured_extra_stack["extraStackExpectedPair"] = np.asarray(pair_out, dtype=np.float32).copy()
                    extra_evoformer_invocation += 1

                jax.debug.callback(
                    receive_extra,
                    activations["msa"], activations["pair"], masks["msa"], masks["pair"],
                    output["msa"], output["pair"], ordered=True,
                )
            return output

        modules.EvoformerIteration.__call__ = intercepted_evoformer_call
    try:
        model_name, runner, params = load_models_and_params(
            num_models=1,
            use_templates=args.model_number in (1, 2),
            num_recycles=args.recycles,
            recycle_early_stop_tolerance=-1,
            num_ensemble=1,
            model_order=[args.model_number],
            model_type="alphafold2_ptm",
            data_dir=args.data_dir,
            max_seq=max_seq,
            max_extra_seq=max_extra_seq,
            use_fuse=False,
            use_bfloat16=False,
            use_dropout=False,
            save_all=False,
        )[0]
        raw_features = build_monomer_feature(
            sequence,
            a3m_text,
            mk_mock_template(sequence),
        )
        features = runner.process_features(raw_features, random_seed=0)
        if args.capture_evoformer_stack:
            feature_names = (
                "aatype", "seq_mask", "residue_index", "target_feat", "msa_feat", "msa_mask",
                "extra_msa", "extra_has_deletion", "extra_deletion_value", "extra_msa_mask",
                "template_aatype", "template_all_atom_positions", "template_all_atom_masks", "template_mask",
                "atom14_atom_exists", "atom37_atom_exists", "residx_atom37_to_atom14",
            )
            for feature_name in feature_names:
                value = np.asarray(features[feature_name])
                captured_features[f"feature_{feature_name}"] = np.asarray(
                    value[capture_recycle], dtype=np.float32
                ).copy()
                if value.shape[0] == args.recycles + 1:
                    for recycle in range(args.recycles + 1):
                        captured_features[f"feature_{feature_name}_recycle{recycle}"] = np.asarray(
                            value[recycle], dtype=np.float32
                        ).copy()
        runner.params = params
        def record_recycle(result, recycle):
            metrics: dict[str, float | int] = {
                "recycle": int(recycle),
                "meanPlddt": float(np.mean(result["plddt"])),
                "rankingConfidence": float(result["ranking_confidence"]),
            }
            if "ptm" in result:
                metrics["ptm"] = float(result["ptm"])
            if "tol" in result:
                metrics["tol"] = float(result["tol"])
            recycle_metrics.append(metrics)

        prediction, recycle_iteration = runner.predict(
            features,
            random_seed=0,
            callback=record_recycle,
        )
        jax.effects_barrier()
    finally:
        modules.TriangleMultiplication.__call__ = original_call
        modules.EvoformerIteration.__call__ = original_evoformer_call
        modules.TemplateEmbedding.__call__ = original_template_call

    if set(captured) != {"z", "mask", "expected"}:
        raise RuntimeError(
            f"failed to capture invocation {target_invocation}; "
            f"observed {invocation} main outgoing invocations"
        )
    if args.capture_evoformer and len(captured_evoformer) != 6:
        raise RuntimeError(
            f"failed to capture Evoformer invocation {target_invocation}; "
            f"observed {evoformer_invocation} main block invocations"
        )
    if args.capture_evoformer_stack and len(captured_evoformer_stack) != 6 + 4 * (args.recycles + 1):
        raise RuntimeError(
            f"failed to capture main Evoformer stack at recycle {capture_recycle}; "
            f"observed {evoformer_invocation} main block invocations"
        )
    if args.capture_evoformer_stack and len(captured_extra_stack) != 6 + 4 * (args.recycles + 1):
        raise RuntimeError(
            f"failed to capture extra-MSA stack at recycle {capture_recycle}; "
            f"observed {extra_evoformer_invocation} extra block invocations"
        )
    if args.capture_evoformer_stack and template_invocation != args.recycles + 1:
        raise RuntimeError(
            f"expected one template embedding per recycle; observed {template_invocation}"
        )

    prefix = (
        "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
        "triangle_multiplication_outgoing"
    )

    def block_params(component: str) -> dict[str, np.ndarray]:
        values = params[f"{prefix}/{component}"]
        return {name: np.asarray(value[args.block], dtype=np.float32) for name, value in values.items()}

    layer_norm_in = block_params("layer_norm_input")
    left_projection = block_params("left_projection")
    left_gate = block_params("left_gate")
    right_projection = block_params("right_projection")
    right_gate = block_params("right_gate")
    layer_norm_out = block_params("center_layer_norm")
    output_projection = block_params("output_projection")
    output_gate = block_params("gating_linear")
    tensors = {
        **captured,
        **captured_evoformer,
        **captured_evoformer_stack,
        **captured_extra_stack,
        **captured_features,
        "layerNormInWeight": layer_norm_in["scale"],
        "layerNormInBias": layer_norm_in["offset"],
        "linearAPWeight": left_projection["weights"].T,
        "linearAPBias": left_projection["bias"],
        "linearAGWeight": left_gate["weights"].T,
        "linearAGBias": left_gate["bias"],
        "linearBPWeight": right_projection["weights"].T,
        "linearBPBias": right_projection["bias"],
        "linearBGWeight": right_gate["weights"].T,
        "linearBGBias": right_gate["bias"],
        "layerNormOutWeight": layer_norm_out["scale"],
        "layerNormOutBias": layer_norm_out["offset"],
        "linearZWeight": output_projection["weights"].T,
        "linearZBias": output_projection["bias"],
        "linearGWeight": output_gate["weights"].T,
        "linearGBias": output_gate["bias"],
    }

    haiku_parameters: dict[str, dict[str, str]] = {}
    haiku_stack_parameters: dict[str, dict[str, str]] = {}
    haiku_extra_stack_parameters: dict[str, dict[str, str]] = {}
    embedding_parameters: dict[str, dict[str, str]] = {}
    if args.capture_evoformer:
        evoformer_prefix = "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
        parameter_index = 0
        for path in sorted(params):
            if not path.startswith(evoformer_prefix):
                continue
            parameter_records: dict[str, str] = {}
            for parameter_name, stacked_value in sorted(params[path].items()):
                value = np.asarray(stacked_value)
                if value.shape[0] != 48:
                    raise RuntimeError(f"expected stacked block axis for {path}/{parameter_name}: {value.shape}")
                tensor_name = f"haiku_{parameter_index:04d}"
                parameter_index += 1
                tensors[tensor_name] = np.asarray(value[args.block], dtype=np.float32)
                parameter_records[parameter_name] = tensor_name
            haiku_parameters[path.removeprefix(evoformer_prefix)] = parameter_records
    if args.capture_evoformer_stack:
        evoformer_prefix = "alphafold/alphafold_iteration/evoformer/evoformer_iteration/"
        parameter_index = 0
        for path in sorted(params):
            if not path.startswith(evoformer_prefix):
                continue
            parameter_records = {}
            for parameter_name, stacked_value in sorted(params[path].items()):
                value = np.asarray(stacked_value)
                if value.shape[0] != 48:
                    raise RuntimeError(f"expected stacked block axis for {path}/{parameter_name}: {value.shape}")
                tensor_name = f"stack_haiku_{parameter_index:04d}"
                parameter_index += 1
                tensors[tensor_name] = np.asarray(value, dtype=np.float32)
                parameter_records[parameter_name] = tensor_name
            haiku_stack_parameters[path.removeprefix(evoformer_prefix)] = parameter_records
        extra_prefix = "alphafold/alphafold_iteration/evoformer/extra_msa_stack/"
        parameter_index = 0
        for path in sorted(params):
            if not path.startswith(extra_prefix):
                continue
            parameter_records = {}
            for parameter_name, stacked_value in sorted(params[path].items()):
                value = np.asarray(stacked_value)
                if value.shape[0] != 4:
                    raise RuntimeError(f"expected four stacked extra-MSA blocks for {path}/{parameter_name}: {value.shape}")
                tensor_name = f"extra_stack_haiku_{parameter_index:04d}"
                parameter_index += 1
                tensors[tensor_name] = np.asarray(value, dtype=np.float32)
                parameter_records[parameter_name] = tensor_name
            haiku_extra_stack_parameters[path.removeprefix(extra_prefix)] = parameter_records
        embedding_prefix = "alphafold/alphafold_iteration/evoformer/"
        parameter_index = 0
        for path in sorted(params):
            if not path.startswith(embedding_prefix):
                continue
            relative_path = path.removeprefix(embedding_prefix)
            if relative_path.startswith(("evoformer_iteration/", "extra_msa_stack/", "template")):
                continue
            parameter_records = {}
            for parameter_name, value in sorted(params[path].items()):
                tensor_name = f"embedding_haiku_{parameter_index:04d}"
                parameter_index += 1
                tensors[tensor_name] = np.asarray(value, dtype=np.float32)
                parameter_records[parameter_name] = tensor_name
            embedding_parameters[relative_path] = parameter_records

    args.output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        array = np.asarray(tensor, dtype="<f4", order="C")
        filename = f"{name}.f32.bin"
        (args.output / filename).write_bytes(array.tobytes())
        records[name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}

    length = len(sequence)
    manifest = {
        "formatVersion": 1,
        "operator": "TriangleMultiplicationOutgoing",
        "source": (
            f"AlphaFold model_{args.model_number}_ptm query-only MSA, "
            f"recycle {capture_recycle}, main Evoformer block {args.block}"
        ),
        "sequence": {"name": args.name, "aminoAcids": sequence},
        "input": {
            "kind": input_kind,
            "msaDepth": msa_depth,
            "maxSeq": max_seq,
            "maxExtraSeq": max_extra_seq,
        },
        "model": {
            "name": model_name,
            "block": args.block,
            "recycles": args.recycles,
            "capturedRecycle": capture_recycle,
        },
        "referencePrediction": {
            "meanPlddt": float(np.mean(prediction["plddt"])),
            "rankingConfidence": float(prediction["ranking_confidence"]),
            "recycleIteration": int(recycle_iteration),
            "recycleMetrics": recycle_metrics,
        },
        "shape": {"length": length, "cZ": 128, "cHidden": 128},
        "epsilon": 1e-5,
        "tensors": records,
    }
    if args.capture_evoformer:
        manifest["evoformerBlock"] = {
            "captured": True,
            "parameterFormat": "haiku",
            "parameters": haiku_parameters,
        }
    if args.capture_evoformer_stack:
        manifest["evoformerStack"] = {
            "captured": True,
            "blocks": 48,
            "capturedRecycle": capture_recycle,
            "parameterFormat": "stacked-haiku",
            "parameters": haiku_stack_parameters,
        }
        manifest["extraMsaStack"] = {
            "captured": True,
            "blocks": 4,
            "capturedRecycle": capture_recycle,
            "parameterFormat": "stacked-haiku",
            "parameters": haiku_extra_stack_parameters,
        }
        manifest["embedding"] = {
            "parameterFormat": "haiku",
            "parameters": embedding_parameters,
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"captured {args.name}: L={length}, model={model_name}, block={args.block}, "
        f"msa_depth={msa_depth}, invocations={invocation}, "
        f"mean_pLDDT={np.mean(prediction['plddt']):.3f}, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
