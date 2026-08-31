#!/usr/bin/env python3
"""Capture deterministic official Multimer-v3 IPA and prediction references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import GeneratorType
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chains", default="AC:GG")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".cache" / "colabfold")
    parser.add_argument("--model-number", type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    chains = [chain.strip().upper() for chain in args.chains.split(":")]
    if len(chains) < 2 or any(not chain for chain in chains):
        raise SystemExit("--chains requires at least two non-empty colon-separated chains")

    original_sum = np.sum

    def compatible_sum(value: Any, *sum_args: Any, **sum_kwargs: Any) -> Any:
        return original_sum(list(value) if isinstance(value, GeneratorType) else value, *sum_args, **sum_kwargs)

    np.sum = compatible_sum  # type: ignore[assignment]
    import jax
    from colabfold.alphafold.models import load_models_and_params
    from colabfold.batch import generate_input_feature, mk_mock_template
    from alphafold.model import folding_multimer, modules_multimer

    a3ms = [f">101\n{chain}\n" for chain in chains]
    raw_features, _ = generate_input_feature(
        chains, [1] * len(chains), a3ms, a3ms,
        [mk_mock_template(chain) for chain in chains], True, "alphafold2_multimer_v3", 1,
    )
    model_name, runner, params = load_models_and_params(
        num_models=1, use_templates=False, num_recycles=0, recycle_early_stop_tolerance=-1,
        num_ensemble=1, model_order=[args.model_number], model_type="alphafold2_multimer_v3",
        data_dir=args.data_dir, max_seq=1, max_extra_seq=1, use_fuse=False,
        use_bfloat16=False, use_dropout=False, save_all=True,
    )[0]
    processed = runner.process_features(raw_features, random_seed=0)
    captured: dict[str, np.ndarray] = {}
    ipa_calls = 0
    original_ipa = folding_multimer.InvariantPointAttention.__call__
    original_structure = folding_multimer.StructureModule.__call__
    original_msa_feat = modules_multimer.create_msa_feat
    original_extra_feat = modules_multimer.create_extra_msa_feature

    def create_msa_feat(batch: Any) -> Any:
        output = original_msa_feat(batch)

        def receive(value: Any) -> None:
            captured.setdefault("feature_msa_feat", np.asarray(value, dtype=np.float32).copy())

        jax.debug.callback(receive, output, ordered=True)
        return output

    def create_extra_msa_feature(batch: Any, num_extra_msa: int) -> Any:
        output, mask = original_extra_feat(batch, num_extra_msa)

        def receive(value: Any, mask_value: Any) -> None:
            captured.setdefault("feature_extra_msa_feat", np.asarray(value, dtype=np.float32).copy())
            captured.setdefault("feature_extra_msa_mask", np.asarray(mask_value, dtype=np.float32).copy())

        jax.debug.callback(
            receive, output, mask, ordered=True,
        )
        return output, mask

    def ipa_call(self: Any, inputs_1d: Any, inputs_2d: Any, mask: Any, rigid: Any) -> Any:
        nonlocal ipa_calls
        output = original_ipa(self, inputs_1d, inputs_2d, mask, rigid)

        def receive(act: Any, pair: Any, mask_value: Any, rigid_value: Any, output_value: Any) -> None:
            nonlocal ipa_calls
            if ipa_calls == 0:
                captured["ipaActivations"] = np.asarray(act, dtype=np.float32).copy()
                captured["ipaNormalizedPair"] = np.asarray(pair, dtype=np.float32).copy()
                captured["ipaMask"] = np.asarray(mask_value, dtype=np.float32).copy()
                captured["ipaRigidMatrix"] = np.asarray(rigid_value, dtype=np.float32).copy()
                captured["ipaExpected"] = np.asarray(output_value, dtype=np.float32).copy()
            ipa_calls += 1

        jax.debug.callback(receive, inputs_1d, inputs_2d, mask, rigid.to_array(), output, ordered=True)
        return output

    def structure_call(self: Any, representations: Any, batch: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        output = original_structure(self, representations, batch, *call_args, **call_kwargs)

        def receive(pair: Any) -> None:
            if "structureInputPair" not in captured:
                captured["structureInputPair"] = np.asarray(pair, dtype=np.float32).copy()

        jax.debug.callback(receive, representations["pair"], ordered=True)
        return output

    folding_multimer.InvariantPointAttention.__call__ = ipa_call
    folding_multimer.StructureModule.__call__ = structure_call
    modules_multimer.create_msa_feat = create_msa_feat
    modules_multimer.create_extra_msa_feature = create_extra_msa_feature
    try:
        runner.params = params
        prediction, recycle_iteration = runner.predict(processed, random_seed=0)
        jax.effects_barrier()
    finally:
        folding_multimer.InvariantPointAttention.__call__ = original_ipa
        folding_multimer.StructureModule.__call__ = original_structure
        modules_multimer.create_msa_feat = original_msa_feat
        modules_multimer.create_extra_msa_feature = original_extra_feat

    if not {"ipaActivations", "ipaMask", "ipaRigidMatrix", "ipaExpected", "structureInputPair"} <= captured.keys():
        raise RuntimeError(f"failed to capture first IPA call; observed {ipa_calls}")
    length = sum(map(len, chains))
    # The first structure iteration always starts from an identity frame. Store
    # the exact seven-value representation used by the WebGPU implementation.
    captured["ipaAffine"] = np.tile(np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32), (length, 1))
    for feature_name in (
        "aatype", "seq_mask", "residue_index", "msa", "deletion_matrix", "msa_mask",
        "extra_msa", "extra_has_deletion", "extra_deletion_value", "extra_msa_mask",
        "asym_id", "entity_id", "sym_id", "atom37_atom_exists", "residx_atom37_to_atom14",
    ):
        if feature_name in processed:
            captured[f"feature_{feature_name}"] = np.asarray(processed[feature_name], dtype=np.float32).copy()
    captured["predictionAtom37"] = np.asarray(prediction["structure_module"]["final_atom_positions"], dtype=np.float32)
    captured["predictionPlddt"] = np.asarray(prediction["plddt"], dtype=np.float32)
    captured["predictionPae"] = np.asarray(prediction["predicted_aligned_error"], dtype=np.float32)

    ipa_prefix = "alphafold/alphafold_iteration/structure_module/fold_iteration/invariant_point_attention"
    parameter_paths = [path for path in sorted(params) if path.startswith(ipa_prefix)]
    parameter_records: dict[str, dict[str, str]] = {}
    parameter_index = 0
    for path in parameter_paths:
        relative = path.removeprefix(ipa_prefix).removeprefix("/")
        for name, value in sorted(params[path].items()):
            tensor_name = f"ipaParameter_{parameter_index:03d}_{name}"
            parameter_index += 1
            captured[tensor_name] = np.asarray(value, dtype=np.float32)
            parameter_records.setdefault(relative, {})[name] = tensor_name
    for name in ("scale", "offset"):
        path = "alphafold/alphafold_iteration/structure_module/pair_layer_norm"
        tensor_name = f"pairNorm_{name}"
        captured[tensor_name] = np.asarray(params[path][name], dtype=np.float32)
        parameter_records.setdefault("pair_layer_norm", {})[name] = tensor_name

    args.output.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, Any]] = {}
    for name, value in captured.items():
        array = np.asarray(value, dtype="<f4", order="C")
        filename = f"{name}.f32.bin"
        (args.output / filename).write_bytes(array.tobytes())
        records[name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
    manifest = {
        "formatVersion": 1,
        "source": "official AlphaFold-Multimer-v3 JAX float32 inference",
        "model": {"name": model_name, "type": "alphafold2_multimer_v3", "number": args.model_number},
        "chains": chains,
        "recycleIteration": int(recycle_iteration),
        "reference": {
            "meanPlddt": float(np.mean(prediction["plddt"])), "ptm": float(prediction["ptm"]),
            "iptm": float(prediction["iptm"]), "rankingConfidence": float(prediction["ranking_confidence"]),
        },
        "ipaParameters": parameter_records,
        "tensors": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Captured {model_name}, chains={args.chains}, IPA calls={ipa_calls}, output={args.output}")


if __name__ == "__main__":
    main()
