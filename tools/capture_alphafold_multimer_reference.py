#!/usr/bin/env python3
"""Capture deterministic official Multimer-v3 IPA and prediction references."""

from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--model-number", type=int, choices=(1,), default=1)
    parser.add_argument("--recycles", type=int, default=0)
    parser.add_argument("--unpaired-a3m", type=Path, nargs="+")
    parser.add_argument("--paired-a3m", type=Path, nargs="+")
    parser.add_argument("--max-msa-sequences", type=int, default=1)
    parser.add_argument("--max-extra-sequences", type=int, default=1)
    args = parser.parse_args()
    chains = [chain.strip().upper() for chain in args.chains.split(":")]
    if len(chains) < 2 or any(not chain for chain in chains):
        raise SystemExit("--chains requires at least two non-empty colon-separated chains")
    if args.recycles < 0:
        raise SystemExit("--recycles must be non-negative")
    if args.max_msa_sequences < 1 or args.max_extra_sequences < 1:
        raise SystemExit("MSA row limits must be positive")

    unique_chains: list[str] = []
    cardinalities: list[int] = []
    for chain in chains:
        if chain in unique_chains:
            cardinalities[unique_chains.index(chain)] += 1
        else:
            unique_chains.append(chain)
            cardinalities.append(1)

    def load_a3ms(paths: list[Path] | None, label: str) -> list[str]:
        if paths is None:
            return [f">{101 + index}\n{chain}\n" for index, chain in enumerate(unique_chains)]
        if len(paths) != len(unique_chains):
            raise SystemExit(
                f"--{label}-a3m requires one file per unique chain ({len(unique_chains)} expected)"
            )
        return [path.read_text() for path in paths]

    unpaired_a3ms = load_a3ms(args.unpaired_a3m, "unpaired")
    paired_a3ms = load_a3ms(args.paired_a3m, "paired")

    original_sum = np.sum

    def compatible_sum(value: Any, *sum_args: Any, **sum_kwargs: Any) -> Any:
        return original_sum(list(value) if isinstance(value, GeneratorType) else value, *sum_args, **sum_kwargs)

    np.sum = compatible_sum  # type: ignore[assignment]
    import jax
    from colabfold.alphafold.models import load_models_and_params
    from colabfold.batch import generate_input_feature, mk_mock_template
    from alphafold.model import folding_multimer, modules, modules_multimer

    raw_features, _ = generate_input_feature(
        unique_chains, cardinalities, unpaired_a3ms, paired_a3ms,
        [mk_mock_template(chain) for chain in unique_chains], True, "alphafold2_multimer_v3",
        args.max_msa_sequences,
    )
    model_name, runner, params = load_models_and_params(
        num_models=1, use_templates=False, num_recycles=args.recycles, recycle_early_stop_tolerance=-1,
        num_ensemble=1, model_order=[args.model_number], model_type="alphafold2_multimer_v3",
        data_dir=args.data_dir, max_seq=args.max_msa_sequences,
        max_extra_seq=args.max_extra_sequences, use_fuse=False,
        use_bfloat16=False, use_dropout=False, save_all=True,
    )[0]
    processed = runner.process_features(raw_features, random_seed=0)
    captured: dict[str, np.ndarray] = {}
    ipa_calls = 0
    msa_feature_calls = 0
    extra_feature_calls = 0
    original_ipa = folding_multimer.InvariantPointAttention.__call__
    original_structure = folding_multimer.StructureModule.__call__
    original_msa_feat = modules_multimer.create_msa_feat
    original_extra_feat = modules_multimer.create_extra_msa_feature
    original_evoformer = modules.EvoformerIteration.__call__
    evoformer_calls = {"extra": 0, "main": 0}

    def evoformer_call(self: Any, activations: Any, masks: Any, *call_args: Any, **call_kwargs: Any) -> Any:
        output = original_evoformer(self, activations, masks, *call_args, **call_kwargs)
        kind = "extra" if self.is_extra_msa else "main"

        def receive(input_msa: Any, input_pair: Any, msa_mask: Any, pair_mask: Any,
                    output_msa: Any, output_pair: Any) -> None:
            block = evoformer_calls[kind]
            if block == 0:
                captured[f"{kind}Block0InputMsa"] = np.asarray(input_msa, dtype=np.float32).copy()
                captured[f"{kind}Block0InputPair"] = np.asarray(input_pair, dtype=np.float32).copy()
                captured[f"{kind}Block0MsaMask"] = np.asarray(msa_mask, dtype=np.float32).copy()
                captured[f"{kind}Block0PairMask"] = np.asarray(pair_mask, dtype=np.float32).copy()
                captured[f"{kind}Block0ExpectedMsa"] = np.asarray(output_msa, dtype=np.float32).copy()
                captured[f"{kind}Block0ExpectedPair"] = np.asarray(output_pair, dtype=np.float32).copy()
            evoformer_calls[kind] += 1

        jax.debug.callback(
            receive, activations["msa"], activations["pair"], masks["msa"], masks["pair"],
            output["msa"], output["pair"], ordered=True,
        )
        return output

    def create_msa_feat(batch: Any) -> Any:
        output = original_msa_feat(batch)

        def receive(value: Any, mask_value: Any) -> None:
            nonlocal msa_feature_calls
            captured[f"feature_msa_feat_recycle{msa_feature_calls}"] = np.asarray(
                value, dtype=np.float32
            ).copy()
            captured[f"feature_msa_mask_recycle{msa_feature_calls}"] = np.asarray(
                mask_value, dtype=np.float32
            ).copy()
            msa_feature_calls += 1

        jax.debug.callback(receive, output, batch["msa_mask"], ordered=True)
        return output

    def create_extra_msa_feature(batch: Any, num_extra_msa: int) -> Any:
        output, mask = original_extra_feat(batch, num_extra_msa)

        def receive(value: Any, mask_value: Any) -> None:
            nonlocal extra_feature_calls
            captured[f"feature_extra_msa_feat_recycle{extra_feature_calls}"] = np.asarray(
                value, dtype=np.float32
            ).copy()
            captured[f"feature_extra_msa_mask_recycle{extra_feature_calls}"] = np.asarray(
                mask_value, dtype=np.float32
            ).copy()
            extra_feature_calls += 1

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

        def receive(pair: Any, single: Any, msa_first_row: Any) -> None:
            if "structureInputPair" not in captured:
                captured["structureInputPair"] = np.asarray(pair, dtype=np.float32).copy()
                captured["structureInputSingle"] = np.asarray(single, dtype=np.float32).copy()
                captured["structureInputMsaFirstRow"] = np.asarray(msa_first_row, dtype=np.float32).copy()

        jax.debug.callback(
            receive, representations["pair"], representations["single"],
            representations["msa_first_row"], ordered=True,
        )
        return output

    folding_multimer.InvariantPointAttention.__call__ = ipa_call
    folding_multimer.StructureModule.__call__ = structure_call
    modules_multimer.create_msa_feat = create_msa_feat
    modules_multimer.create_extra_msa_feature = create_extra_msa_feature
    modules.EvoformerIteration.__call__ = evoformer_call
    try:
        runner.params = params
        recycle_metrics: list[dict[str, float | int]] = []

        def record_recycle(result: Any, recycle: int) -> None:
            ptm = float(result["ptm"])
            iptm = float(result["iptm"])
            recycle_metrics.append({
                "recycle": int(recycle),
                "meanPlddt": float(np.mean(result["plddt"])),
                "ptm": ptm,
                "iptm": iptm,
                "rankingConfidence": 0.2 * ptm + 0.8 * iptm,
            })

        prediction, recycle_iteration = runner.predict(processed, random_seed=0, callback=record_recycle)
        jax.effects_barrier()
    finally:
        folding_multimer.InvariantPointAttention.__call__ = original_ipa
        folding_multimer.StructureModule.__call__ = original_structure
        modules_multimer.create_msa_feat = original_msa_feat
        modules_multimer.create_extra_msa_feature = original_extra_feat
        modules.EvoformerIteration.__call__ = original_evoformer

    if not {"ipaActivations", "ipaMask", "ipaRigidMatrix", "ipaExpected", "structureInputPair",
            "extraBlock0InputMsa", "mainBlock0InputMsa"} <= captured.keys():
        raise RuntimeError(f"failed to capture first IPA call; observed {ipa_calls}")
    length = sum(map(len, chains))
    # The first structure iteration always starts from an identity frame. Store
    # the exact seven-value representation used by the WebGPU implementation.
    captured["ipaAffine"] = np.tile(np.asarray([1, 0, 0, 0, 0, 0, 0], dtype=np.float32), (length, 1))
    feature_ranks = {
        "aatype": 1, "seq_mask": 1, "residue_index": 1,
        "asym_id": 1, "entity_id": 1, "sym_id": 1,
    }
    for feature_name, base_rank in feature_ranks.items():
        if feature_name not in processed:
            raise RuntimeError(f"processed Multimer features are missing {feature_name}")
        value = np.asarray(processed[feature_name])
        has_recycle_axis = value.ndim == base_rank + 1
        if value.ndim != base_rank and not has_recycle_axis:
            raise RuntimeError(f"processed feature {feature_name} has unexpected shape {value.shape}")
        for recycle in range(args.recycles + 1):
            selected = value[recycle] if has_recycle_axis else value
            captured[f"feature_{feature_name}_recycle{recycle}"] = np.asarray(
                selected, dtype=np.float32
            ).copy()
    for recycle in range(args.recycles + 1):
        def feature(name: str) -> np.ndarray:
            return captured[f"feature_{name}_recycle{recycle}"]

        captured[f"feature_target_feat_recycle{recycle}"] = np.eye(
            21, dtype=np.float32
        )[feature("aatype").astype(np.int32)]
        msa_feature_name = f"feature_msa_feat_recycle{recycle}"
        extra_feature_name = f"feature_extra_msa_feat_recycle{recycle}"
        if msa_feature_name not in captured or extra_feature_name not in captured:
            raise RuntimeError(
                f"failed to capture Multimer MSA features; observed {msa_feature_calls} MSA and "
                f"{extra_feature_calls} extra-MSA calls"
            )
        extra_feat = captured.pop(extra_feature_name)
        captured[f"feature_extra_msa_recycle{recycle}"] = np.argmax(
            extra_feat[..., :23], axis=-1
        ).astype(np.float32)
        captured[f"feature_extra_has_deletion_recycle{recycle}"] = extra_feat[..., 23].astype(np.float32)
        captured[f"feature_extra_deletion_value_recycle{recycle}"] = extra_feat[..., 24].astype(np.float32)
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
        "uniqueChains": unique_chains,
        "cardinalities": cardinalities,
        "recycles": args.recycles,
        "recycleIteration": int(recycle_iteration),
        "reference": {
            "meanPlddt": float(np.mean(prediction["plddt"])), "ptm": float(prediction["ptm"]),
            "iptm": float(prediction["iptm"]), "rankingConfidence": float(prediction["ranking_confidence"]),
            "multimerRankingConfidence": 0.2 * float(prediction["ptm"]) + 0.8 * float(prediction["iptm"]),
            "recycleMetrics": recycle_metrics,
        },
        "msaInput": {
            "maxMsaSequences": args.max_msa_sequences,
            "maxExtraSequences": args.max_extra_sequences,
            "unpairedSha256": [hashlib.sha256(value.encode()).hexdigest() for value in unpaired_a3ms],
            "pairedSha256": [hashlib.sha256(value.encode()).hexdigest() for value in paired_a3ms],
        },
        "ipaParameters": parameter_records,
        "tensors": records,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Captured {model_name}, chains={args.chains}, IPA calls={ipa_calls}, output={args.output}")


if __name__ == "__main__":
    main()
