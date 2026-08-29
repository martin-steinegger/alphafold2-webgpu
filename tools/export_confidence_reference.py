#!/usr/bin/env python3
"""Export AF2 pLDDT/PAE head parameters and final-recycle reference logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from alphafold.common import confidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("params", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.parent

    def tensor(name: str) -> np.ndarray:
        record = manifest["tensors"][name]
        return np.fromfile(directory / record["file"], dtype="<f4").reshape(record["shape"])

    params = np.load(args.params, allow_pickle=False)
    groups = {
        "predictedLddt": "alphafold/alphafold_iteration/predicted_lddt_head/",
        "predictedAlignedError": "alphafold/alphafold_iteration/predicted_aligned_error_head/",
    }
    parameter_records: dict[str, dict[str, dict[str, str]]] = {}
    index = 0
    for group, prefix in groups.items():
        records: dict[str, dict[str, str]] = {}
        for key in sorted(params.files):
            if not key.startswith(prefix):
                continue
            module, name = key.removeprefix(prefix).split("//")
            tensor_name = f"confidence_haiku_{index:04d}"
            index += 1
            value = np.asarray(params[key], dtype="<f4", order="C")
            filename = f"{tensor_name}.f32.bin"
            (directory / filename).write_bytes(value.tobytes())
            manifest["tensors"][tensor_name] = {"file": filename, "shape": list(value.shape), "dtype": "float32"}
            records.setdefault(module, {})[name] = tensor_name
        parameter_records[group] = records

    def p(group: str, module: str, name: str) -> np.ndarray:
        tensor_name = parameter_records[group][module][name]
        return tensor(tensor_name)

    scale = p("predictedLddt", "input_layer_norm", "scale")
    offset = p("predictedLddt", "input_layer_norm", "offset")
    def calculate(act: np.ndarray, pair: np.ndarray):
        act = (act - act.mean(-1, keepdims=True)) / np.sqrt(act.var(-1, keepdims=True) + 1e-5) * scale + offset
        act = np.maximum(0, act @ p("predictedLddt", "act_0", "weights") + p("predictedLddt", "act_0", "bias"))
        act = np.maximum(0, act @ p("predictedLddt", "act_1", "weights") + p("predictedLddt", "act_1", "bias"))
        lddt_logits = act @ p("predictedLddt", "logits", "weights") + p("predictedLddt", "logits", "bias")
        pae_logits = pair @ p("predictedAlignedError", "logits", "weights") + p("predictedAlignedError", "logits", "bias")
        breaks = np.linspace(0.0, 31.0, pae_logits.shape[-1] - 1, dtype=np.float32)
        probs = np.exp(lddt_logits - lddt_logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        plddt_centers = (np.arange(lddt_logits.shape[-1], dtype=np.float32) + 0.5) / lddt_logits.shape[-1]
        plddt = probs @ plddt_centers * 100.0
        return lddt_logits, pae_logits, breaks, plddt, confidence.predicted_tm_score(pae_logits, breaks)

    lddt_logits, pae_logits, breaks, plddt, ptm = calculate(
        tensor("structureFinalRepresentation"), tensor("structureInputPair"),
    )
    values = {
        "confidenceLddtLogits": lddt_logits,
        "confidencePlddt": plddt,
        "confidencePaeLogits": pae_logits,
        "confidencePaeBreaks": breaks,
    }
    for name, value in values.items():
        array = np.asarray(value, dtype="<f4", order="C")
        filename = f"{name}.f32.bin"
        (directory / filename).write_bytes(array.tobytes())
        manifest["tensors"][name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
    recycle_references = []
    for recycle in range(4):
        recycle_values = calculate(
            tensor(f"structureRecycle{recycle}FinalRepresentation"),
            tensor(f"stackRecycle{recycle}ExpectedPair"),
        )
        recycle_lddt, recycle_pae, _, recycle_plddt, recycle_ptm = recycle_values
        for name, value in {
            f"confidenceRecycle{recycle}LddtLogits": recycle_lddt,
            f"confidenceRecycle{recycle}PaeLogits": recycle_pae,
            f"confidenceRecycle{recycle}Plddt": recycle_plddt,
        }.items():
            array = np.asarray(value, dtype="<f4", order="C")
            filename = f"{name}.f32.bin"
            (directory / filename).write_bytes(array.tobytes())
            manifest["tensors"][name] = {"file": filename, "shape": list(array.shape), "dtype": "float32"}
        recycle_references.append({"meanPlddt": float(np.mean(recycle_plddt)), "ptm": float(recycle_ptm)})
    manifest["confidenceHeads"] = {
        "parameters": parameter_records,
        "reference": {"meanPlddt": float(np.mean(plddt)), "ptm": float(ptm)},
        "recycleReferences": recycle_references,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"exported confidence heads: pLDDT={np.mean(plddt):.4f}, pTM={ptm:.4f}")


if __name__ == "__main__":
    main()
