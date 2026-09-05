#!/usr/bin/env python3
"""Capture AlphaFold's own template features and template embedding.

Reads the tensors `tools/dump-template-features.ts` wrote, recomputes every one
of them with the official implementation, and reports the largest disagreement.
The structure is parsed a second time here with Biopython, so the coordinates
this compares against did not come from the port being checked.

Then, with official parameters, it runs `TemplateEmbedding` on those features
and writes the pair update out as the reference the GPU module is tested
against.

  capture_alphafold_template_reference.py <dump-dir> <structure> [--params P] [--chain A]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load(directory: Path, name: str, meta: dict) -> np.ndarray:
    dtype = np.int32 if meta["dtypes"][name] == "int32" else np.float32
    values = np.fromfile(directory / f"{name}.bin", dtype=dtype)
    return values.reshape(meta["shapes"][name])


def biopython_atom37(structure_path: Path, chain_id: str):
    """Sequence, atom37 positions and masks, read independently of the port."""
    from Bio.PDB import PDBParser, MMCIFParser
    from Bio.SeqUtils import seq1
    from alphafold.common import residue_constants as rc

    text = structure_path.read_text()
    parser = MMCIFParser(QUIET=True) if "_atom_site." in text else PDBParser(QUIET=True)
    model = list(parser.get_structure("template", str(structure_path)))[0]
    chain = model[chain_id]
    residues = [r for r in chain if r.id[0] == " " and seq1(r.get_resname()) not in ("X", "")]
    positions = np.zeros((len(residues), rc.atom_type_num, 3), dtype=np.float32)
    mask = np.zeros((len(residues), rc.atom_type_num), dtype=np.float32)
    for index, residue in enumerate(residues):
        for atom in residue:
            name = atom.get_name()
            if name not in rc.atom_order:
                continue
            if atom.get_altloc() not in (" ", "A"):
                continue
            positions[index, rc.atom_order[name]] = atom.get_coord()
            mask[index, rc.atom_order[name]] = 1.0
    return "".join(seq1(r.get_resname()) for r in residues), positions, mask


def report(name: str, ours: np.ndarray, theirs: np.ndarray) -> float:
    if ours.shape != theirs.shape:
        raise SystemExit(f"{name}: shape {ours.shape} against {theirs.shape}")
    difference = float(np.max(np.abs(ours.astype(np.float64) - theirs.astype(np.float64))))
    print(f"  {name:28s} max |difference| {difference:.3e}")
    return difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("structure", type=Path)
    parser.add_argument("--params", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    meta = json.loads((args.dump / "template.json").read_text())
    aatype = load(args.dump, "aatype", meta)
    positions = load(args.dump, "atomPositions", meta)
    mask = load(args.dump, "atomMask", meta)
    length = aatype.shape[0]

    print(f"query {length} residues, chain {meta['chain']}, "
          f"{meta['alignment']['alignedResidues']} aligned")

    print("structure reading, against Biopython:")
    sequence, chain_positions, chain_mask = biopython_atom37(args.structure, meta["chain"])
    if sequence != meta["chainSequence"]:
        raise SystemExit(f"chain sequence differs:\n  ours   {meta['chainSequence']}\n"
                         f"  theirs {sequence}")
    print(f"  sequence                     {len(sequence)} residues, identical")
    worst = report("chain positions", load(args.dump, "chainAtomPositions", meta), chain_positions)
    worst = max(worst, report("chain atom mask", load(args.dump, "chainAtomMask", meta), chain_mask))

    print("template features, against AlphaFold:")
    from alphafold.common import residue_constants as rc
    from alphafold.model import all_atom
    import jax.numpy as jnp

    # Pseudo-beta, as data_transforms.make_pseudo_beta builds it.
    is_glycine = aatype == rc.restype_order["G"]
    pseudo_beta = np.where(
        is_glycine[:, None], positions[:, rc.atom_order["CA"]], positions[:, rc.atom_order["CB"]])
    pseudo_beta_mask = np.where(
        is_glycine, mask[:, rc.atom_order["CA"]], mask[:, rc.atom_order["CB"]])
    worst = max(worst, report("pseudo beta", load(args.dump, "pseudoBeta", meta), pseudo_beta))
    worst = max(worst, report("pseudo beta mask",
                              load(args.dump, "pseudoBetaMask", meta), pseudo_beta_mask))
    backbone = (mask[:, rc.atom_order["N"]] * mask[:, rc.atom_order["CA"]]
                * mask[:, rc.atom_order["C"]])
    worst = max(worst, report("backbone mask", load(args.dump, "backboneMask", meta), backbone))

    torsions = all_atom.atom37_to_torsion_angles(
        aatype=jnp.asarray(aatype)[None], all_atom_pos=jnp.asarray(positions)[None],
        all_atom_mask=jnp.asarray(mask)[None],
        # See the note in src/input/template-torsions.ts: an undefined angle is
        # noise from a degenerate frame, and this is AlphaFold's own way of
        # pinning it to something a port can reproduce.
        placeholder_for_undefined=True)
    worst = max(worst, report("torsion sin/cos", load(args.dump, "torsionSinCos", meta),
                              np.asarray(torsions["torsion_angles_sin_cos"][0])))
    worst = max(worst, report("alternative sin/cos",
                              load(args.dump, "alternativeTorsionSinCos", meta),
                              np.asarray(torsions["alt_torsion_angles_sin_cos"][0])))
    worst = max(worst, report("torsion mask", load(args.dump, "torsionMask", meta),
                              np.asarray(torsions["torsion_angles_mask"][0])))

    angle_features = np.concatenate([
        np.eye(22, dtype=np.float32)[aatype],
        np.asarray(torsions["torsion_angles_sin_cos"][0]).reshape(length, 14),
        np.asarray(torsions["alt_torsion_angles_sin_cos"][0]).reshape(length, 14),
        np.asarray(torsions["torsion_angles_mask"][0]),
    ], axis=-1)
    worst = max(worst, report("angle features",
                              load(args.dump, "angleFeatures", meta), angle_features))
    worst = max(worst, report("angle row mask", load(args.dump, "angleRowMask", meta),
                              np.asarray(torsions["torsion_angles_mask"][0])[:, 2]))

    print(f"worst disagreement over every feature: {worst:.3e}")

    if args.params is None:
        return

    out = args.out if args.out is not None else args.dump
    out.mkdir(parents=True, exist_ok=True)
    pair_feature, pair_update = template_embedding(args.params, aatype, positions, mask)
    msa_row = template_msa_row(args.params, angle_features)
    for name, tensor in (("templatePairFeature", pair_feature), ("templatePairUpdate", pair_update),
                         ("templateMsaRow", msa_row)):
        tensor.astype("<f4").tofile(out / f"{name}.bin")
        print(f"  wrote {name} {tensor.shape}")
    (out / "reference.json").write_text(json.dumps({
        "source": "alphafold model_1_ptm, template_embedding",
        "query": meta["query"], "chain": meta["chain"],
        "shapes": {"templatePairFeature": list(pair_feature.shape),
                   "templatePairUpdate": list(pair_update.shape),
                   "templateMsaRow": list(msa_row.shape)},
        "worstFeatureDisagreement": worst,
    }, indent=2) + "\n")


def template_msa_row(params_path: Path, angle_features):
    """`template_single_embedding` then relu then `template_projection`."""
    import jax
    import jax.numpy as jnp

    parameters = np.load(params_path, allow_pickle=False)
    evoformer = "alphafold/alphafold_iteration/evoformer/"

    def weight(module: str, name: str):
        return jnp.asarray(parameters[f"{evoformer}{module}//{name}"])

    hidden = jnp.dot(jnp.asarray(angle_features), weight("template_single_embedding", "weights"))
    hidden = jax.nn.relu(hidden + weight("template_single_embedding", "bias"))
    row = jnp.dot(hidden, weight("template_projection", "weights"))
    return np.asarray(row + weight("template_projection", "bias"))


def template_embedding(params_path: Path, aatype, positions, mask):
    """`SingleTemplateEmbedding` inputs and `TemplateEmbedding`'s output."""
    import haiku as hk
    import jax
    import jax.numpy as jnp
    from alphafold.model import config as af_config
    from alphafold.model import modules

    length = aatype.shape[0]
    parameters = np.load(params_path, allow_pickle=False)
    prefix = "alphafold/alphafold_iteration/evoformer/template_embedding/"
    subtree: dict[str, dict[str, np.ndarray]] = {}
    for key in parameters.files:
        if not key.startswith(prefix):
            continue
        module, name = key.removeprefix(prefix).split("//")
        subtree.setdefault(f"template_embedding/{module}", {})[name] = jnp.asarray(parameters[key])

    model_config = af_config.model_config("model_1_ptm")
    embedding_config = model_config.model.embeddings_and_evoformer.template
    # ColabFold's AlphaFold defaults the triangle multiplication to the fused
    # projection weights of the 2.3 parameters. The 2021 model_1_ptm file this
    # port ships has the separate ones, whose names differ.
    stack = embedding_config.template_pair_stack
    stack.triangle_multiplication_outgoing.fuse_projection_weights = False
    stack.triangle_multiplication_incoming.fuse_projection_weights = False
    global_config = model_config.model.global_config

    # One template, in the four slots AlphaFold always pads to.
    templates = 4
    batch = {
        "template_aatype": jnp.asarray(np.stack([aatype] + [np.full_like(aatype, 21)] * 3)),
        "template_all_atom_positions": jnp.asarray(
            np.stack([positions] + [np.zeros_like(positions)] * 3)),
        "template_all_atom_masks": jnp.asarray(np.stack([mask] + [np.zeros_like(mask)] * 3)),
        "template_mask": jnp.asarray(np.array([1.0] + [0.0] * 3, dtype=np.float32)),
    }
    from alphafold.common import residue_constants as rc
    is_glycine = batch["template_aatype"] == rc.restype_order["G"]
    batch["template_pseudo_beta"] = jnp.where(
        is_glycine[..., None],
        batch["template_all_atom_positions"][:, :, rc.atom_order["CA"]],
        batch["template_all_atom_positions"][:, :, rc.atom_order["CB"]])
    batch["template_pseudo_beta_mask"] = jnp.where(
        is_glycine,
        batch["template_all_atom_masks"][:, :, rc.atom_order["CA"]],
        batch["template_all_atom_masks"][:, :, rc.atom_order["CB"]])

    query = jnp.zeros((length, length, 128), dtype=jnp.float32)
    mask_2d = jnp.ones((length, length), dtype=jnp.float32)
    # The design rests on this: with one unmasked template the pointwise
    # attention softmaxes over a single key, so its output cannot depend on the
    # query pair, and the whole module is a constant for the prediction rather
    # than something to recompute every recycle. Check it rather than trust it.
    noisy_query = jax.random.normal(jax.random.PRNGKey(1), (length, length, 128)) * 3.0

    def forward(query_embedding, template_batch, pair_mask):
        return modules.TemplateEmbedding(embedding_config, global_config)(
            query_embedding, template_batch, pair_mask, is_training=False)

    transformed = hk.transform(forward)
    apply = jax.jit(transformed.apply)
    update = apply(subtree, jax.random.PRNGKey(0), query, batch, mask_2d)
    with_noise = apply(subtree, jax.random.PRNGKey(0), noisy_query, batch, mask_2d)
    swing = float(np.max(np.abs(np.asarray(update) - np.asarray(with_noise))))
    print(f"  one template, pair update against a random query: {swing:.3e}")
    if swing > 1e-4:
        raise SystemExit("the template embedding depends on the query pair; it cannot be cached")

    feature = single_template_pair_feature(embedding_config, batch, 0)
    del templates
    return np.asarray(feature), np.asarray(update)


def single_template_pair_feature(config, batch, index):
    """The 88-channel input to `embedding2d`, as SingleTemplateEmbedding builds it."""
    import jax.numpy as jnp
    from alphafold.common import residue_constants as rc
    from alphafold.model import modules, quat_affine

    length = batch["template_aatype"].shape[1]
    template_mask = batch["template_pseudo_beta_mask"][index]
    template_mask_2d = template_mask[:, None] * template_mask[None, :]
    dgram = modules.dgram_from_positions(
        batch["template_pseudo_beta"][index], **config.dgram_features)
    to_concat = [dgram, template_mask_2d[:, :, None]]
    aatype = jnp.eye(22)[batch["template_aatype"][index]]
    to_concat.append(jnp.tile(aatype[None, :, :], [length, 1, 1]))
    to_concat.append(jnp.tile(aatype[:, None, :], [1, length, 1]))
    n, ca, c = (rc.atom_order[a] for a in ("N", "CA", "C"))
    raw = batch["template_all_atom_positions"][index]
    rot, trans = quat_affine.make_transform_from_reference(
        n_xyz=raw[:, n], ca_xyz=raw[:, ca], c_xyz=raw[:, c])
    affines = quat_affine.QuatAffine(
        quaternion=quat_affine.rot_to_quat(rot, unstack_inputs=True),
        translation=trans, rotation=rot, unstack_inputs=True)
    points = [jnp.expand_dims(x, axis=-2) for x in affines.translation]
    affine_vec = affines.invert_point(points, extra_dims=1)
    unit_vector = [jnp.zeros_like(x)[..., None] for x in affine_vec]
    to_concat.extend(unit_vector)
    frame_mask = (batch["template_all_atom_masks"][index][..., n]
                  * batch["template_all_atom_masks"][index][..., ca]
                  * batch["template_all_atom_masks"][index][..., c])
    to_concat.append((frame_mask[:, None] * frame_mask[None, :])[..., None])
    act = jnp.concatenate(to_concat, axis=-1)
    return act * template_mask_2d[..., None]


if __name__ == "__main__":
    main()
