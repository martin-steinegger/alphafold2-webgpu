"""Times ColabFold's nearest-centre clustering at the shapes this project folds.

AlphaFold computes the agreement between every extra row and every centre as a
matmul over a 23-way one-hot: [extra, res * 23] @ [res * 23, centres]. That is
23x the arithmetic of comparing the codes directly, because 22 of every 23
products are zero, but it is a BLAS call rather than a loop. Which of the two
wins is a question about hardware, not about asymptotics, so it is measured.

Run in the colabfold env:
  /home/milot/miniforge3/envs/colabfold/bin/python tools/bench_colabfold_clustering.py
"""
import os
import sys
import time

DEVICE = os.environ.get("BENCH_DEVICE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", DEVICE)

import jax
import jax.numpy as jnp
import numpy as np

RESIDUES = int(os.environ.get("BENCH_RESIDUES", "825"))
CENTRES = int(os.environ.get("BENCH_CENTRES", "508"))
EXTRAS = int(os.environ.get("BENCH_EXTRAS", "1024"))
REPEATS = int(os.environ.get("BENCH_REPEATS", "5"))


def assignment_only(msa, msa_mask, extra_msa, extra_mask, weights, deletion, extra_deletion):
    """Just the agreement and the soft assignment: our `featurise: cluster`."""
    msa_one_hot = msa_mask[:, :, None] * jax.nn.one_hot(msa, 23)
    extra_one_hot = extra_mask[:, :, None] * jax.nn.one_hot(extra_msa, 23)
    agreement = jnp.einsum("mrc,nrc->nm", extra_one_hot, weights * msa_one_hot)
    assignment = jax.nn.softmax(1e3 * agreement, axis=0)
    assignment *= jnp.einsum("mr,nr->mn", msa_mask, extra_mask)
    return assignment


def clusters(msa, msa_mask, extra_msa, extra_mask, weights, deletion, extra_deletion):
    """The whole of `nearest_neighbor_clusters`: our cluster plus profile."""
    msa_one_hot = msa_mask[:, :, None] * jax.nn.one_hot(msa, 23)
    extra_one_hot = extra_mask[:, :, None] * jax.nn.one_hot(extra_msa, 23)
    msa_masked = msa_mask[:, :, None] * msa_one_hot
    extra_masked = extra_mask[:, :, None] * extra_one_hot
    agreement = jnp.einsum("mrc,nrc->nm", extra_masked, weights * msa_masked)
    assignment = jax.nn.softmax(1e3 * agreement, axis=0)
    assignment *= jnp.einsum("mr,nr->mn", msa_mask, extra_mask)
    count = jnp.sum(assignment, axis=-1) + 1.0
    msa_sum = jnp.einsum("nm,mrc->nrc", assignment, extra_masked) + msa_masked
    profile = msa_sum / count[:, None, None]
    del_sum = jnp.einsum("nm,mc->nc", assignment, extra_mask * extra_deletion) + deletion
    return profile, del_sum / count[:, None]


def main():
    rng = np.random.default_rng(0)
    msa = jnp.asarray(rng.integers(0, 23, (CENTRES, RESIDUES), dtype=np.int32))
    extra = jnp.asarray(rng.integers(0, 23, (EXTRAS, RESIDUES), dtype=np.int32))
    msa_mask = jnp.ones((CENTRES, RESIDUES), jnp.float32)
    extra_mask = jnp.ones((EXTRAS, RESIDUES), jnp.float32)
    weights = jnp.array([1.0] * 21 + [0.0] + [0.0], jnp.float32)
    deletion = jnp.zeros((CENTRES, RESIDUES), jnp.float32)
    extra_deletion = jnp.zeros((EXTRAS, RESIDUES), jnp.float32)
    args = (msa, msa_mask, extra, extra_mask, weights, deletion, extra_deletion)

    print(f"backend        {jax.default_backend()}")
    print(f"shape          {CENTRES} centres, {EXTRAS} extras, {RESIDUES} residues")
    for name, function in [("assignment only", assignment_only), ("whole function", clusters)]:
        compiled = jax.jit(function)
        start = time.perf_counter()
        jax.block_until_ready(compiled(*args))
        compile_ms = (time.perf_counter() - start) * 1000
        times = []
        for _ in range(REPEATS):
            start = time.perf_counter()
            jax.block_until_ready(compiled(*args))
            times.append((time.perf_counter() - start) * 1000)
        print(f"{name:16s} {min(times):7.1f} ms best, {np.median(times):7.1f} ms median"
              f"  (first call {compile_ms:.0f} ms, with the compile)")
    flops = 2 * EXTRAS * CENTRES * RESIDUES * 23
    print(f"one agreement matmul is {flops / 1e9:.1f} GFLOP")


if __name__ == "__main__":
    sys.exit(main())
