"""Batched graph assembly: batching must not change model output."""

from __future__ import annotations

import numpy as np
import torch

from gnff.graphs import BatchCache, build_batch, edge_vectors
from gnff.lattice import bcc_supercell, rattle_batch
from gnff.models import MpnnConfig, MpnnPotential


def _frames():
    frames = rattle_batch("TiZrNb", 2, 0.1, 3, seed=0, batch=0)
    frames += rattle_batch("Nb", 3, 0.1, 2, seed=1, batch=0)  # mixed sizes
    return frames


def test_batched_equals_single():
    torch.manual_seed(0)
    model = MpnnPotential(MpnnConfig(hidden=16, n_blocks=2, n_rbf=8, cutoff=5.0))
    frames = _frames()
    batch = build_batch(frames, 5.0)
    e_batch, f_batch = model.energy_and_forces(batch)
    off = 0
    for k, fr in enumerate(frames):
        single = build_batch([fr], 5.0)
        e, f = model.energy_and_forces(single)
        assert abs(float(e.detach()) - float(e_batch[k].detach())) < 1e-9
        np.testing.assert_allclose(
            f.detach().numpy(), f_batch.detach().numpy()[off : off + fr.n_atoms], atol=1e-9
        )
        off += fr.n_atoms


def test_cache_assemble_matches_build_batch():
    frames = _frames()
    cache = BatchCache(frames, 5.5, 4.5)
    direct = build_batch(frames, 5.5, 4.5)
    assembled = cache.assemble(list(range(len(frames))))
    for attr in ("positions", "species", "edge_i", "edge_j", "shift_vec", "struct_of_atom"):
        np.testing.assert_array_equal(
            getattr(direct, attr).numpy(), getattr(assembled, attr).numpy()
        )
    np.testing.assert_array_equal(direct.tri_ij.numpy(), assembled.tri_ij.numpy())
    np.testing.assert_array_equal(direct.tri_ik.numpy(), assembled.tri_ik.numpy())


def test_edge_vectors_lengths():
    fr = bcc_supercell("TiZr", 2, seed=9)
    batch = build_batch([fr], 5.0)
    _, r = edge_vectors(batch, batch.positions)
    assert float(r.min()) > 1.0
    assert float(r.max()) < 5.0
