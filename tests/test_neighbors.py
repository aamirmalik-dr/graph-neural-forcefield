"""Neighbor list correctness against ase.neighborlist."""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list as ase_neighbor_list

from gnff.frames import ELEMENTS
from gnff.lattice import bcc_supercell
from gnff.neighbors import build_triplets, neighbor_pairs, pair_distances, pairs_within


def _as_atoms(fr) -> Atoms:
    return Atoms(
        symbols=[ELEMENTS[int(s)] for s in fr.species],
        positions=fr.positions,
        cell=fr.cell,
        pbc=True,
    )


def _pair_sets(fr, cutoff):
    ours = neighbor_pairs(fr.positions, fr.cell, cutoff)
    i, j, s = ase_neighbor_list("ijS", _as_atoms(fr), cutoff)
    ref = {(int(a), int(b), tuple(int(x) for x in sh)) for a, b, sh in zip(i, j, s)}
    got = {
        (int(a), int(b), tuple(int(x) for x in sh)) for a, b, sh in zip(ours.i, ours.j, ours.shifts)
    }
    return got, ref


def test_matches_ase_ideal_cell():
    fr = bcc_supercell("TiZrNb", 2, seed=0)
    got, ref = _pair_sets(fr, 5.5)
    assert got == ref


def test_matches_ase_rattled_and_large():
    fr = bcc_supercell("ZrNb", 3, seed=1)
    rng = np.random.default_rng(0)
    fr.positions = fr.positions + rng.normal(0, 0.2, fr.positions.shape)
    got, ref = _pair_sets(fr, 5.0)
    assert got == ref


def test_matches_ase_sheared_cell():
    fr = bcc_supercell("TiNb", 2, seed=2)
    strain = np.eye(3)
    strain[0, 1] = 0.08
    frac = fr.positions @ np.linalg.inv(fr.cell)
    fr.cell = fr.cell @ strain.T
    fr.positions = frac @ fr.cell
    got, ref = _pair_sets(fr, 5.5)
    assert got == ref


def test_matches_ase_atoms_outside_cell():
    """MD moves atoms outside the box; shifts must stay consistent."""
    fr = bcc_supercell("Ti", 2, seed=3)
    fr.positions[0] += fr.cell.sum(axis=0)  # move atom by a full lattice diagonal
    fr.positions[3] -= 2.0 * fr.cell[1]
    got, ref = _pair_sets(fr, 5.5)
    assert got == ref


def test_distances_below_cutoff():
    fr = bcc_supercell("TiZr", 2, seed=4)
    pairs = neighbor_pairs(fr.positions, fr.cell, 4.8)
    d = pair_distances(pairs, fr.positions, fr.cell)
    assert (d < 4.8).all()
    assert (d > 1.0).all()


def test_pairs_are_symmetric():
    fr = bcc_supercell("TiZrNb", 2, seed=5)
    pairs = neighbor_pairs(fr.positions, fr.cell, 5.0)
    fwd = {(int(a), int(b), tuple(map(int, s))) for a, b, s in zip(pairs.i, pairs.j, pairs.shifts)}
    rev = {(b, a, tuple(-np.array(s))) for a, b, s in fwd}
    assert fwd == rev


def test_triplets_count_small_case():
    """A center with m kept neighbors contributes m*(m-1)/2 unordered triplets."""
    fr = bcc_supercell("Nb", 2, seed=6)
    pairs = neighbor_pairs(fr.positions, fr.cell, 4.5)
    keep = pairs_within(pairs, fr.positions, fr.cell, 3.0)  # first shell only: 8 neighbors
    tri = build_triplets(pairs, keep)
    per_center = np.bincount(pairs.i[keep], minlength=fr.n_atoms)
    expected = int(sum(m * (m - 1) // 2 for m in per_center))
    assert len(tri) == expected
    # every triplet shares its center atom
    assert (pairs.i[tri.pair_ij] == pairs.i[tri.pair_ik]).all()


@pytest.mark.parametrize("cutoff", [3.0, 5.0, 6.5])
def test_cutoff_scaling(cutoff):
    fr = bcc_supercell("TiZrNb", 2, seed=7)
    got, ref = _pair_sets(fr, cutoff)
    assert got == ref
