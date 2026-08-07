"""Periodic neighbor list for general triclinic cells.

Implemented from scratch with an explicit periodic-image search. The image
range per lattice direction comes from the perpendicular spacing of the
corresponding lattice planes, so sheared and strained cells are handled
correctly. Positions are wrapped into the cell before the search and the
returned integer shifts are corrected back, so atoms outside the cell (as MD
produces) are also handled correctly. Verified against ase.neighborlist in
the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairList:
    """Directed neighbor pairs of one periodic structure.

    For pair p, the displacement vector from atom i to the relevant periodic
    image of atom j is positions[j] + shifts[p] @ cell - positions[i].

    Attributes:
        i: (P,) center-atom indices.
        j: (P,) neighbor-atom indices.
        shifts: (P, 3) integer lattice shifts of the neighbor image.
    """

    i: np.ndarray
    j: np.ndarray
    shifts: np.ndarray

    def __len__(self) -> int:
        return len(self.i)


def image_ranges(cell: np.ndarray, cutoff: float) -> np.ndarray:
    """Number of periodic images needed per lattice direction.

    The perpendicular distance between lattice planes normal to reciprocal
    vector k is 1 / |row k of inv(cell).T|; images are needed out to
    ceil(cutoff / spacing).
    """
    inv = np.linalg.inv(cell)
    spacings = 1.0 / np.linalg.norm(inv, axis=1)
    return np.ceil(cutoff / spacings).astype(int)


def neighbor_pairs(positions: np.ndarray, cell: np.ndarray, cutoff: float) -> PairList:
    """All directed pairs (i, j, shift) with distance < cutoff.

    Args:
        positions: (N, 3) Cartesian coordinates; may lie outside the cell.
        cell: (3, 3) lattice matrix, rows are lattice vectors.
        cutoff: Cutoff radius in Angstrom.

    Returns:
        PairList with shifts valid for the ORIGINAL (unwrapped) positions.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    frac = positions @ np.linalg.inv(cell)
    wrap = np.floor(frac).astype(int)
    wrapped = (frac - wrap) @ cell

    n1, n2, n3 = image_ranges(cell, cutoff)
    shifts = np.array(
        [
            (s1, s2, s3)
            for s1 in range(-n1, n1 + 1)
            for s2 in range(-n2, n2 + 1)
            for s3 in range(-n3, n3 + 1)
        ],
        dtype=int,
    )
    shift_cart = shifts @ cell

    pi, pj, ps = [], [], []
    for s, sc in zip(shifts, shift_cart):
        # d[a, b] = |wrapped[b] + sc - wrapped[a]|
        delta = wrapped[None, :, :] + sc[None, None, :] - wrapped[:, None, :]
        dist = np.linalg.norm(delta, axis=2)
        mask = dist < cutoff
        if (s == 0).all():
            np.fill_diagonal(mask, False)
        a, b = np.nonzero(mask)
        if len(a):
            pi.append(a)
            pj.append(b)
            ps.append(np.tile(s, (len(a), 1)))
    if not pi:
        return PairList(
            i=np.zeros(0, dtype=int), j=np.zeros(0, dtype=int), shifts=np.zeros((0, 3), dtype=int)
        )
    i = np.concatenate(pi)
    j = np.concatenate(pj)
    s = np.concatenate(ps)
    # Undo the wrap: original positions differ from wrapped by wrap @ cell.
    s = s + wrap[i] - wrap[j]
    order = np.lexsort((j, i))
    return PairList(i=i[order], j=j[order], shifts=s[order])


def pair_distances(pairs: PairList, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Distances of every pair, for the original positions."""
    vec = positions[pairs.j] + pairs.shifts @ cell - positions[pairs.i]
    return np.linalg.norm(vec, axis=1)


@dataclass
class TripletList:
    """Unordered angular triplets (j, i, k) indexed into a PairList.

    pair_ij and pair_ik index pairs that share center atom i; each unordered
    neighbor pair around a center appears exactly once.
    """

    pair_ij: np.ndarray
    pair_ik: np.ndarray

    def __len__(self) -> int:
        return len(self.pair_ij)


def pairs_within(pairs: PairList, positions: np.ndarray, cell: np.ndarray, rc: float) -> np.ndarray:
    """Indices of pairs whose distance is below a (shorter) cutoff rc."""
    return np.nonzero(pair_distances(pairs, positions, cell) < rc)[0]


def build_triplets(pairs: PairList, keep: np.ndarray) -> TripletList:
    """All unordered neighbor pairs around each center among kept pairs.

    Args:
        pairs: Full pair list (sorted by center atom i).
        keep: Indices into pairs restricted to the angular cutoff.

    Returns:
        TripletList indexing the FULL pair list.
    """
    tij, tik = [], []
    centers = pairs.i[keep]
    order = np.argsort(centers, kind="stable")
    keep_sorted = keep[order]
    centers_sorted = centers[order]
    uniq, starts = np.unique(centers_sorted, return_index=True)
    starts = list(starts) + [len(centers_sorted)]
    for u in range(len(uniq)):
        block = keep_sorted[starts[u] : starts[u + 1]]
        m = len(block)
        if m < 2:
            continue
        a, b = np.triu_indices(m, k=1)
        tij.append(block[a])
        tik.append(block[b])
    if not tij:
        return TripletList(pair_ij=np.zeros(0, dtype=int), pair_ik=np.zeros(0, dtype=int))
    return TripletList(pair_ij=np.concatenate(tij), pair_ik=np.concatenate(tik))
