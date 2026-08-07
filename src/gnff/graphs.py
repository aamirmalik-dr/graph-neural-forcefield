"""Batched atomistic graphs for potential evaluation.

Structures are packed into one flat tensor batch: atom indices are offset
into a single array, and each pair carries its Cartesian shift vector
(integer shift premultiplied by that structure's own cell), so strained and
unstrained frames of different sizes evaluate together in single vectorized
tensor operations. Triplet indices for angular descriptors are attached only
when an angular cutoff is requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from gnff.frames import Frame
from gnff.neighbors import build_triplets, neighbor_pairs, pairs_within


@dataclass
class Batch:
    """Flat tensor view of one or more structures.

    Attributes:
        positions: (N, 3) Cartesian positions.
        species: (N,) element indices.
        edge_i: (P,) center atom of each directed pair.
        edge_j: (P,) neighbor atom of each directed pair.
        shift_vec: (P, 3) constant Cartesian shift of the neighbor image.
        struct_of_atom: (N,) structure index of each atom.
        n_structs: Number of structures in the batch.
        n_atoms: (S,) atoms per structure.
        ang_keep: (Pa,) pair indices within the angular cutoff, or None.
        tri_ij, tri_ik: (T,) triplet pair indices, or None.
    """

    positions: torch.Tensor
    species: torch.Tensor
    edge_i: torch.Tensor
    edge_j: torch.Tensor
    shift_vec: torch.Tensor
    struct_of_atom: torch.Tensor
    n_structs: int
    n_atoms: torch.Tensor
    ang_keep: torch.Tensor | None = None
    tri_ij: torch.Tensor | None = None
    tri_ik: torch.Tensor | None = None


def build_batch(
    frames: list[Frame],
    cutoff: float,
    angular_cutoff: float | None = None,
    dtype: torch.dtype = torch.float64,
) -> Batch:
    """Pack frames into one Batch.

    Args:
        frames: Structures to batch.
        cutoff: Pair cutoff in Angstrom.
        angular_cutoff: If given, also build triplets within this radius.
        dtype: Torch float dtype for positions and shift vectors.

    Returns:
        Batch ready for model evaluation; positions require grad only after
        the caller sets requires_grad on the positions tensor.
    """
    pos, spec, ei, ej, sv, soa, nat = [], [], [], [], [], [], []
    keep_l, tij_l, tik_l = [], [], []
    atom_off = 0
    pair_off = 0
    for s, fr in enumerate(frames):
        pairs = neighbor_pairs(fr.positions, fr.cell, cutoff)
        pos.append(fr.positions)
        spec.append(fr.species)
        ei.append(pairs.i + atom_off)
        ej.append(pairs.j + atom_off)
        sv.append(pairs.shifts @ fr.cell)
        soa.append(np.full(fr.n_atoms, s))
        nat.append(fr.n_atoms)
        if angular_cutoff is not None:
            keep = pairs_within(pairs, fr.positions, fr.cell, angular_cutoff)
            tri = build_triplets(pairs, keep)
            keep_l.append(keep + pair_off)
            tij_l.append(tri.pair_ij + pair_off)
            tik_l.append(tri.pair_ik + pair_off)
        atom_off += fr.n_atoms
        pair_off += len(pairs)
    batch = Batch(
        positions=torch.tensor(np.concatenate(pos), dtype=dtype),
        species=torch.tensor(np.concatenate(spec), dtype=torch.long),
        edge_i=torch.tensor(np.concatenate(ei), dtype=torch.long),
        edge_j=torch.tensor(np.concatenate(ej), dtype=torch.long),
        shift_vec=torch.tensor(np.concatenate(sv), dtype=dtype),
        struct_of_atom=torch.tensor(np.concatenate(soa), dtype=torch.long),
        n_structs=len(frames),
        n_atoms=torch.tensor(nat, dtype=torch.long),
    )
    if angular_cutoff is not None:
        batch.ang_keep = torch.tensor(np.concatenate(keep_l), dtype=torch.long)
        batch.tri_ij = torch.tensor(np.concatenate(tij_l), dtype=torch.long)
        batch.tri_ik = torch.tensor(np.concatenate(tik_l), dtype=torch.long)
    return batch


def edge_vectors(batch: Batch, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Displacement vectors and lengths for every directed pair.

    Args:
        batch: Packed batch.
        positions: (N, 3) tensor (typically batch.positions, possibly with
            requires_grad=True so forces can come from autograd).

    Returns:
        (vec, r): (P, 3) vectors and (P,) distances.
    """
    vec = positions[batch.edge_j] + batch.shift_vec - positions[batch.edge_i]
    r = torch.linalg.norm(vec, dim=1)
    return vec, r


class BatchCache:
    """Per-frame graph parts computed once, assembled into batches on demand.

    Neighbor lists depend only on the fixed structures, so training epochs
    reuse them; assembling a minibatch is cheap numpy concatenation.
    """

    def __init__(
        self,
        frames: list[Frame],
        cutoff: float,
        angular_cutoff: float | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.frames = frames
        self.cutoff = cutoff
        self.angular_cutoff = angular_cutoff
        self.dtype = dtype
        self.parts = []
        for fr in frames:
            pairs = neighbor_pairs(fr.positions, fr.cell, cutoff)
            part = {
                "pos": fr.positions,
                "spec": fr.species,
                "ei": pairs.i,
                "ej": pairs.j,
                "sv": pairs.shifts @ fr.cell,
                "n": fr.n_atoms,
                "np": len(pairs),
            }
            if angular_cutoff is not None:
                keep = pairs_within(pairs, fr.positions, fr.cell, angular_cutoff)
                tri = build_triplets(pairs, keep)
                part["keep"], part["tij"], part["tik"] = keep, tri.pair_ij, tri.pair_ik
            self.parts.append(part)

    def assemble(self, indices: list[int]) -> Batch:
        """Build a Batch from cached parts for the given frame indices."""
        atom_off, pair_off = 0, 0
        pos, spec, ei, ej, sv, soa, nat = [], [], [], [], [], [], []
        keep_l, tij_l, tik_l = [], [], []
        for s, k in enumerate(indices):
            p = self.parts[k]
            pos.append(p["pos"])
            spec.append(p["spec"])
            ei.append(p["ei"] + atom_off)
            ej.append(p["ej"] + atom_off)
            sv.append(p["sv"])
            soa.append(np.full(p["n"], s))
            nat.append(p["n"])
            if self.angular_cutoff is not None:
                keep_l.append(p["keep"] + pair_off)
                tij_l.append(p["tij"] + pair_off)
                tik_l.append(p["tik"] + pair_off)
            atom_off += p["n"]
            pair_off += p["np"]
        batch = Batch(
            positions=torch.tensor(np.concatenate(pos), dtype=self.dtype),
            species=torch.tensor(np.concatenate(spec), dtype=torch.long),
            edge_i=torch.tensor(np.concatenate(ei), dtype=torch.long),
            edge_j=torch.tensor(np.concatenate(ej), dtype=torch.long),
            shift_vec=torch.tensor(np.concatenate(sv), dtype=self.dtype),
            struct_of_atom=torch.tensor(np.concatenate(soa), dtype=torch.long),
            n_structs=len(indices),
            n_atoms=torch.tensor(nat, dtype=torch.long),
        )
        if self.angular_cutoff is not None:
            batch.ang_keep = torch.tensor(np.concatenate(keep_l), dtype=torch.long)
            batch.tri_ij = torch.tensor(np.concatenate(tij_l), dtype=torch.long)
            batch.tri_ik = torch.tensor(np.concatenate(tik_l), dtype=torch.long)
        return batch


def labels_tensor(frames: list[Frame], dtype: torch.dtype = torch.float64):
    """Stack energy and force labels of a frame list into tensors."""
    energies = torch.tensor([fr.energy for fr in frames], dtype=dtype)
    forces = torch.tensor(np.concatenate([fr.forces for fr in frames]), dtype=dtype)
    return energies, forces
