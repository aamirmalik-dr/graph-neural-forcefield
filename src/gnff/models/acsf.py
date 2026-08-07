"""Element-resolved ACSF descriptors and a compact descriptor network.

These are the matched-budget baselines the message-passing network is judged
against: the same hand-built atom-centered symmetry functions used by the
descriptor repo of this cluster (8 shifted-Gaussian radial G2 per neighbor
element and 4 angular G4 sets per element pair, 48 features), feeding either
a small per-element network (this module) or a ridge regression
(gnff.models.ridge). Everything is differentiable in positions so forces
come from autograd.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn

from gnff.frames import Frame
from gnff.graphs import Batch, edge_vectors
from gnff.models.mpnn import N_ELEMENTS, SSP

# Unordered element-pair index for angular features: (t1, t2) -> 0..5
_PAIR_INDEX = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
N_PAIR_TYPES = 6


def _default_radial() -> list[tuple[float, float]]:
    """Shifted-Gaussian (rs, eta) grid covering the first three BCC shells."""
    centers = np.linspace(2.2, 5.2, 8)
    delta = centers[1] - centers[0]
    eta = 1.0 / (2.0 * delta * delta)
    return [(float(rs), float(eta)) for rs in centers]


def _default_angular() -> list[tuple[float, float, float]]:
    """(eta, zeta, lambda) sets, standard narrow/wide angular resolution."""
    return [(0.02, 1.0, 1.0), (0.02, 1.0, -1.0), (0.02, 4.0, 1.0), (0.02, 4.0, -1.0)]


@dataclass
class AcsfParams:
    """Symmetry-function hyperparameters (identical to the descriptor repo).

    Attributes:
        rc_radial: Radial cutoff in Angstrom.
        rc_angular: Angular cutoff in Angstrom (triplets are cubic in
            neighbor count, so this is shorter).
        radial: (rs, eta) shifted-Gaussian parameters.
        angular: (eta, zeta, lambda) G4 parameters.
    """

    rc_radial: float = 5.5
    rc_angular: float = 4.5
    radial: list[tuple[float, float]] = field(default_factory=_default_radial)
    angular: list[tuple[float, float, float]] = field(default_factory=_default_angular)

    @property
    def n_features(self) -> int:
        return N_ELEMENTS * len(self.radial) + N_PAIR_TYPES * len(self.angular)


def _cutoff(r: torch.Tensor, rc: float) -> torch.Tensor:
    fc = 0.5 * (torch.cos(math.pi * r / rc) + 1.0)
    return torch.where(r < rc, fc, torch.zeros_like(fc))


def compute_acsf(positions: torch.Tensor, batch: Batch, params: AcsfParams) -> torch.Tensor:
    """ACSF descriptors for every atom in the batch.

    Args:
        positions: (N, 3) tensor, differentiable; pass a requires_grad tensor
            to take forces by autograd.
        batch: Packed batch built with angular_cutoff=params.rc_angular.
        params: Symmetry-function parameters.

    Returns:
        (N, n_features) descriptor tensor.
    """
    if batch.tri_ij is None:
        raise ValueError("batch was built without triplets; pass angular_cutoff when batching")
    n_atoms = positions.shape[0]
    dtype = positions.dtype
    vec, r = edge_vectors(batch, positions)
    spec_j = batch.species[batch.edge_j]

    n_rad = len(params.radial)
    rs = torch.tensor([p[0] for p in params.radial], dtype=dtype)
    eta = torch.tensor([p[1] for p in params.radial], dtype=dtype)
    fc = _cutoff(r, params.rc_radial)
    g2 = torch.exp(-eta[None, :] * (r[:, None] - rs[None, :]) ** 2) * fc[:, None]
    rad_row = batch.edge_i * N_ELEMENTS + spec_j
    radial_out = torch.zeros(n_atoms * N_ELEMENTS, n_rad, dtype=dtype).index_add(0, rad_row, g2)

    n_ang = len(params.angular)
    ang_out = torch.zeros(n_atoms * N_PAIR_TYPES, n_ang, dtype=dtype)
    if len(batch.tri_ij) > 0:
        v_ij, v_ik = vec[batch.tri_ij], vec[batch.tri_ik]
        r_ij, r_ik = r[batch.tri_ij], r[batch.tri_ik]
        v_jk = v_ik - v_ij
        r_jk2 = (v_jk * v_jk).sum(dim=1)
        r_jk = torch.sqrt(torch.clamp(r_jk2, min=1e-12))
        cos_t = torch.clamp((v_ij * v_ik).sum(dim=1) / (r_ij * r_ik), -1.0, 1.0)
        rca = params.rc_angular
        fc3 = _cutoff(r_ij, rca) * _cutoff(r_ik, rca) * _cutoff(r_jk, rca)
        r2sum = r_ij * r_ij + r_ik * r_ik + r_jk2
        centers = batch.edge_i[batch.tri_ij]
        t1, t2 = spec_j[batch.tri_ij], spec_j[batch.tri_ik]
        pair_type = torch.tensor(_PAIR_INDEX, dtype=torch.long)[t1, t2]
        feats = []
        for eta_a, zeta, lam in params.angular:
            ang = (2.0 ** (1.0 - zeta)) * (1.0 + lam * cos_t) ** zeta
            feats.append(ang * torch.exp(-eta_a * r2sum) * fc3)
        g4 = torch.stack(feats, dim=1)
        ang_out = ang_out.index_add(0, centers * N_PAIR_TYPES + pair_type, g4)

    return torch.cat([radial_out.reshape(n_atoms, -1), ang_out.reshape(n_atoms, -1)], dim=1)


class AcsfNet(nn.Module):
    """Per-element feedforward networks on standardized ACSF descriptors.

    The Behler-Parrinello construction: each element has its own small MLP
    mapping an atom's descriptors to an atomic energy; structure energy is
    the sum plus fixed per-element references. Features constant on the
    training set get unit standard deviation so unseen compositions cannot
    blow up dead features.
    """

    def __init__(
        self,
        params: AcsfParams | None = None,
        hidden: tuple[int, int] = (32, 32),
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.params = params or AcsfParams()
        self.hidden = tuple(hidden)
        n_feat = self.params.n_features
        self.nets = nn.ModuleList()
        for _ in range(N_ELEMENTS):
            layers: list[nn.Module] = []
            widths = [n_feat, *hidden]
            for a, b in zip(widths[:-1], widths[1:]):
                layers += [nn.Linear(a, b), SSP()]
            layers.append(nn.Linear(widths[-1], 1))
            self.nets.append(nn.Sequential(*layers))
        self.register_buffer("feat_mean", torch.zeros(n_feat, dtype=dtype))
        self.register_buffer("feat_std", torch.ones(n_feat, dtype=dtype))
        self.register_buffer("elem_ref", torch.zeros(N_ELEMENTS, dtype=dtype))
        self.to(dtype)

    def fit_scaler(self, descriptors: torch.Tensor) -> None:
        """Set feature mean/std from training descriptors (dead features -> std 1)."""
        with torch.no_grad():
            self.feat_mean.copy_(descriptors.mean(dim=0))
            std = descriptors.std(dim=0)
            self.feat_std.copy_(torch.where(std > 1e-10, std, torch.ones_like(std)))

    def set_reference(self, frames: list[Frame]) -> None:
        """Fit per-element reference energies by least squares on frames."""
        counts = np.stack([fr.element_counts() for fr in frames]).astype(float)
        energies = np.array([fr.energy for fr in frames])
        ref, *_ = np.linalg.lstsq(counts, energies, rcond=None)
        with torch.no_grad():
            self.elem_ref.copy_(torch.tensor(ref, dtype=self.elem_ref.dtype))

    def atomic_energies(self, batch: Batch, positions: torch.Tensor) -> torch.Tensor:
        """Per-atom energies (N,), differentiable in positions.

        Strictly local by construction: an atom's energy depends only on its
        neighborhood within the descriptor cutoffs.
        """
        desc = compute_acsf(positions, batch, self.params)
        z = (desc - self.feat_mean) / self.feat_std
        atom_e = torch.zeros(len(z), dtype=z.dtype)
        for e in range(N_ELEMENTS):
            mask = batch.species == e
            if mask.any():
                atom_e = atom_e + torch.where(
                    mask, self.nets[e](z).squeeze(-1), torch.zeros_like(atom_e)
                )
        return atom_e + self.elem_ref[batch.species]

    def structure_energies(self, batch: Batch, positions: torch.Tensor) -> torch.Tensor:
        """Total energies per structure, differentiable in positions."""
        atom_e = self.atomic_energies(batch, positions)
        return torch.zeros(batch.n_structs, dtype=atom_e.dtype).index_add(
            0, batch.struct_of_atom, atom_e
        )

    def energy_and_forces(
        self, batch: Batch, create_graph: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energies (S,) and autograd forces (N, 3) for a batch."""
        positions = batch.positions.detach().clone().requires_grad_(True)
        energy = self.structure_energies(batch, positions)
        (grad,) = torch.autograd.grad(energy.sum(), positions, create_graph=create_graph)
        return energy, -grad

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_checkpoint(self) -> dict:
        return {
            "kind": "acsf_net",
            "acsf": asdict(self.params),
            "hidden": list(self.hidden),
            "state": self.state_dict(),
        }

    @staticmethod
    def from_checkpoint(ckpt: dict) -> AcsfNet:
        p = ckpt["acsf"]
        params = AcsfParams(
            rc_radial=p["rc_radial"],
            rc_angular=p["rc_angular"],
            radial=[tuple(x) for x in p["radial"]],
            angular=[tuple(x) for x in p["angular"]],
        )
        model = AcsfNet(params, hidden=tuple(ckpt["hidden"]))
        model.load_state_dict(ckpt["state"])
        return model
