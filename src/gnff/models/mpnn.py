"""SchNet-style continuous-filter message-passing potential, from scratch.

The architecture follows the SchNet idea (Schuett et al., 2018) but every
line is original: learned element embeddings, a smooth Gaussian radial basis
under a cosine cutoff envelope, a stack of continuous-filter convolution
interaction blocks with residual updates, and a per-atom energy readout on
top of fixed per-element reference energies. Forces are exact analytic
derivatives obtained by autograd, F = -dE/dr, so energy conservation is
inherited from the smoothness of the network. Because the filter is
multiplied by an envelope that vanishes at the cutoff with zero slope, the
predicted energy is continuous as a neighbor crosses the cutoff sphere (the
physics benchmark measures this).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from gnff.frames import Frame
from gnff.graphs import Batch, edge_vectors

N_ELEMENTS = 3


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    """softplus(x) - log 2; smooth, zero at zero, used throughout SchNet."""
    return nn.functional.softplus(x) - math.log(2.0)


def cosine_envelope(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Smooth cutoff envelope: 1 at r=0, 0 with zero slope at r=cutoff."""
    fc = 0.5 * (torch.cos(math.pi * r / cutoff) + 1.0)
    return torch.where(r < cutoff, fc, torch.zeros_like(fc))


class GaussianBasis(nn.Module):
    """Gaussian radial basis expansion of pair distances."""

    def __init__(self, cutoff: float, n_rbf: int, dtype: torch.dtype = torch.float64):
        super().__init__()
        centers = torch.linspace(0.0, cutoff, n_rbf, dtype=dtype)
        width = centers[1] - centers[0]
        self.register_buffer("centers", centers)
        self.gamma = float(0.5 / (width * width))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.gamma * (r[:, None] - self.centers[None, :]) ** 2)


class Interaction(nn.Module):
    """One continuous-filter convolution block with a residual update."""

    def __init__(self, hidden: int, n_rbf: int):
        super().__init__()
        self.filter_net = nn.Sequential(
            nn.Linear(n_rbf, hidden),
            SSP(),
            nn.Linear(hidden, hidden),
        )
        self.in_dense = nn.Linear(hidden, hidden, bias=False)
        self.out_dense = nn.Linear(hidden, hidden)
        self.atomwise = nn.Sequential(nn.Linear(hidden, hidden), SSP(), nn.Linear(hidden, hidden))

    def forward(
        self,
        x: torch.Tensor,
        rbf: torch.Tensor,
        envelope: torch.Tensor,
        edge_i: torch.Tensor,
        edge_j: torch.Tensor,
    ) -> torch.Tensor:
        w = self.filter_net(rbf) * envelope[:, None]
        messages = self.in_dense(x)[edge_j] * w
        agg = torch.zeros_like(x).index_add(0, edge_i, messages)
        v = self.atomwise(self.out_dense(agg))
        return x + v


class SSP(nn.Module):
    """Shifted-softplus activation module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return shifted_softplus(x)


@dataclass
class MpnnConfig:
    """Hyperparameters of the message-passing potential.

    Attributes:
        hidden: Embedding and feature width.
        n_blocks: Number of interaction blocks.
        n_rbf: Number of Gaussian basis functions.
        cutoff: Graph cutoff radius in Angstrom.
    """

    hidden: int = 48
    n_blocks: int = 3
    n_rbf: int = 24
    cutoff: float = 5.0


class MpnnPotential(nn.Module):
    """Message-passing potential: embeddings, interactions, energy readout.

    The total energy of a structure is a sum of per-atom energies:
    fixed per-element reference (least squares on the training set) plus a
    learned residual scaled by out_scale.
    """

    def __init__(self, config: MpnnConfig, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(N_ELEMENTS, config.hidden)
        self.basis = GaussianBasis(config.cutoff, config.n_rbf, dtype=dtype)
        self.blocks = nn.ModuleList(
            Interaction(config.hidden, config.n_rbf) for _ in range(config.n_blocks)
        )
        self.readout = nn.Sequential(
            nn.Linear(config.hidden, config.hidden // 2),
            SSP(),
            nn.Linear(config.hidden // 2, 1),
        )
        self.register_buffer("elem_ref", torch.zeros(N_ELEMENTS, dtype=dtype))
        self.out_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.to(dtype)

    def set_reference(self, frames: list[Frame]) -> None:
        """Fit per-element reference energies by least squares on frames.

        Also initializes out_scale to the standard deviation of the residual
        per-atom energies so the readout starts at the right magnitude.
        """
        counts = np.stack([fr.element_counts() for fr in frames]).astype(float)
        energies = np.array([fr.energy for fr in frames])
        ref, *_ = np.linalg.lstsq(counts, energies, rcond=None)
        residual = (energies - counts @ ref) / counts.sum(axis=1)
        with torch.no_grad():
            self.elem_ref.copy_(torch.tensor(ref, dtype=self.elem_ref.dtype))
            self.out_scale.fill_(max(float(residual.std()), 1e-3))

    def atomic_energies(self, batch: Batch, positions: torch.Tensor) -> torch.Tensor:
        """Per-atom energies (N,), differentiable in positions.

        After k interaction blocks an atom's energy depends on structure out
        to k times the cutoff radius; this method is what the range test
        probes.
        """
        _, r = edge_vectors(batch, positions)
        rbf = self.basis(r)
        envelope = cosine_envelope(r, self.config.cutoff)
        x = self.embedding(batch.species)
        for block in self.blocks:
            x = block(x, rbf, envelope, batch.edge_i, batch.edge_j)
        return self.out_scale * self.readout(x).squeeze(-1) + self.elem_ref[batch.species]

    def structure_energies(self, batch: Batch, positions: torch.Tensor) -> torch.Tensor:
        """Total energies per structure, differentiable in positions."""
        atom_e = self.atomic_energies(batch, positions)
        return torch.zeros(batch.n_structs, dtype=positions.dtype).index_add(
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
        return {"kind": "mpnn", "config": asdict(self.config), "state": self.state_dict()}

    @staticmethod
    def from_checkpoint(ckpt: dict) -> MpnnPotential:
        model = MpnnPotential(MpnnConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state"])
        return model
