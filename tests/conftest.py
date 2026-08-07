"""Shared fixtures: small structures and synthetically labeled frames."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnff.frames import Frame
from gnff.graphs import build_batch
from gnff.lattice import bcc_supercell, rattle_batch
from gnff.models import RidgePotential


@pytest.fixture(scope="session")
def rattled_cell() -> Frame:
    fr = bcc_supercell("TiZrNb", 2, seed=1)
    rng = np.random.default_rng(2)
    fr.positions = fr.positions + rng.normal(0.0, 0.08, fr.positions.shape)
    return fr


@pytest.fixture(scope="session")
def oracle() -> RidgePotential:
    """A fixed random linear potential used as label source in tests."""
    torch.manual_seed(0)
    model = RidgePotential()
    with torch.no_grad():
        model.weight.copy_(0.05 * torch.randn_like(model.weight))
        model.bias.copy_(0.01 * torch.randn_like(model.bias))
        model.elem_ref.copy_(torch.tensor([-5.0, -6.0, -7.0], dtype=torch.float64))
    return model


def label_with(model, frames: list[Frame]) -> list[Frame]:
    """Label frames in place using a potential model as oracle."""
    for fr in frames:
        batch = build_batch([fr], model.params.rc_radial, model.params.rc_angular)
        e, f = model.energy_and_forces(batch)
        fr.energy = float(e.detach())
        fr.forces = f.detach().numpy()
    return frames


@pytest.fixture(scope="session")
def labeled_frames(oracle) -> list[Frame]:
    per_comp = []
    for k, comp in enumerate(("Ti", "Nb", "TiZrNb")):
        batch = rattle_batch(comp, 2, 0.08, 6, seed=100 + k, batch=0)
        batch += rattle_batch(comp, 2, 0.15, 6, seed=200 + k, batch=1)
        per_comp.append(batch)
    # Interleave compositions so any slice of the list covers all three.
    frames = [fr for triple in zip(*per_comp) for fr in triple]
    return label_with(oracle, frames)
