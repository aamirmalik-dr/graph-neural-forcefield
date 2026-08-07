"""Model physics: invariances, exact forces, cutoff continuity, checkpoints."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnff.checkpoints import load_checkpoint, save_checkpoint
from gnff.evaluate import model_cutoffs, predict
from gnff.frames import Frame
from gnff.graphs import build_batch
from gnff.models import AcsfNet, MpnnConfig, MpnnPotential, RidgePotential
from gnff.physics import dimer_scan, force_consistency, invariance_report


def _models():
    torch.manual_seed(1)
    mpnn = MpnnPotential(MpnnConfig(hidden=16, n_blocks=2, n_rbf=8, cutoff=5.0))
    acsf = AcsfNet(hidden=(8, 8))
    ridge = RidgePotential()
    with torch.no_grad():
        ridge.weight.copy_(0.05 * torch.randn_like(ridge.weight))
        ridge.bias.copy_(0.01 * torch.randn_like(ridge.bias))
    return {"mpnn": mpnn, "acsf_net": acsf, "ridge": ridge}


@pytest.mark.parametrize("name", ["mpnn", "acsf_net", "ridge"])
def test_invariances(name):
    model = _models()[name]
    report = invariance_report(model)
    assert report["translation_delta_mev"] < 1e-6
    assert report["rotation_delta_mev"] < 1e-6
    assert report["permutation_delta_mev"] < 1e-6


@pytest.mark.parametrize("name", ["mpnn", "acsf_net", "ridge"])
def test_forces_match_finite_differences(name):
    model = _models()[name]
    report = force_consistency(model)
    assert report["max_abs_error_ev_per_a"] < 5e-7


def test_mpnn_energy_continuous_at_cutoff():
    model = _models()["mpnn"]
    scan = dimer_scan(model, span=0.3, step=0.002)
    # smooth envelope: consecutive 0.002 A steps change energy by < 0.5 meV
    assert scan["max_step_change_mev"] < 0.5
    # beyond the cutoff the dimer energy is exactly the isolated-atom sum
    assert scan["tail_flatness_mev"] < 1e-9


def test_acsf_energy_continuous_at_cutoff():
    model = _models()["acsf_net"]
    scan = dimer_scan(model, span=0.3, step=0.002)
    assert scan["max_step_change_mev"] < 0.5
    assert scan["tail_flatness_mev"] < 1e-9


def test_message_passing_extends_range():
    """Interaction blocks propagate information beyond the bare cutoff.

    On an A-B-C chain with C outside A's cutoff but inside B's, moving C
    changes A's ATOMIC energy through message passing (C perturbs B's
    features, which reach A in the next block). The descriptor model's
    atomic energy is strictly local by construction, so A's atomic energy
    is exactly unchanged. This is the architectural difference the
    benchmark probes, asserted here mechanically.
    """
    torch.manual_seed(0)
    mpnn = MpnnPotential(MpnnConfig(hidden=16, n_blocks=2, n_rbf=8, cutoff=5.0))
    acsf = AcsfNet(hidden=(8, 8))

    def energy_of_a(model, x_c: float) -> float:
        fr = Frame(
            species=np.array([0, 1, 2]),
            positions=np.array([[10.0, 10.0, 10.0], [14.0, 10.0, 10.0], [x_c, 10.0, 10.0]]),
            cell=np.eye(3) * 40.0,
        )
        rc, rca = model_cutoffs(model)
        batch = build_batch([fr], rc, rca)
        with torch.no_grad():
            atom_e = model.atomic_energies(batch, batch.positions)
        return float(atom_e[0])

    # C at 18.0: 8 A from A (beyond every cutoff), 4 A from B (in range).
    mpnn_delta = abs(energy_of_a(mpnn, 18.0) - energy_of_a(mpnn, 18.4))
    acsf_delta = abs(energy_of_a(acsf, 18.0) - energy_of_a(acsf, 18.4))
    assert mpnn_delta > 1e-10
    assert acsf_delta < 1e-14


@pytest.mark.parametrize("name", ["mpnn", "acsf_net", "ridge"])
def test_checkpoint_roundtrip(name, tmp_path, rattled_cell):
    model = _models()[name]
    if name == "acsf_net":
        rc, rca = model_cutoffs(model)
        batch = build_batch([rattled_cell], rc, rca)
        from gnff.models.acsf import compute_acsf

        model.fit_scaler(compute_acsf(batch.positions, batch, model.params))
    path = tmp_path / f"{name}.pt"
    save_checkpoint(model, path)
    loaded = load_checkpoint(path)
    fr = rattled_cell
    fr_l = Frame(fr.species.copy(), fr.positions.copy(), fr.cell.copy())
    e0, f0 = predict(model, [fr_l])
    e1, f1 = predict(loaded, [fr_l])
    np.testing.assert_allclose(e0, e1, atol=1e-12)
    np.testing.assert_allclose(f0, f1, atol=1e-12)


def test_parameter_counts():
    models = _models()
    assert models["ridge"].n_parameters() == 147
    assert models["acsf_net"].n_parameters() > models["ridge"].n_parameters()
    assert models["mpnn"].n_parameters() > models["acsf_net"].n_parameters()


def test_scaler_dead_features_get_unit_std():
    model = AcsfNet(hidden=(8, 8))
    desc = torch.zeros(10, model.params.n_features, dtype=torch.float64)
    desc[:, 0] = torch.linspace(0, 1, 10)
    model.fit_scaler(desc)
    assert float(model.feat_std[1]) == 1.0
    assert float(model.feat_std[0]) < 1.0
