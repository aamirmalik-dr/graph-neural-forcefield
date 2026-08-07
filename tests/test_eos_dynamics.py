"""EOS fitting and the NVE integrator on analytically known systems."""

from __future__ import annotations

import numpy as np
import torch

from gnff.dynamics import run_nve
from gnff.eos import EV_PER_A3_TO_GPA, birch_murnaghan, fit_eos, scan_frames
from gnff.frames import Frame
from gnff.lattice import bcc_supercell


def test_birch_murnaghan_roundtrip():
    v = np.linspace(14.0, 20.0, 13)
    e = birch_murnaghan(v, e0=-8.0, v0=17.0, b0=1.0, b0p=4.5)
    fit = fit_eos(v, e)
    assert abs(fit["e0_ev_per_atom"] + 8.0) < 1e-8
    assert abs(fit["v0_a3_per_atom"] - 17.0) < 1e-8
    assert abs(fit["b0_gpa"] - 1.0 * EV_PER_A3_TO_GPA) < 1e-6
    assert abs(fit["a0_angstrom"] - (2 * 17.0) ** (1 / 3)) < 1e-9


def test_scan_frames_geometry():
    frames, volumes = scan_frames("Nb", reps=2, seed=0)
    assert len(frames) == 13
    a = 3.31
    assert abs(volumes[6] - a**3 / 2.0) < 0.05  # middle of the scan ~ ideal volume
    # same occupancy at every volume
    for fr in frames[1:]:
        np.testing.assert_array_equal(fr.species, frames[0].species)


class _Harmonic:
    """Analytic well around a reference structure, for integrator tests."""

    def __init__(self, ref: Frame, k: float = 2.0):
        self.ref = torch.tensor(ref.positions, dtype=torch.float64)
        self.k = k
        from gnff.models import MpnnConfig

        self.config = MpnnConfig(cutoff=4.0)

    def energy_and_forces(self, batch, create_graph=False):
        pos = batch.positions.detach().clone().requires_grad_(True)
        e = 0.5 * self.k * ((pos - self.ref) ** 2).sum()
        (g,) = torch.autograd.grad(e, pos, create_graph=create_graph)
        return e[None], -g


def test_nve_conserves_energy_for_harmonic_well():
    fr = bcc_supercell("TiZrNb", 2, seed=0)
    model = _Harmonic(fr)
    report = run_nve(model, fr, temperature_k=300.0, n_steps=400, timestep_fs=1.0, seed=0)
    assert abs(report["drift_mev_per_atom_per_ps"]) < 0.05
    assert report["e_tot_std_mev_per_atom"] < 0.5


def test_nve_drift_grows_with_bad_timestep():
    """Sanity: a very large timestep degrades conservation measurably."""
    fr = bcc_supercell("TiZrNb", 2, seed=0)
    model = _Harmonic(fr)
    good = run_nve(model, fr, 300.0, n_steps=400, timestep_fs=1.0, seed=0)
    bad = run_nve(model, fr, 300.0, n_steps=400, timestep_fs=8.0, seed=0)
    assert bad["e_tot_std_mev_per_atom"] > good["e_tot_std_mev_per_atom"]
