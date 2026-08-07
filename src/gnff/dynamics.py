"""Velocity-Verlet NVE molecular dynamics with a trained potential.

A from-scratch microcanonical integrator used for the stability benchmark:
if the learned potential energy surface is smooth and the forces are its
exact gradient (autograd guarantees this), total energy should be conserved
up to integrator error. The benchmark reports the linear drift of total
energy in meV/atom/ps and its standard deviation.
"""

from __future__ import annotations

import numpy as np
import torch

from gnff.evaluate import model_cutoffs
from gnff.frames import Frame
from gnff.graphs import build_batch

# ASE-compatible units: eV, Angstrom, amu; time unit then is
# 1 internal = 1 A sqrt(amu/eV) = 10.1805 fs.
MASSES_AMU = {0: 47.867, 1: 91.224, 2: 92.90637}  # Ti, Zr, Nb
FS_PER_INTERNAL = 10.180505


def model_energy_forces(model, frame: Frame) -> tuple[float, np.ndarray]:
    """One energy+forces call of a trained potential on one frame."""
    rc, rca = model_cutoffs(model)
    batch = build_batch([frame], rc, rca)
    with torch.enable_grad():
        e, f = model.energy_and_forces(batch)
    return float(e.detach()), f.detach().numpy()


def run_nve(
    model,
    frame: Frame,
    temperature_k: float,
    n_steps: int,
    timestep_fs: float = 2.0,
    seed: int = 0,
) -> dict:
    """NVE run with Maxwell-Boltzmann initial velocities.

    Args:
        model: Trained potential.
        frame: Starting structure (positions are copied).
        temperature_k: Initial velocity temperature.
        n_steps: Number of velocity-Verlet steps.
        timestep_fs: Timestep in femtoseconds.
        seed: Velocity seed.

    Returns:
        Report with per-sample total energies and the drift statistics.
    """
    rng = np.random.default_rng(seed)
    kb = 8.617333262e-5  # eV/K
    n = frame.n_atoms
    masses = np.array([MASSES_AMU[int(s)] for s in frame.species])[:, None]
    velocities = rng.normal(0.0, 1.0, (n, 3)) * np.sqrt(kb * temperature_k / masses)
    velocities -= velocities.mean(axis=0)

    fr = Frame(
        species=frame.species.copy(),
        positions=frame.positions.copy(),
        cell=frame.cell.copy(),
        composition=frame.composition,
    )
    dt = timestep_fs / FS_PER_INTERNAL
    energy, forces = model_energy_forces(model, fr)
    times_ps, e_tot = [], []
    for step in range(n_steps):
        velocities = velocities + 0.5 * dt * forces / masses
        fr.positions = fr.positions + dt * velocities
        energy, forces = model_energy_forces(model, fr)
        velocities = velocities + 0.5 * dt * forces / masses
        if step % 10 == 0:
            kinetic = 0.5 * float((masses * velocities**2).sum())
            times_ps.append((step + 1) * timestep_fs / 1000.0)
            e_tot.append((energy + kinetic) / n)
    times = np.asarray(times_ps)
    e_tot = np.asarray(e_tot)
    slope = float(np.polyfit(times, e_tot, 1)[0])
    return {
        "temperature_k": temperature_k,
        "steps": n_steps,
        "timestep_fs": timestep_fs,
        "times_ps": times.tolist(),
        "e_tot_ev_per_atom": e_tot.tolist(),
        "drift_mev_per_atom_per_ps": round(1000.0 * slope, 4),
        "e_tot_std_mev_per_atom": round(1000.0 * float(e_tot.std()), 4),
    }
