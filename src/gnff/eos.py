"""Equation-of-state benchmark: E-V curves and Birch-Murnaghan fits.

Scans the lattice constant of an ideal BCC supercell, evaluates the model
(and optionally the teacher) at each volume, and fits the third-order
Birch-Murnaghan equation of state to extract the equilibrium lattice
constant a0 and bulk modulus B0. The resulting numbers are compared against
the teacher and against real-DFT Materials Project anchors.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from gnff.evaluate import model_cutoffs
from gnff.frames import Frame
from gnff.graphs import build_batch
from gnff.lattice import bcc_supercell

EV_PER_A3_TO_GPA = 160.21766208


def birch_murnaghan(v: np.ndarray, e0: float, v0: float, b0: float, b0p: float) -> np.ndarray:
    """Third-order Birch-Murnaghan energy per atom as a function of volume."""
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + 9.0 * v0 * b0 / 16.0 * (
        (eta - 1.0) ** 3 * b0p + (eta - 1.0) ** 2 * (6.0 - 4.0 * eta)
    )


def scan_frames(
    composition: str, reps: int = 2, seed: int = 7, scales: np.ndarray | None = None
) -> tuple[list[Frame], np.ndarray]:
    """Isotropically scaled ideal BCC cells for one composition.

    Returns:
        (frames, volumes_per_atom); the same random occupancy is used at
        every volume so the scan is smooth.
    """
    if scales is None:
        scales = np.linspace(0.94, 1.06, 13)
    base = bcc_supercell(composition, reps, seed=seed)
    frames = []
    volumes = []
    for s in scales:
        fr = Frame(
            species=base.species.copy(),
            positions=base.positions * s,
            cell=base.cell * s,
            composition=composition,
        )
        frames.append(fr)
        volumes.append(np.linalg.det(fr.cell) / fr.n_atoms)
    return frames, np.asarray(volumes)


def fit_eos(volumes: np.ndarray, energies_per_atom: np.ndarray) -> dict:
    """Fit Birch-Murnaghan; returns a0 (BCC, Angstrom), B0 (GPa), E0.

    Args:
        volumes: Volume per atom, cubic Angstrom.
        energies_per_atom: Energy per atom, eV.
    """
    k = int(np.argmin(energies_per_atom))
    p0 = [float(energies_per_atom[k]), float(volumes[k]), 1.0, 4.0]
    popt, _ = curve_fit(birch_murnaghan, volumes, energies_per_atom, p0=p0, maxfev=20000)
    e0, v0, b0, b0p = popt
    return {
        "e0_ev_per_atom": float(e0),
        "v0_a3_per_atom": float(v0),
        # BCC: 2 atoms per conventional cell, a0 = (2 v0)^(1/3)
        "a0_angstrom": float((2.0 * v0) ** (1.0 / 3.0)),
        "b0_gpa": float(b0 * EV_PER_A3_TO_GPA),
        "b0_prime": float(b0p),
    }


def model_eos(model, composition: str, reps: int = 2, seed: int = 7) -> dict:
    """EOS of a trained potential for one composition."""
    frames, volumes = scan_frames(composition, reps=reps, seed=seed)
    rc, rca = model_cutoffs(model)
    energies = []
    for fr in frames:
        batch = build_batch([fr], rc, rca)
        e, _ = model.energy_and_forces(batch)
        energies.append(float(e.detach()) / fr.n_atoms)
    fit = fit_eos(volumes, np.asarray(energies))
    fit["volumes"] = volumes.tolist()
    fit["energies_per_atom"] = energies
    return fit


def teacher_eos(calc, composition: str, reps: int = 2, seed: int = 7) -> dict:
    """EOS of the surrogate teacher for the same scan."""
    from gnff.labeling import frame_to_atoms

    frames, volumes = scan_frames(composition, reps=reps, seed=seed)
    energies = []
    for fr in frames:
        atoms = frame_to_atoms(fr)
        atoms.calc = calc
        energies.append(float(atoms.get_potential_energy()) / fr.n_atoms)
    fit = fit_eos(volumes, np.asarray(energies))
    fit["volumes"] = volumes.tolist()
    fit["energies_per_atom"] = energies
    return fit
