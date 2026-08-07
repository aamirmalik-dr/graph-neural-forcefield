"""Physics sanity checks: cutoff continuity and symmetry invariances.

A potential that jumps as a neighbor crosses the cutoff sphere cannot
conserve energy in MD; a potential that changes under translation,
rotation, or atom permutation is not a function of the physical structure.
Both properties are verified numerically here (and enforced by unit tests).
"""

from __future__ import annotations

import numpy as np

from gnff.evaluate import model_cutoffs
from gnff.frames import Frame
from gnff.graphs import build_batch
from gnff.lattice import bcc_supercell


def _energy(model, frame: Frame) -> float:
    rc, rca = model_cutoffs(model)
    batch = build_batch([frame], rc, rca)
    e, _ = model.energy_and_forces(batch)
    return float(e.detach())


def dimer_scan(model, span: float = 0.6, step: float = 0.005) -> dict:
    """Energy of an isolated Ti-Nb dimer as the bond crosses the cutoff.

    The dimer sits in a large box so no periodic image is within range.

    Returns:
        Report with the scan, the largest energy change between consecutive
        points near the cutoff, and the tail flatness beyond the cutoff.
    """
    rc, _ = model_cutoffs(model)
    box = 4.0 * rc
    distances = np.arange(rc - span, rc + span, step)
    energies = []
    for d in distances:
        fr = Frame(
            species=np.array([0, 2]),
            positions=np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]),
            cell=np.eye(3) * box,
            composition="TiNb",
        )
        energies.append(_energy(model, fr))
    energies = np.asarray(energies)
    inside = distances < rc
    jumps = np.abs(np.diff(energies))
    tail = energies[~inside]
    return {
        "cutoff": rc,
        "distances": distances.tolist(),
        "energies_ev": energies.tolist(),
        "max_step_change_mev": float(1000.0 * jumps.max()),
        "step_angstrom": step,
        "tail_flatness_mev": float(1000.0 * (tail.max() - tail.min())),
    }


def crystal_crossing_scan(
    model, composition: str = "TiZrNb", reps: int = 2, span: float = 0.5, step: float = 0.005
) -> dict:
    """Energy of a BCC cell as one displaced atom's neighbors cross the cutoff.

    One atom is dragged along [100]; many neighbor distances sweep through
    the cutoff during the scan. The maximum energy change between
    consecutive points, normalized by the step, bounds any discontinuity.
    """
    base = bcc_supercell(composition, reps, seed=11)
    displacements = np.arange(-span, span, step)
    energies = []
    for d in displacements:
        fr = Frame(
            species=base.species.copy(),
            positions=base.positions.copy(),
            cell=base.cell.copy(),
            composition=composition,
        )
        fr.positions[0, 0] += d
        energies.append(_energy(model, fr))
    energies = np.asarray(energies)
    jumps = np.abs(np.diff(energies))
    return {
        "displacements": displacements.tolist(),
        "energies_ev": energies.tolist(),
        "max_step_change_mev": float(1000.0 * jumps.max()),
        "median_step_change_mev": float(1000.0 * np.median(jumps)),
        "step_angstrom": step,
    }


def invariance_report(model, composition: str = "TiZrNb", reps: int = 2, seed: int = 3) -> dict:
    """Max energy change under translation, rotation, and permutation."""
    rng = np.random.default_rng(seed)
    base = bcc_supercell(composition, reps, seed=seed)
    base.positions = base.positions + rng.normal(0.0, 0.08, base.positions.shape)
    e0 = _energy(model, base)

    shifted = Frame(
        species=base.species.copy(),
        positions=base.positions + rng.uniform(-3.0, 3.0, 3),
        cell=base.cell.copy(),
    )
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    rotated = Frame(
        species=base.species.copy(),
        positions=base.positions @ q.T,
        cell=base.cell @ q.T,
    )
    perm = rng.permutation(base.n_atoms)
    permuted = Frame(
        species=base.species[perm].copy(),
        positions=base.positions[perm].copy(),
        cell=base.cell.copy(),
    )
    return {
        "energy_ev": e0,
        "translation_delta_mev": float(1000.0 * abs(_energy(model, shifted) - e0)),
        "rotation_delta_mev": float(1000.0 * abs(_energy(model, rotated) - e0)),
        "permutation_delta_mev": float(1000.0 * abs(_energy(model, permuted) - e0)),
    }


def force_consistency(model, composition: str = "TiZrNb", reps: int = 2, seed: int = 5) -> dict:
    """Compare autograd forces against central finite differences."""
    rng = np.random.default_rng(seed)
    base = bcc_supercell(composition, reps, seed=seed)
    base.positions = base.positions + rng.normal(0.0, 0.08, base.positions.shape)
    rc, rca = model_cutoffs(model)
    batch = build_batch([base], rc, rca)
    _, forces = model.energy_and_forces(batch)
    forces = forces.detach().numpy()
    eps = 1e-4
    max_err = 0.0
    for atom in (0, base.n_atoms // 2):
        for axis in range(3):
            plus = Frame(base.species.copy(), base.positions.copy(), base.cell.copy())
            minus = Frame(base.species.copy(), base.positions.copy(), base.cell.copy())
            plus.positions[atom, axis] += eps
            minus.positions[atom, axis] -= eps
            fd = -(_energy(model, plus) - _energy(model, minus)) / (2 * eps)
            max_err = max(max_err, abs(fd - forces[atom, axis]))
    return {"max_abs_error_ev_per_a": float(max_err), "eps": eps}
