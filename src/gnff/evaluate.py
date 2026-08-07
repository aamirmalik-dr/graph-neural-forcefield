"""Shared evaluation helpers for all potential models.

Models are duck-typed: descriptor models expose .params (with radial and
angular cutoffs) while the message-passing model exposes .config (with a
single graph cutoff). Every model implements energy_and_forces(batch).
"""

from __future__ import annotations

import time

import numpy as np

from gnff.frames import Frame
from gnff.graphs import build_batch


def model_cutoffs(model) -> tuple[float, float | None]:
    """(pair cutoff, angular cutoff or None) for any potential model."""
    if hasattr(model, "params"):
        return model.params.rc_radial, model.params.rc_angular
    return model.config.cutoff, None


def predict(model, frames: list[Frame], chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Predicted energies (S,) and forces (sum N, 3) over a frame list."""
    rc, rca = model_cutoffs(model)
    e_l, f_l = [], []
    for c0 in range(0, len(frames), chunk):
        part = frames[c0 : c0 + chunk]
        batch = build_batch(part, rc, rca)
        energy, forces = model.energy_and_forces(batch)
        e_l.append(energy.detach().numpy())
        f_l.append(forces.detach().numpy())
    return np.concatenate(e_l), np.concatenate(f_l)


def evaluate_potential(model, frames: list[Frame], chunk: int = 32) -> tuple[float, float]:
    """Energy MAE (meV/atom) and force MAE (meV/A) on labeled frames."""
    e_pred, f_pred = predict(model, frames, chunk=chunk)
    n_atoms = np.array([fr.n_atoms for fr in frames])
    e_true = np.array([fr.energy for fr in frames])
    f_true = np.concatenate([fr.forces for fr in frames])
    e_mae = 1000.0 * float(np.mean(np.abs(e_pred - e_true) / n_atoms))
    f_mae = 1000.0 * float(np.mean(np.abs(f_pred - f_true)))
    return e_mae, f_mae


def error_summary(model, frames: list[Frame], chunk: int = 32) -> dict:
    """MAE and RMSE for energies (meV/atom) and forces (meV/A)."""
    e_pred, f_pred = predict(model, frames, chunk=chunk)
    n_atoms = np.array([fr.n_atoms for fr in frames])
    e_true = np.array([fr.energy for fr in frames])
    f_true = np.concatenate([fr.forces for fr in frames])
    de = (e_pred - e_true) / n_atoms
    df = (f_pred - f_true).reshape(-1)
    return {
        "e_mae_mev_per_atom": 1000.0 * float(np.abs(de).mean()),
        "e_rmse_mev_per_atom": 1000.0 * float(np.sqrt((de**2).mean())),
        "f_mae_mev_per_a": 1000.0 * float(np.abs(df).mean()),
        "f_rmse_mev_per_a": 1000.0 * float(np.sqrt((df**2).mean())),
    }


def time_inference(model, frame: Frame, repeats: int = 20) -> float:
    """Mean wall milliseconds for one energy+forces call on one frame."""
    rc, rca = model_cutoffs(model)
    batch = build_batch([frame], rc, rca)
    model.energy_and_forces(batch)  # warm up
    start = time.perf_counter()
    for _ in range(repeats):
        batch = build_batch([frame], rc, rca)
        model.energy_and_forces(batch)
    return 1000.0 * (time.perf_counter() - start) / repeats
