"""Training loop for the neural potentials.

One loop serves both the message-passing and the descriptor network so the
head-to-head benchmark is a matched-budget comparison: same data, same
optimizer family, same epoch and batch budget, same weighted
energy-plus-force loss. Validation error selects the reported model state.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from gnff.evaluate import evaluate_potential, model_cutoffs
from gnff.frames import Frame
from gnff.graphs import BatchCache
from gnff.models.acsf import AcsfNet, compute_acsf


@dataclass
class TrainSettings:
    """Optimization budget and loss weighting.

    Attributes:
        epochs: Full passes over the training frames.
        batch_size: Frames per minibatch.
        lr: Adam learning rate.
        force_weight: Weight of the force MSE term relative to the per-atom
            energy MSE term (eV and eV/A units).
        weight_decay: Adam weight decay.
        val_every: Epochs between validation evaluations.
        seed: Torch and shuffle seed.
    """

    epochs: int = 80
    batch_size: int = 24
    lr: float = 3e-3
    force_weight: float = 0.1
    weight_decay: float = 0.0
    val_every: int = 5
    seed: int = 0


def fit_acsf_scaler(model: AcsfNet, cache: BatchCache, chunk: int = 48) -> None:
    """Fit the descriptor scaler of an AcsfNet on cached training frames."""
    descs = []
    with torch.no_grad():
        for c0 in range(0, len(cache.frames), chunk):
            batch = cache.assemble(list(range(c0, min(c0 + chunk, len(cache.frames)))))
            descs.append(compute_acsf(batch.positions, batch, model.params))
    model.fit_scaler(torch.cat(descs))


def train_potential(
    model,
    train_frames: list[Frame],
    val_frames: list[Frame],
    settings: TrainSettings,
    log_every: int = 10,
) -> dict:
    """Train a neural potential with a weighted energy+force loss.

    Args:
        model: MpnnPotential or AcsfNet.
        train_frames: Labeled training frames.
        val_frames: Labeled validation frames (model selection only).
        settings: Budget and loss weighting.
        log_every: Epochs between progress prints.

    Returns:
        Report dict with history, best epoch, and wall seconds.
    """
    torch.manual_seed(settings.seed)
    rng = np.random.default_rng(settings.seed)
    rc, rca = model_cutoffs(model)
    cache = BatchCache(train_frames, rc, rca)
    model.set_reference(train_frames)
    if isinstance(model, AcsfNet):
        fit_acsf_scaler(model, cache)

    energies = torch.tensor([fr.energy for fr in train_frames], dtype=torch.float64)
    n_atoms = torch.tensor([fr.n_atoms for fr in train_frames], dtype=torch.float64)
    forces = [torch.tensor(fr.forces, dtype=torch.float64) for fr in train_frames]

    opt = torch.optim.Adam(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    history = []
    best = None
    start = time.perf_counter()
    for epoch in range(1, settings.epochs + 1):
        order = rng.permutation(len(train_frames))
        epoch_loss = 0.0
        n_batches = 0
        for b0 in range(0, len(order), settings.batch_size):
            idx = order[b0 : b0 + settings.batch_size].tolist()
            batch = cache.assemble(idx)
            e_pred, f_pred = model.energy_and_forces(batch, create_graph=True)
            e_true = energies[idx]
            f_true = torch.cat([forces[k] for k in idx])
            na = n_atoms[idx]
            e_loss = (((e_pred - e_true) / na) ** 2).mean()
            f_loss = ((f_pred - f_true) ** 2).mean()
            loss = e_loss + settings.force_weight * f_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        record = {"epoch": epoch, "train_loss": epoch_loss / max(1, n_batches)}
        if epoch % settings.val_every == 0 or epoch == settings.epochs:
            e_mae, f_mae = evaluate_potential(model, val_frames)
            record["val_e_mae"] = e_mae
            record["val_f_mae"] = f_mae
            score = (e_mae / 1000.0) ** 2 + settings.force_weight * (f_mae / 1000.0) ** 2
            if best is None or score < best[0]:
                best = (score, epoch, copy.deepcopy(model.state_dict()))
        history.append(record)
        if epoch % log_every == 0 or epoch == settings.epochs:
            msg = f"epoch {epoch}/{settings.epochs} loss {record['train_loss']:.5f}"
            if "val_e_mae" in record:
                msg += (
                    f" val E {record['val_e_mae']:.2f} meV/atom F {record['val_f_mae']:.1f} meV/A"
                )
            print(msg, flush=True)
    seconds = time.perf_counter() - start
    if best is not None:
        model.load_state_dict(best[2])
    return {
        "history": history,
        "best_epoch": None if best is None else best[1],
        "train_seconds": round(seconds, 1),
        "settings": settings.__dict__,
    }
