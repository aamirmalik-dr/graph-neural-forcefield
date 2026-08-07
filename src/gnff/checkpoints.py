"""Checkpoint persistence for all potential models."""

from __future__ import annotations

from pathlib import Path

import torch

from gnff.models.acsf import AcsfNet
from gnff.models.mpnn import MpnnPotential
from gnff.models.ridge import RidgePotential


def save_checkpoint(model, path: str | Path) -> None:
    """Save any potential model to a torch checkpoint file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.to_checkpoint(), str(path))


def load_checkpoint(path: str | Path):
    """Load a potential model saved by save_checkpoint."""
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    kind = ckpt["kind"]
    if kind == "mpnn":
        return MpnnPotential.from_checkpoint(ckpt)
    if kind == "acsf_net":
        return AcsfNet.from_checkpoint(ckpt)
    if kind == "ridge":
        return RidgePotential.from_checkpoint(ckpt)
    raise ValueError(f"unknown checkpoint kind {kind}")
