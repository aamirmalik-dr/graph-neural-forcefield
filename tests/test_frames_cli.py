"""extxyz round trip and CLI smoke tests."""

from __future__ import annotations

import json

import numpy as np
import torch

from gnff.checkpoints import save_checkpoint
from gnff.cli import main as cli_main
from gnff.frames import read_extxyz, write_extxyz
from gnff.models import MpnnConfig, MpnnPotential


def test_extxyz_roundtrip(labeled_frames, tmp_path):
    path = tmp_path / "frames.extxyz"
    write_extxyz(labeled_frames[:5], path)
    back = read_extxyz(path)
    assert len(back) == 5
    for a, b in zip(labeled_frames[:5], back):
        np.testing.assert_array_equal(a.species, b.species)
        np.testing.assert_allclose(a.positions, b.positions, atol=1e-8)
        np.testing.assert_allclose(a.cell, b.cell, atol=1e-8)
        assert abs(a.energy - b.energy) < 1e-8
        np.testing.assert_allclose(a.forces, b.forces, atol=1e-7)
        assert a.group == b.group
        assert a.composition == b.composition


def test_cli_eval_and_predict(labeled_frames, tmp_path, capsys):
    torch.manual_seed(0)
    model = MpnnPotential(MpnnConfig(hidden=8, n_blocks=1, n_rbf=8, cutoff=5.0))
    model.set_reference(labeled_frames[:8])
    ckpt = tmp_path / "model.pt"
    save_checkpoint(model, ckpt)
    data = tmp_path / "frames.extxyz"
    write_extxyz(labeled_frames[:3], data)

    assert cli_main(["eval", "--checkpoint", str(ckpt), "--data", str(data)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert set(out) >= {"e_mae_mev_per_atom", "f_mae_mev_per_a"}

    assert cli_main(["predict", "--checkpoint", str(ckpt), "--structure", str(data)]) == 0
    assert "frame 0" in capsys.readouterr().out


def test_cli_md_smoke(labeled_frames, tmp_path, capsys):
    torch.manual_seed(0)
    model = MpnnPotential(MpnnConfig(hidden=8, n_blocks=1, n_rbf=8, cutoff=5.0))
    model.set_reference(labeled_frames[:8])
    ckpt = tmp_path / "model.pt"
    save_checkpoint(model, ckpt)
    code = cli_main(
        [
            "md",
            "--checkpoint",
            str(ckpt),
            "--composition",
            "Nb",
            "--steps",
            "40",
            "--reps",
            "2",
        ]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert "drift_mev_per_atom_per_ps" in out
