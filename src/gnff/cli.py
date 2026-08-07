"""Command-line interface: generate, train, eval, predict, eos, md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def _cmd_generate(args: argparse.Namespace) -> int:
    from gnff.dataset import build_dataset
    from gnff.frames import write_extxyz

    cfg = yaml.safe_load(Path(args.config).read_text())
    frames, provenance = build_dataset(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_extxyz(frames, out)
    Path(args.provenance).write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: v for k, v in provenance.items() if k != "config"}, indent=2))
    return 0


def _build_model(name: str, cfg: dict):
    from gnff.models import AcsfNet, MpnnConfig, MpnnPotential, RidgePotential

    if name == "mpnn":
        m = cfg.get("mpnn", {})
        return MpnnPotential(
            MpnnConfig(
                hidden=int(m.get("hidden", 48)),
                n_blocks=int(m.get("n_blocks", 3)),
                n_rbf=int(m.get("n_rbf", 24)),
                cutoff=float(m.get("cutoff", 5.0)),
            )
        )
    if name == "acsf_net":
        return AcsfNet(hidden=tuple(cfg.get("acsf_hidden", [32, 32])))
    if name == "ridge":
        return RidgePotential()
    raise ValueError(f"unknown model {name}")


def _cmd_train(args: argparse.Namespace) -> int:
    from gnff.checkpoints import save_checkpoint
    from gnff.evaluate import error_summary
    from gnff.frames import read_extxyz
    from gnff.models import RidgePotential
    from gnff.splits import group_split
    from gnff.train import TrainSettings, train_potential

    cfg = yaml.safe_load(Path(args.config).read_text())
    frames = read_extxyz(args.data)
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg.get("holdout_compositions", [])),
        seed=int(cfg.get("split_seed", 0)),
    )
    train = [frames[k] for k in split.train]
    val = [frames[k] for k in split.val]
    test = [frames[k] for k in split.test]
    model = _build_model(args.model, cfg)
    if isinstance(model, RidgePotential):
        report = model.fit(
            train,
            val,
            alphas=tuple(cfg.get("ridge_alphas", [1e-6, 1e-4, 1e-2])),
            force_weight=float(cfg.get("train", {}).get("force_weight", 0.1)),
        )
    else:
        t = cfg.get("train", {})
        settings = TrainSettings(
            epochs=int(t.get("epochs", 80)),
            batch_size=int(t.get("batch_size", 24)),
            lr=float(t.get("lr", 3e-3)),
            force_weight=float(t.get("force_weight", 0.1)),
            weight_decay=float(t.get("weight_decay", 0.0)),
            val_every=int(t.get("val_every", 5)),
            seed=int(t.get("seed", 0)),
        )
        report = train_potential(model, train, val, settings)
    save_checkpoint(model, args.out)
    summary = error_summary(model, test)
    print(json.dumps({"test": summary, "report_keys": sorted(report)}, indent=2))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from gnff.checkpoints import load_checkpoint
    from gnff.evaluate import error_summary
    from gnff.frames import read_extxyz

    model = load_checkpoint(args.checkpoint)
    frames = [fr for fr in read_extxyz(args.data) if fr.energy is not None]
    print(json.dumps(error_summary(model, frames), indent=2))
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from gnff.checkpoints import load_checkpoint
    from gnff.evaluate import predict
    from gnff.frames import read_extxyz

    model = load_checkpoint(args.checkpoint)
    frames = read_extxyz(args.structure)
    energies, forces = predict(model, frames)
    off = 0
    for k, fr in enumerate(frames):
        fmax = float(np.abs(forces[off : off + fr.n_atoms]).max())
        off += fr.n_atoms
        print(
            f"frame {k}: {fr.n_atoms} atoms, E = {energies[k]:.4f} eV "
            f"({energies[k] / fr.n_atoms:.4f} eV/atom), max|F| = {fmax:.3f} eV/A"
        )
    return 0


def _cmd_eos(args: argparse.Namespace) -> int:
    from gnff.checkpoints import load_checkpoint
    from gnff.eos import model_eos

    model = load_checkpoint(args.checkpoint)
    fit = model_eos(model, args.composition, reps=args.reps)
    print(
        json.dumps(
            {k: v for k, v in fit.items() if k not in ("volumes", "energies_per_atom")},
            indent=2,
        )
    )
    return 0


def _cmd_md(args: argparse.Namespace) -> int:
    from gnff.checkpoints import load_checkpoint
    from gnff.dynamics import run_nve
    from gnff.lattice import bcc_supercell

    model = load_checkpoint(args.checkpoint)
    frame = bcc_supercell(args.composition, args.reps, seed=args.seed)
    report = run_nve(
        model,
        frame,
        temperature_k=args.temperature,
        n_steps=args.steps,
        timestep_fs=args.timestep,
        seed=args.seed,
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("times_ps", "e_tot_ev_per_atom")},
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gnff", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="generate and teacher-label the dataset")
    p.add_argument("--config", default="configs/dataset.yaml")
    p.add_argument("--out", default="data/full/dataset.extxyz")
    p.add_argument("--provenance", default="data/provenance.json")
    p.set_defaults(func=_cmd_generate)

    p = sub.add_parser("train", help="train a potential on a labeled extxyz dataset")
    p.add_argument("--data", required=True)
    p.add_argument("--model", choices=["mpnn", "acsf_net", "ridge"], default="mpnn")
    p.add_argument("--config", default="configs/main.yaml")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="evaluate a checkpoint on labeled frames")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("predict", help="energies and forces for structures in an extxyz file")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--structure", required=True)
    p.set_defaults(func=_cmd_predict)

    p = sub.add_parser("eos", help="equation of state of a trained potential")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--composition", default="Nb")
    p.add_argument("--reps", type=int, default=2)
    p.set_defaults(func=_cmd_eos)

    p = sub.add_parser("md", help="NVE stability run with a trained potential")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--composition", default="TiZrNb")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--timestep", type=float, default=2.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_md)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
