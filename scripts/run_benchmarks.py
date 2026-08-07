"""Run the fixed-seed benchmark suite.

Usage:
    python scripts/run_benchmarks.py --mode main
    python scripts/run_benchmarks.py --mode data_efficiency
    python scripts/run_benchmarks.py --mode split_gap
    python scripts/run_benchmarks.py --mode eos
    python scripts/run_benchmarks.py --mode nve
    python scripts/run_benchmarks.py --mode physics

Each mode reads its YAML config from configs/, runs on the generated dataset
in data/full/dataset.extxyz, and writes one JSON to results/. All seeds are
fixed in the configs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "data" / "full" / "dataset.extxyz"
RESULTS = REPO / "results"
MODELS = REPO / "models"


def _load(config_name: str) -> dict:
    return yaml.safe_load((REPO / "configs" / f"{config_name}.yaml").read_text())


def _dataset():
    from gnff.frames import read_extxyz

    frames = read_extxyz(DATASET)
    print(f"loaded {len(frames)} frames from {DATASET}", flush=True)
    return frames


def _sets(frames, split):
    return (
        [frames[k] for k in split.train],
        [frames[k] for k in split.val],
        [frames[k] for k in split.test],
        [frames[k] for k in split.transfer],
    )


def _fresh_net(name: str, cfg: dict):
    from gnff.models import AcsfNet, MpnnConfig, MpnnPotential

    torch.manual_seed(int(cfg.get("train", {}).get("seed", 0)))
    if name == "mpnn":
        m = cfg["mpnn"]
        return MpnnPotential(
            MpnnConfig(
                hidden=int(m["hidden"]),
                n_blocks=int(m["n_blocks"]),
                n_rbf=int(m["n_rbf"]),
                cutoff=float(m["cutoff"]),
            )
        )
    return AcsfNet(hidden=tuple(cfg["acsf_hidden"]))


def _settings(cfg: dict, lr: float, epochs: int | None = None):
    from gnff.train import TrainSettings

    t = cfg["train"]
    return TrainSettings(
        epochs=int(epochs if epochs is not None else t["epochs"]),
        batch_size=int(t["batch_size"]),
        lr=float(lr),
        force_weight=float(t["force_weight"]),
        weight_decay=float(t.get("weight_decay", 0.0)),
        val_every=int(t["val_every"]),
        seed=int(t["seed"]),
    )


def _screen_lr(name: str, cfg: dict, train, val) -> tuple[float, list[dict]]:
    """Short validation screen over the shared learning-rate grid."""
    from gnff.evaluate import evaluate_potential
    from gnff.train import train_potential

    fw = float(cfg["train"]["force_weight"])
    records = []
    best = None
    for lr in cfg["lr_grid"]:
        model = _fresh_net(name, cfg)
        train_potential(model, train, val, _settings(cfg, lr, epochs=cfg["screen_epochs"]))
        e_mae, f_mae = evaluate_potential(model, val)
        score = (e_mae / 1000.0) ** 2 + fw * (f_mae / 1000.0) ** 2
        records.append({"lr": lr, "val_e_mae": e_mae, "val_f_mae": f_mae})
        print(f"  screen {name} lr={lr}: val E {e_mae:.2f} F {f_mae:.1f}", flush=True)
        if best is None or score < best[0]:
            best = (score, lr)
    return best[1], records


def _parity_payload(model, frames, force_sample: int = 1500, seed: int = 0) -> dict:
    """True/predicted energies per frame and a fixed subsample of force components."""
    from gnff.evaluate import predict

    e_pred, f_pred = predict(model, frames)
    n_atoms = np.array([fr.n_atoms for fr in frames])
    e_true = np.array([fr.energy for fr in frames])
    f_true = np.concatenate([fr.forces for fr in frames]).reshape(-1)
    f_pred = f_pred.reshape(-1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f_true), size=min(force_sample, len(f_true)), replace=False)
    return {
        "e_true_per_atom": (e_true / n_atoms).tolist(),
        "e_pred_per_atom": (e_pred / n_atoms).tolist(),
        "f_true_sample": f_true[idx].tolist(),
        "f_pred_sample": f_pred[idx].tolist(),
    }


def run_main() -> dict:
    from gnff.checkpoints import save_checkpoint
    from gnff.evaluate import error_summary, time_inference
    from gnff.lattice import bcc_supercell
    from gnff.models import RidgePotential
    from gnff.splits import group_split
    from gnff.train import train_potential

    cfg = _load("main")
    frames = _dataset()
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=int(cfg["split_seed"]),
    )
    train, val, test, transfer = _sets(frames, split)
    print(
        f"split: train {len(train)} val {len(val)} test {len(test)} transfer {len(transfer)}",
        flush=True,
    )
    timing_frame = bcc_supercell("TiZrNb", 3, seed=123)
    rng = np.random.default_rng(123)
    timing_frame.positions = timing_frame.positions + rng.normal(0, 0.08, (54, 3))

    out: dict = {
        "split_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "transfer": len(transfer),
        }
    }
    start_all = time.perf_counter()
    for name in ("mpnn", "acsf_net"):
        print(f"=== {name} ===", flush=True)
        lr, screen = _screen_lr(name, cfg, train, val)
        model = _fresh_net(name, cfg)
        t0 = time.perf_counter()
        report = train_potential(model, train, val, _settings(cfg, lr))
        seconds = time.perf_counter() - t0
        save_checkpoint(model, MODELS / f"{name}.pt")
        out[name] = {
            "chosen_lr": lr,
            "lr_screen": screen,
            "n_parameters": model.n_parameters(),
            "train_seconds": round(seconds, 1),
            "best_epoch": report["best_epoch"],
            "history": report["history"],
            "test": error_summary(model, test),
            "transfer": error_summary(model, transfer),
            "inference_ms_54atoms": round(
                time_inference(model, timing_frame, repeats=int(cfg["timing_repeats"])), 2
            ),
            "parity_test": _parity_payload(model, test),
        }
        print(json.dumps(out[name]["test"], indent=2), flush=True)

    print("=== ridge ===", flush=True)
    ridge = RidgePotential()
    t0 = time.perf_counter()
    fit_report = ridge.fit(
        train,
        val,
        alphas=tuple(float(a) for a in cfg["ridge_alphas"]),
        force_weight=float(cfg["train"]["force_weight"]),
    )
    seconds = time.perf_counter() - t0
    save_checkpoint(ridge, MODELS / "ridge.pt")
    out["ridge"] = {
        "fit": fit_report,
        "n_parameters": ridge.n_parameters(),
        "train_seconds": round(seconds, 1),
        "test": error_summary(ridge, test),
        "transfer": error_summary(ridge, transfer),
        "inference_ms_54atoms": round(
            time_inference(ridge, timing_frame, repeats=int(cfg["timing_repeats"])), 2
        ),
        "parity_test": _parity_payload(ridge, test),
    }
    print(json.dumps(out["ridge"]["test"], indent=2), flush=True)
    out["wall_seconds"] = round(time.perf_counter() - start_all, 1)
    return out


def run_data_efficiency() -> dict:
    from gnff.evaluate import evaluate_potential
    from gnff.models import RidgePotential
    from gnff.splits import group_split
    from gnff.train import train_potential

    cfg = _load("data_efficiency")
    frames = _dataset()
    split = group_split(
        frames,
        holdout_compositions=tuple(cfg["holdout_compositions"]),
        seed=int(cfg["split_seed"]),
    )
    train, val, test, _ = _sets(frames, split)

    main_path = RESULTS / "main.json"
    lrs = {"mpnn": 1e-3, "acsf_net": 3e-3}
    if main_path.exists():
        main = json.loads(main_path.read_text())
        lrs = {k: main[k]["chosen_lr"] for k in ("mpnn", "acsf_net")}
    print(f"using learning rates from main benchmark: {lrs}", flush=True)

    # Nested subsets of WHOLE training groups so every size stays leakage-safe.
    rng = np.random.default_rng(int(cfg["subset_seed"]))
    groups = sorted({fr.group for fr in train})
    rng.shuffle(groups)
    by_group: dict[str, list] = {}
    for fr in train:
        by_group.setdefault(fr.group, []).append(fr)

    out: dict = {"sizes": [], "results": {m: [] for m in ("mpnn", "acsf_net", "ridge")}}
    start_all = time.perf_counter()
    for size in cfg["sizes"]:
        target = len(train) if size == "all" else int(size)
        subset, used = [], []
        for g in groups:
            if len(subset) >= target:
                break
            subset.extend(by_group[g])
            used.append(g)
        n = len(subset)
        out["sizes"].append(n)
        print(f"--- size {size}: {n} frames from {len(used)} groups ---", flush=True)
        for name in ("mpnn", "acsf_net"):
            model = _fresh_net(name, cfg)
            t0 = time.perf_counter()
            train_potential(model, subset, val, _settings(cfg, lrs[name]), log_every=100)
            e_mae, f_mae = evaluate_potential(model, test)
            out["results"][name].append(
                {
                    "n_train": n,
                    "e_mae_mev_per_atom": e_mae,
                    "f_mae_mev_per_a": f_mae,
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
            print(f"  {name}: E {e_mae:.2f} F {f_mae:.1f}", flush=True)
        ridge = RidgePotential()
        t0 = time.perf_counter()
        ridge.fit(
            subset,
            val,
            alphas=tuple(float(a) for a in cfg["ridge_alphas"]),
            force_weight=float(cfg["train"]["force_weight"]),
        )
        e_mae, f_mae = evaluate_potential(ridge, test)
        out["results"]["ridge"].append(
            {
                "n_train": n,
                "e_mae_mev_per_atom": e_mae,
                "f_mae_mev_per_a": f_mae,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        print(f"  ridge: E {e_mae:.2f} F {f_mae:.1f}", flush=True)
    out["wall_seconds"] = round(time.perf_counter() - start_all, 1)
    return out


def run_split_gap() -> dict:
    from gnff.evaluate import evaluate_potential
    from gnff.splits import group_split, random_frame_split
    from gnff.train import train_potential

    cfg = _load("split_gap")
    frames = _dataset()
    model_names = list(cfg.get("models", ["mpnn"]))
    main_path = RESULTS / "main.json"
    lrs = {name: 3e-3 for name in model_names}
    if main_path.exists():
        main = json.loads(main_path.read_text())
        lrs = {name: main[name]["chosen_lr"] for name in model_names}

    out: dict = {"models": {}, "learning_rates": lrs}
    start_all = time.perf_counter()
    for name in model_names:
        protocols: dict = {"group": [], "random_frame": []}
        for protocol, splitter in (("group", group_split), ("random_frame", random_frame_split)):
            for seed in cfg["split_seeds"]:
                split = splitter(
                    frames,
                    holdout_compositions=tuple(cfg["holdout_compositions"]),
                    seed=int(seed),
                )
                train, val, test, _ = _sets(frames, split)
                model = _fresh_net(name, cfg)
                train_potential(model, train, val, _settings(cfg, lrs[name]), log_every=100)
                e_mae, f_mae = evaluate_potential(model, test)
                protocols[protocol].append(
                    {
                        "split_seed": seed,
                        "n_test": len(test),
                        "e_mae_mev_per_atom": e_mae,
                        "f_mae_mev_per_a": f_mae,
                    }
                )
                print(f"{name} {protocol} seed {seed}: E {e_mae:.2f} F {f_mae:.1f}", flush=True)
        entry: dict = {"protocols": protocols}
        for protocol, runs in protocols.items():
            entry[f"{protocol}_e_mae_mean"] = round(
                float(np.mean([r["e_mae_mev_per_atom"] for r in runs])), 2
            )
            entry[f"{protocol}_e_mae_std"] = round(
                float(np.std([r["e_mae_mev_per_atom"] for r in runs])), 2
            )
            entry[f"{protocol}_f_mae_mean"] = round(
                float(np.mean([r["f_mae_mev_per_a"] for r in runs])), 1
            )
            entry[f"{protocol}_f_mae_std"] = round(
                float(np.std([r["f_mae_mev_per_a"] for r in runs])), 1
            )
        out["models"][name] = entry
    out["wall_seconds"] = round(time.perf_counter() - start_all, 1)
    return out


def run_eos() -> dict:
    from gnff.checkpoints import load_checkpoint
    from gnff.eos import model_eos, teacher_eos
    from gnff.labeling import get_teacher

    cfg = _load("eos")
    model = load_checkpoint(REPO / cfg["checkpoint"])
    calc, _ = get_teacher(cfg["teacher"])
    anchors = {}
    for candidate in ("anchors_mp.json", "anchors_literature.json"):
        path = REPO / "data" / candidate
        if path.exists():
            anchors = json.loads(path.read_text())
            print(f"anchors from {candidate}", flush=True)
            break
    out: dict = {"anchor_source": anchors.get("source", "none")}
    start_all = time.perf_counter()
    for comp in cfg["compositions"]:
        m = model_eos(model, comp, reps=int(cfg["reps"]), seed=int(cfg["seed"]))
        t = teacher_eos(calc, comp, reps=int(cfg["reps"]), seed=int(cfg["seed"]))
        entry = {
            "model": m,
            "teacher": t,
            "a0_model": round(m["a0_angstrom"], 3),
            "a0_teacher": round(t["a0_angstrom"], 3),
            "b0_model_gpa": round(m["b0_gpa"], 1),
            "b0_teacher_gpa": round(t["b0_gpa"], 1),
        }
        if comp in anchors.get("anchors", {}):
            a = anchors["anchors"][comp]
            entry["a0_anchor_dft"] = a["a0_angstrom"]
            entry["b0_anchor_dft_gpa"] = a["b0_gpa"]
            entry["anchor_id"] = a.get("material_id", a.get("citation", ""))
        out[comp] = entry
        print(
            f"{comp}: a0 model {entry['a0_model']} teacher {entry['a0_teacher']}"
            f" B0 model {entry['b0_model_gpa']} teacher {entry['b0_teacher_gpa']}",
            flush=True,
        )
    out["wall_seconds"] = round(time.perf_counter() - start_all, 1)
    return out


def run_nve() -> dict:
    from gnff.checkpoints import load_checkpoint
    from gnff.dynamics import run_nve as nve
    from gnff.lattice import bcc_supercell

    cfg = _load("nve")
    model = load_checkpoint(REPO / cfg["checkpoint"])
    out: dict = {"runs": []}
    start_all = time.perf_counter()
    for spec in cfg["runs"]:
        frame = bcc_supercell(spec["composition"], int(spec["reps"]), seed=int(spec["seed"]))
        report = nve(
            model,
            frame,
            temperature_k=float(spec["temperature_k"]),
            n_steps=int(spec["steps"]),
            timestep_fs=float(spec["timestep_fs"]),
            seed=int(spec["seed"]),
        )
        report["composition"] = spec["composition"]
        report["n_atoms"] = frame.n_atoms
        out["runs"].append(report)
        print(
            f"{spec['composition']} {spec['temperature_k']} K: drift "
            f"{report['drift_mev_per_atom_per_ps']} meV/atom/ps",
            flush=True,
        )
    out["wall_seconds"] = round(time.perf_counter() - start_all, 1)
    return out


def run_physics() -> dict:
    from gnff.checkpoints import load_checkpoint
    from gnff.physics import (
        crystal_crossing_scan,
        dimer_scan,
        force_consistency,
        invariance_report,
    )

    cfg = _load("physics")
    out: dict = {}
    start_all = time.perf_counter()
    for name, rel in cfg["checkpoints"].items():
        model = load_checkpoint(REPO / rel)
        out[name] = {
            "dimer": dimer_scan(
                model, span=float(cfg["dimer_span"]), step=float(cfg["dimer_step"])
            ),
            "crystal": crystal_crossing_scan(
                model, span=float(cfg["crystal_span"]), step=float(cfg["crystal_step"])
            ),
            "invariance": invariance_report(model),
            "force_consistency": force_consistency(model),
        }
        print(
            f"{name}: dimer max step {out[name]['dimer']['max_step_change_mev']:.4f} meV, "
            f"rotation delta {out[name]['invariance']['rotation_delta_mev']:.2e} meV",
            flush=True,
        )
    out["wall_seconds"] = round(time.perf_counter() - start_all, 1)
    return out


MODES = {
    "main": run_main,
    "data_efficiency": run_data_efficiency,
    "split_gap": run_split_gap,
    "eos": run_eos,
    "nve": run_nve,
    "physics": run_physics,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    out = MODES[args.mode]()
    path = RESULTS / f"{args.mode}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
