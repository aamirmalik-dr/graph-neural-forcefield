"""Compile results/*.json into the summary results/metrics.json.

Usage:
    python scripts/make_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def _load(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def main() -> int:
    provenance = json.loads((REPO / "data" / "provenance.json").read_text())
    teacher = provenance["teacher"]
    main = _load("main")
    eff = _load("data_efficiency")
    gap = _load("split_gap")
    eos = _load("eos")
    nve = _load("nve")
    physics = _load("physics")

    metrics: dict = {
        "label_provenance": {
            "training_labels": (
                f"surrogate model labels from {teacher['package']} {teacher['version']} "
                f"({teacher['checkpoint']}), NOT DFT"
            ),
            "teacher_license": teacher["license"],
            "dft_anchors": "Materials Project API (CC BY 4.0)" if eos else "none",
        },
        "dataset": {
            "n_frames": provenance["n_frames"],
            "n_groups": provenance["n_groups"],
            "atoms_min": provenance["atoms_min"],
            "atoms_max": provenance["atoms_max"],
            "labeling_seconds_per_frame": provenance["labeling_seconds_per_frame"],
        },
    }
    if main:
        metrics["dataset"]["split"] = main["split_sizes"]
        models = {}
        for name in ("mpnn", "acsf_net", "ridge"):
            m = main[name]
            entry = {
                "n_parameters": m["n_parameters"],
                "train_seconds": m["train_seconds"],
                "inference_ms_54atoms": m["inference_ms_54atoms"],
                "test_e_mae_mev_per_atom": round(m["test"]["e_mae_mev_per_atom"], 2),
                "test_f_mae_mev_per_a": round(m["test"]["f_mae_mev_per_a"], 1),
                "test_e_rmse_mev_per_atom": round(m["test"]["e_rmse_mev_per_atom"], 2),
                "test_f_rmse_mev_per_a": round(m["test"]["f_rmse_mev_per_a"], 1),
                "transfer_e_mae_mev_per_atom": round(m["transfer"]["e_mae_mev_per_atom"], 2),
                "transfer_f_mae_mev_per_a": round(m["transfer"]["f_mae_mev_per_a"], 1),
            }
            if "chosen_lr" in m:
                entry["chosen_lr"] = m["chosen_lr"]
            if name == "ridge":
                entry["chosen_alpha"] = m["fit"]["chosen_alpha"]
            models[name] = entry
        metrics["models"] = models
    if eff:
        metrics["data_efficiency"] = {
            "sizes": eff["sizes"],
            "e_mae": {
                k: [round(r["e_mae_mev_per_atom"], 2) for r in eff["results"][k]]
                for k in eff["results"]
            },
            "f_mae": {
                k: [round(r["f_mae_mev_per_a"], 1) for r in eff["results"][k]]
                for k in eff["results"]
            },
        }
    if gap:
        metrics["split_gap"] = {
            name: {
                k: entry[k]
                for k in (
                    "group_e_mae_mean",
                    "group_e_mae_std",
                    "random_frame_e_mae_mean",
                    "random_frame_e_mae_std",
                    "group_f_mae_mean",
                    "group_f_mae_std",
                    "random_frame_f_mae_mean",
                    "random_frame_f_mae_std",
                )
            }
            for name, entry in gap["models"].items()
        }
    if eos:
        metrics["eos"] = {
            comp: {
                k: v
                for k, v in entry.items()
                if k
                in (
                    "a0_model",
                    "a0_teacher",
                    "b0_model_gpa",
                    "b0_teacher_gpa",
                    "a0_anchor_dft",
                    "b0_anchor_dft_gpa",
                    "anchor_id",
                )
            }
            for comp, entry in eos.items()
            if isinstance(entry, dict) and "model" in entry
        }
    if nve:
        metrics["nve"] = [
            {
                k: run[k]
                for k in (
                    "composition",
                    "temperature_k",
                    "steps",
                    "timestep_fs",
                    "drift_mev_per_atom_per_ps",
                    "e_tot_std_mev_per_atom",
                )
            }
            for run in nve["runs"]
        ]
    if physics:
        metrics["physics"] = {
            name: {
                "dimer_max_step_change_mev": round(
                    physics[name]["dimer"]["max_step_change_mev"], 4
                ),
                "dimer_tail_flatness_mev": physics[name]["dimer"]["tail_flatness_mev"],
                "crystal_max_step_change_mev": round(
                    physics[name]["crystal"]["max_step_change_mev"], 3
                ),
                "rotation_delta_mev": physics[name]["invariance"]["rotation_delta_mev"],
                "translation_delta_mev": physics[name]["invariance"]["translation_delta_mev"],
                "permutation_delta_mev": physics[name]["invariance"]["permutation_delta_mev"],
                "force_fd_max_error_ev_per_a": physics[name]["force_consistency"][
                    "max_abs_error_ev_per_a"
                ],
            }
            for name in ("mpnn", "acsf_net")
            if name in physics
        }
    walls = {}
    for name in ("main", "data_efficiency", "split_gap", "eos", "nve", "physics"):
        data = _load(name)
        if data and "wall_seconds" in data:
            walls[name] = data["wall_seconds"]
    metrics["wall_seconds"] = walls

    out = RESULTS / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
