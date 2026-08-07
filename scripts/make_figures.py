"""Rebuild every committed figure from the results JSONs.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGDIR = RESULTS / "figures"

# One fixed color per model across every figure (validated palette).
COLORS = {"mpnn": "#2a78d6", "acsf_net": "#eb6834", "ridge": "#1baf7a"}
LABELS = {"mpnn": "message passing (GNN)", "acsf_net": "ACSF network", "ridge": "ridge on ACSF"}
TEACHER = "#52514e"
ANCHOR = "#e34948"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "figure.dpi": 150,
    }
)


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text())


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig_data_efficiency() -> None:
    """Hero: test error versus training-set size, three models, both targets."""
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

    data = _load("data_efficiency")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    offsets = {"mpnn": (6, -3), "acsf_net": (6, 5), "ridge": (6, -1)}
    for ax, key, unit, title, direct in (
        (axes[0], "e_mae_mev_per_atom", "meV/atom", "Energy MAE vs training size", False),
        (axes[1], "f_mae_mev_per_a", "meV/A", "Force MAE vs training size", True),
    ):
        for name in ("mpnn", "acsf_net", "ridge"):
            rows = data["results"][name]
            x = [r["n_train"] for r in rows]
            y = [r[key] for r in rows]
            ax.plot(x, y, "-o", color=COLORS[name], lw=2, ms=4.5, label=LABELS[name])
            if direct:
                ax.annotate(
                    LABELS[name].split(" (")[0],
                    (x[-1], y[-1]),
                    textcoords="offset points",
                    xytext=offsets[name],
                    fontsize=8.5,
                    color=COLORS[name],
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator([100, 200, 400, 800]))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("training frames (whole groups)")
        ax.set_ylabel(f"test MAE ({unit})")
        ax.set_title(title, loc="left")
        _style(ax)
    axes[1].set_xlim(right=axes[1].get_xlim()[1] * 1.8)
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "data_efficiency.png", bbox_inches="tight")
    plt.close(fig)


def fig_parity() -> None:
    """Energy and force parity on the test split for all three models."""
    main = _load("main")
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    for col, name in enumerate(("mpnn", "acsf_net", "ridge")):
        p = main[name]["parity_test"]
        e_true = np.array(p["e_true_per_atom"])
        e_pred = np.array(p["e_pred_per_atom"])
        ax = axes[0, col]
        lo, hi = e_true.min(), e_true.max()
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GRID, lw=1)
        ax.plot(e_true, e_pred, ".", color=COLORS[name], ms=3.5, alpha=0.6)
        ax.set_title(
            f"{LABELS[name]}\nenergy MAE {main[name]['test']['e_mae_mev_per_atom']:.1f} meV/atom",
            loc="left",
        )
        ax.set_xlabel("teacher energy (eV/atom)")
        if col == 0:
            ax.set_ylabel("predicted (eV/atom)")
        _style(ax)

        f_true = np.array(p["f_true_sample"])
        f_pred = np.array(p["f_pred_sample"])
        ax = axes[1, col]
        lim = 1.05 * max(np.abs(f_true).max(), np.abs(f_pred).max())
        ax.plot([-lim, lim], [-lim, lim], color=GRID, lw=1)
        ax.plot(f_true, f_pred, ".", color=COLORS[name], ms=3, alpha=0.45)
        ax.set_title(f"force MAE {main[name]['test']['f_mae_mev_per_a']:.0f} meV/A", loc="left")
        ax.set_xlabel("teacher force (eV/A)")
        if col == 0:
            ax.set_ylabel("predicted (eV/A)")
        _style(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "parity.png", bbox_inches="tight")
    plt.close(fig)


def fig_split_gap() -> None:
    """Group vs random-frame split for both networks, three seeds each.

    The descriptor net is the internal control: its random-frame numbers are
    flattered, while the message-passing model barely moves.
    """
    data = _load("split_gap")
    models = list(data["models"])
    tints = {"mpnn": "#9ec5f4", "acsf_net": "#f6c7b0"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for ax, key, unit, title in (
        (axes[0], "e_mae_mev_per_atom", "meV/atom", "Energy MAE by split protocol"),
        (axes[1], "f_mae_mev_per_a", "meV/A", "Force MAE by split protocol"),
    ):
        ticks, tick_labels = [], []
        for m, name in enumerate(models):
            protocols = data["models"][name]["protocols"]
            for p, proto in enumerate(("group", "random_frame")):
                x = 2.6 * m + p
                vals = [r[key] for r in protocols[proto]]
                fill = COLORS[name] if proto == "group" else tints[name]
                ax.bar(x, np.mean(vals), width=0.8, color=fill, edgecolor=SURFACE)
                ax.plot([x] * len(vals), vals, "o", color=INK, ms=3.2, zorder=3)
                ax.annotate(
                    f"{np.mean(vals):.1f}",
                    (x, np.mean(vals)),
                    textcoords="offset points",
                    xytext=(0, 10),
                    fontsize=8.5,
                    color=INK,
                    ha="center",
                )
                ticks.append(x)
                tick_labels.append("group\n(honest)" if proto == "group" else "random\nframe")
            ax.annotate(
                LABELS[name],
                (2.6 * m + 0.5, ax.get_ylim()[0]),
                xytext=(0, -34),
                textcoords="offset points",
                fontsize=8.5,
                color=COLORS[name],
                ha="center",
            )
        ax.set_xticks(ticks, tick_labels, fontsize=7.5)
        ax.set_ylabel(f"test MAE ({unit})")
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.18)
        _style(ax)
    fig.suptitle(
        "Same data and budget; only the split protocol differs (3 seeds, dots)", fontsize=9.5
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "split_gap.png", bbox_inches="tight")
    plt.close(fig)


def fig_eos() -> None:
    """E-V curves of the trained GNN vs the teacher, with DFT anchors."""
    data = _load("eos")
    comps = [c for c in ("Ti", "Zr", "Nb", "TiZrNb") if c in data]
    fig, axes = plt.subplots(1, len(comps), figsize=(2.9 * len(comps), 3.2), sharey=False)
    for ax, comp in zip(np.atleast_1d(axes), comps):
        entry = data[comp]
        for src, color, label in (
            ("model", COLORS["mpnn"], "GNN"),
            ("teacher", TEACHER, "teacher"),
        ):
            v = np.array(entry[src]["volumes"])
            e = np.array(entry[src]["energies_per_atom"])
            e = e - e.min()
            a = (2.0 * v) ** (1.0 / 3.0)
            ls = "-" if src == "model" else "--"
            ax.plot(a, 1000 * e, ls, color=color, lw=2, label=label)
        if "a0_anchor_dft" in entry:
            ax.axvline(entry["a0_anchor_dft"], color=ANCHOR, lw=1.2, ls=":")
            ax.annotate(
                "DFT a0",
                (entry["a0_anchor_dft"], ax.get_ylim()[1] * 0.82),
                fontsize=8,
                color=ANCHOR,
                rotation=90,
                va="top",
                ha="right",
            )
        ax.set_title(
            f"{comp}\na0 {entry['a0_model']:.3f} vs teacher {entry['a0_teacher']:.3f} A", loc="left"
        )
        ax.set_xlabel("lattice constant (A)")
        _style(ax)
    np.atleast_1d(axes)[0].set_ylabel("E - E$_{min}$ (meV/atom)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eos.png", bbox_inches="tight")
    plt.close(fig)


def fig_nve() -> None:
    """Total-energy trace of the NVE stability runs."""
    data = _load("nve")
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    shades = ["#2a78d6", "#eb6834", "#1baf7a"]
    for k, run in enumerate(data["runs"]):
        t = np.array(run["times_ps"])
        e = 1000 * (np.array(run["e_tot_ev_per_atom"]) - run["e_tot_ev_per_atom"][0])
        label = (
            f"{run['composition']} {run['temperature_k']:.0f} K "
            f"(drift {run['drift_mev_per_atom_per_ps']:+.4f} meV/atom/ps)"
        )
        ax.plot(t, e, color=shades[k % 3], lw=1.6, label=label)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("E$_{tot}$ shift (meV/atom)")
    ax.set_title("NVE total energy with the trained GNN (velocity Verlet, 2 fs)", loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGDIR / "nve.png", bbox_inches="tight")
    plt.close(fig)


def fig_continuity() -> None:
    """Dimer energy as the bond crosses the cutoff, both trained models."""
    data = _load("physics")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, name in zip(axes, ("mpnn", "acsf_net")):
        scan = data[name]["dimer"]
        d = np.array(scan["distances"])
        e = 1000 * (np.array(scan["energies_ev"]) - scan["energies_ev"][-1])
        ax.plot(d, e, color=COLORS[name], lw=2)
        ax.axvline(scan["cutoff"], color=MUTED, lw=1, ls=":")
        ax.annotate(
            f"cutoff {scan['cutoff']:.1f} A",
            (scan["cutoff"], e.min() * 0.9),
            fontsize=8,
            color=MUTED,
            rotation=90,
            va="bottom",
            ha="right",
        )
        ax.set_title(
            f"{LABELS[name]}\nmax step change {scan['max_step_change_mev']:.3f} meV "
            f"per {scan['step_angstrom']:.3f} A",
            loc="left",
        )
        ax.set_xlabel("dimer separation (A)")
        _style(ax)
    axes[0].set_ylabel("energy shift (meV)")
    fig.suptitle(
        "Energy continuity as a neighbor crosses the cutoff (trained models)", fontsize=9.5
    )
    fig.tight_layout()
    fig.savefig(FIGDIR / "continuity.png", bbox_inches="tight")
    plt.close(fig)


def fig_training_curves() -> None:
    """Validation error trajectories of the two networks (main benchmark)."""
    main = _load("main")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, key, unit, title in (
        (axes[0], "val_e_mae", "meV/atom", "Validation energy MAE"),
        (axes[1], "val_f_mae", "meV/A", "Validation force MAE"),
    ):
        for name in ("mpnn", "acsf_net"):
            hist = [h for h in main[name]["history"] if key in h]
            ax.plot(
                [h["epoch"] for h in hist],
                [h[key] for h in hist],
                "-o",
                color=COLORS[name],
                lw=1.8,
                ms=3,
                label=LABELS[name],
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel(f"{title.split()[1].lower()} MAE ({unit})")
        ax.set_yscale("log")
        ax.set_title(title, loc="left")
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (
        fig_data_efficiency,
        fig_parity,
        fig_split_gap,
        fig_eos,
        fig_nve,
        fig_continuity,
        fig_training_curves,
    ):
        try:
            fn()
            made.append(fn.__name__)
        except FileNotFoundError as err:
            print(f"skip {fn.__name__}: {err}")
    print(f"made: {', '.join(made)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
