"""Carve the committed sample from the full labeled dataset.

Selects a few hundred frames spanning every composition and batch kind so
the tutorial and the quickstart run with no generation step. The sample is
teacher-labeled (surrogate model labels, not DFT), like everything else.

Usage:
    python scripts/make_sample_data.py [--per-group 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gnff.frames import read_extxyz, write_extxyz

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-group", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    frames = read_extxyz(REPO / "data" / "full" / "dataset.extxyz")
    rng = np.random.default_rng(args.seed)
    by_group: dict[str, list[int]] = {}
    for k, fr in enumerate(frames):
        by_group.setdefault(fr.group, []).append(k)
    picked = []
    for g in sorted(by_group):
        idx = by_group[g]
        take = min(args.per_group, len(idx))
        picked.extend(rng.choice(idx, size=take, replace=False).tolist())
    picked.sort()
    sample = [frames[k] for k in picked]
    out = REPO / "data" / "sample_frames.extxyz"
    write_extxyz(sample, out)
    print(f"wrote {len(sample)} frames from {len(by_group)} groups to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
