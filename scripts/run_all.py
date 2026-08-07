"""Reproduce every committed benchmark result in order.

Assumes the dataset exists (gnff generate) or generates it if missing, then
runs main, data_efficiency, split_gap, eos, nve, and physics, then rebuilds
metrics.json and the figures.

Usage:
    python scripts/run_all.py [--skip-generate]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=REPO)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    dataset = REPO / "data" / "full" / "dataset.extxyz"
    if not dataset.exists() and not args.skip_generate:
        run([sys.executable, "-m", "gnff.cli", "generate"])
    for mode in ("main", "data_efficiency", "split_gap", "eos", "nve", "physics"):
        run([sys.executable, "scripts/run_benchmarks.py", "--mode", mode])
    run([sys.executable, "scripts/make_metrics.py"])
    run([sys.executable, "scripts/make_figures.py"])
    print(f"\ntotal {time.perf_counter() - start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
