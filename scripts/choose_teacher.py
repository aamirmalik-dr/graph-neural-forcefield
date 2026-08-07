"""Measure CPU speed of the candidate teachers and record the choice.

Times one energy+forces call on a rattled 54-atom BCC TiZrNb cell for
CHGNet and MACE-MP-0 small (both pip installed), and writes
data/teacher_choice.json with the timings, versions, and licenses. The
faster teacher labels the dataset; the identity is fixed in
configs/dataset.yaml and recorded again in data/provenance.json.

Usage:
    python scripts/choose_teacher.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gnff.labeling import get_teacher, time_teacher
from gnff.lattice import bcc_supercell

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    frame = bcc_supercell("TiZrNb", 3, seed=0)
    rng = np.random.default_rng(0)
    frame.positions = frame.positions + rng.normal(0.0, 0.08, frame.positions.shape)
    report = {"structure": "54-atom rattled BCC TiZrNb", "candidates": {}}
    for name in ("chgnet", "mace-mp0-small"):
        try:
            calc, info = get_teacher(name)
        except Exception as err:  # missing optional dependency
            report["candidates"][name] = {"error": str(err)}
            continue
        seconds = time_teacher(calc, frame, repeats=5)
        info["seconds_per_call"] = round(seconds, 3)
        report["candidates"][name] = info
        print(f"{name}: {seconds:.3f} s per energy+forces call", flush=True)
    timed = {k: v["seconds_per_call"] for k, v in report["candidates"].items() if "error" not in v}
    report["chosen"] = min(timed, key=timed.get) if timed else None
    out = REPO / "data" / "teacher_choice.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"chosen teacher: {report['chosen']}; wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
