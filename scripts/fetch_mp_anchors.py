"""Fetch real-DFT validation anchors from the Materials Project API.

These anchors (BCC lattice constants and bulk moduli of elemental Ti, Zr,
Nb) are the ONLY real DFT numbers in this project; every training label is a
surrogate model label from the teacher. Materials Project data is CC BY 4.0
and requires a free API key in the MP_API_KEY environment variable.

The API returns primitive cells, so the conventional BCC lattice constant is
recovered from the volume per atom (a0 = (2 V_atom)^(1/3)); the bulk modulus
is the Voigt-Reuss-Hill value. If no key is configured, the committed
literature fallback data/anchors_literature.json is used by the benchmarks
instead.

Usage:
    python scripts/fetch_mp_anchors.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

BCC_IDS = {"Ti": "mp-73", "Zr": "mp-41", "Nb": "mp-75"}


def main() -> int:
    key = os.environ.get("MP_API_KEY", "")
    if not key:
        print("MP_API_KEY is not set; benchmarks will fall back to")
        print("data/anchors_literature.json (committed, with citations).")
        return 1
    from mp_api.client import MPRester

    anchors = {}
    with MPRester(key) as mpr:
        for element, mp_id in BCC_IDS.items():
            docs = mpr.materials.summary.search(
                material_ids=[mp_id],
                fields=["material_id", "volume", "nsites", "bulk_modulus", "symmetry"],
            )
            doc = docs[0]
            v_atom = float(doc.volume) / int(doc.nsites)
            a0 = (2.0 * v_atom) ** (1.0 / 3.0)
            bm = doc.bulk_modulus  # dict or object depending on mp-api version
            if bm is None:
                b0 = None
            elif isinstance(bm, dict):
                b0 = float(bm["vrh"])
            else:
                b0 = float(bm.vrh)
            anchors[element] = {
                "material_id": str(doc.material_id),
                "a0_angstrom": round(a0, 4),
                "b0_gpa": round(b0, 3) if b0 is not None else None,
            }
            print(f"{element} ({mp_id}): a0 {a0:.4f} A, B0 {b0} GPa", flush=True)
    payload = {
        "source": "Materials Project API (DFT), CC BY 4.0",
        "note": (
            "Real DFT validation anchors. Conventional BCC a0 recovered from "
            "volume per atom of the primitive cell; B0 is bulk_modulus.vrh."
        ),
        "citation": (
            "A. Jain et al., APL Materials 1, 011002 (2013). " "https://materialsproject.org"
        ),
        "anchors": anchors,
    }
    out = REPO / "data" / "anchors_mp.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
