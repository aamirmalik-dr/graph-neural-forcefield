# Data provenance

## What the labels are, and are not

Every energy and force label in this repository is a **surrogate model
label**: it was produced by a pretrained universal potential (the teacher)
that was itself trained on DFT data. **No training label here is DFT.** The
teacher identity, version, checkpoint, and license are recorded in
`provenance.json` after generation; for the committed dataset they are also
stated in the README and the model card.

Real DFT enters this project in exactly one place: the equation-of-state
validation anchors for elemental BCC Ti, Zr, and Nb, taken from the
Materials Project (DFT, CC BY 4.0). Run `python scripts/fetch_mp_anchors.py`
with a free `MP_API_KEY` to fetch `anchors_mp.json`; without a key the
benchmarks fall back to the committed `anchors_literature.json`, which lists
the same quantities with citations.

## Files

- `sample_frames.extxyz`: committed sample of the teacher-labeled dataset,
  a few frames from every generation group (all compositions and batch
  kinds). This is a carved subset, committed so the demos run immediately;
  it is NOT the full training set.
- `provenance.json`: teacher identity, frame counts, timings, and the full
  generation config, written by `gnff generate`.
- `anchors_mp.json` / `anchors_literature.json`: real-DFT validation
  anchors, fetched and fallback respectively (both CC BY 4.0, attributed).
- `full/` (gitignored): the full generated dataset lives here locally.

## Regenerating the full dataset

```
gnff generate --config configs/dataset.yaml
```

This rebuilds the exact dataset used by every committed benchmark (fixed
seeds; the recipe is identical to the descriptor repo of this project
cluster so cross-repo comparisons are internally consistent) and labels it
with the teacher on CPU. Expect roughly 10 minutes.

## Attribution

- Teacher labels: CHGNet (pip package `chgnet`, BSD-3-Clause, pretrained on
  the Materials Project MPtrj dataset). Labels are the teacher's
  predictions, not DFT.
- Validation anchors: Materials Project (CC BY 4.0), A. Jain et al., APL
  Materials 1, 011002 (2013).
- Structure and file handling: ASE (LGPL, pip dependency only).
