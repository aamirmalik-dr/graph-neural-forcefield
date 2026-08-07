# Benchmark results

Every number below was measured in this repository's environment (fresh
Python 3.11 venv, torch 2.13.0+cpu, chgnet 0.4.2), on the dataset generated
by `gnff generate` with the committed seeds. Raw outputs live in
`results/*.json`; `results/metrics.json` is the compiled summary; every
figure regenerates from the JSONs via `scripts/make_figures.py`.

Labels are surrogate model labels from the CHGNet teacher, never DFT. All
"error" below means error against the teacher. Real DFT enters only as the
Materials Project EOS anchors.

## Main comparison: matched data, matched budget

Group split (whole generation batches, stratified by batch kind), seed 0:
882 train / 144 val / 192 test frames, 348 transfer frames from two
compositions never seen in training. Both networks got the same
learning-rate screen ({1e-3, 3e-3}, 15 epochs on validation; both chose
3e-3) and the same 80-epoch, batch-24, force-weight-0.1 Adam budget with
best-validation selection. Ridge alpha was tuned on validation over
{1e-10 ... 1e-2} (chose 1e-10; the grid edge was probed and the val error is
monotone in alpha, so smaller values change nothing visible).

| Model | Params | Test E MAE (meV/atom) | Test E RMSE | Test F MAE (meV/A) | Test F RMSE | Transfer E MAE | Transfer F MAE | Train (s) | Inference (ms, 54 atoms) |
|---|---|---|---|---|---|---|---|---|---|
| Message passing (GNN) | 40,082 | **7.00** | 13.41 | **30.9** | 43.3 | 6.38 | **40.5** | 665 | 11.8 |
| ACSF network | 7,971 | 8.20 | 12.31 | 60.0 | 114.7 | 9.67 | 70.5 | 196 | 13.3 |
| Ridge on ACSF | 147 | 9.29 | 18.38 | 86.5 | 147.9 | 6.77 | 91.1 | 61 | 9.7 |

Reading this honestly:

- The GNN's decisive win is forces: 1.9x lower MAE than the ACSF network
  and 2.8x lower than ridge, with the RMSE gap even larger (43 vs 115 and
  148), meaning the descriptor models have heavy error tails the GNN does
  not.
- The energy win is modest (7.0 vs 8.2 vs 9.3), and single-run energy
  numbers carry 1-3 meV/atom of seed-to-seed spread (see the split-protocol
  section), so treat the energy ranking as weak evidence and the force
  ranking as strong evidence.
- On transfer compositions the 147-parameter ridge nearly matches the GNN
  on energies (6.77 vs 6.38 meV/atom). Linear models extrapolate
  composition trends well on this narrow chemistry; message passing is not
  needed to get alloy mixing energetics roughly right. Forces are a
  different story: the GNN keeps a 2.2x transfer force advantage.
- Fairness correction made during the build, stated plainly: the first
  ridge implementation accumulated raw (sum-normalized) normal equations,
  which effectively weighted forces about 16x more than the network loss
  does, and its energy MAE came out 25 meV/atom. Mean-normalizing the
  objective to match the network loss exactly (and re-tuning alpha)
  improved the baseline to the 9.29 reported here. All committed ridge
  numbers use the corrected, better baseline.

## Data efficiency (the hero result)

Same test set at every size; training subsets are nested unions of whole
generation groups (never partial batches), identical for all three models;
same per-size budget.

| n train | GNN E | ACSF E | Ridge E | GNN F | ACSF F | Ridge F |
|---|---|---|---|---|---|---|
| 102 | **35.2** | 67.0 | 44.6 | **88** | 182 | 189 |
| 219 | **15.5** | 84.7 | 76.9 | **100** | 188 | 181 |
| 420 | **12.1** | 27.0 | 20.2 | **80** | 129 | 160 |
| 819 | **9.5** | 10.5 | 10.3 | **37** | 62 | 92 |
| 882 | 10.5 | **9.1** | 9.3 | **33** | 63 | 86 |

(E in meV/atom, F in meV/A.)

- With 102 training frames the GNN already matches the force accuracy the
  ridge baseline reaches with all 882 (88 vs 86 meV/A): roughly 8x data
  efficiency on forces against the linear model, and the ACSF network
  needs 800+ frames to beat it.
- Below about 400 frames the GNN leads on everything, often by 2x or more.
- At full data the energy numbers of all three models converge to within
  about 1.5 meV/atom and the ordering flips between runs (the 10.5 here vs
  7.0 in the main run is the same model and data, differing only in group
  ordering during SGD). The honest conclusion: message passing buys forces
  and small-data accuracy on this chemistry, not full-data energies.
- The non-monotone bump at 219 frames for the descriptor models is real
  (both fits get worse when the second tranche of groups adds harder batch
  kinds before enough data exists to fit them) and reproduces with the
  fixed seed.

## Split-protocol control: what a random-frame split would have claimed

Three split seeds per protocol per model, identical budget everywhere.
Sibling frames from one rattle, strain, or MD batch are near-duplicates, so
a random-frame split leaks siblings into the test set. Measured on the
message-passing model and, as internal control, the descriptor network:

| Model | Protocol | E MAE (meV/atom) | F MAE (meV/A) |
|---|---|---|---|
| GNN | group (honest) | 5.69 +/- 0.94 | 34.6 +/- 3.0 |
| GNN | random frame | 6.51 +/- 1.21 | 35.9 +/- 4.8 |
| ACSF network | group (honest) | 9.65 +/- 2.69 | 66.8 +/- 11.2 |
| ACSF network | random frame | 8.23 +/- 0.56 | 55.5 +/- 4.2 |

- The descriptor network is flattered by the leaky protocol: 15 percent on
  energies, 17 percent on forces, and, more insidiously, the seed spread
  collapses from 2.7 to 0.6 meV/atom, so the leaky protocol also
  understates uncertainty by about 5x.
- The GNN shows no flattery at all on this dataset (the random-frame means
  are, if anything, slightly worse; the differences are within seed
  spread). At this capacity and budget the GNN is not memorizing
  individual frames, so sibling leakage has nothing to exploit.
- The descriptor repo of this cluster measured a much larger 2.8x energy
  gap for the same ACSF architecture using its own training loop (which
  adds learning-rate decay, allowing deeper late-stage memorization). The
  direction is consistent everywhere; the magnitude depends on model AND
  training loop. Either way, only group-split numbers are quoted as model
  quality anywhere in this repository.

## Equation of state vs teacher and real DFT anchors

Birch-Murnaghan fits to 13-point E-V scans of 16-atom cells. Anchors are
Materials Project DFT values fetched via the API (CC BY 4.0); the committed
literature fallback carries the same values with citations.

| Composition | a0 GNN (A) | a0 teacher | a0 DFT anchor | B0 GNN (GPa) | B0 teacher | B0 DFT anchor |
|---|---|---|---|---|---|---|
| Ti | 3.261 | 3.268 | 3.2516 (mp-73) | 68.6 | 74.3 | 111.0 |
| Zr | 3.660 | 3.595 | 3.5817 (mp-41) | 48.5 | 88.6 | 88.4 |
| Nb | 3.336 | 3.327 | 3.3176 (mp-75) | 174.2 | 166.6 | 172.1 |
| TiZrNb | 3.387 | 3.385 | none | 101.3 | 104.9 | none |

- Nb and the equiatomic alloy are excellent (a0 within 0.01 A of the
  teacher, B0 within 8 GPa). Ti is good on geometry; note the teacher
  itself misses the DFT Ti bulk modulus by 37 GPa, and the student
  inherits that (the student can never beat its teacher's physics).
- Zr is the honest weak spot: a0 is 0.065 A too large and B0 collapses to
  48.5 vs the teacher's 88.6 GPa. The same composition was the weak spot
  of the descriptor repo (+0.047 A there), so this is a data-recipe
  limitation (too few strained Zr-rich environments at large volumes), not
  an architecture quirk. Do not use these checkpoints for Zr-rich elastic
  properties.

## NVE stability

Velocity-Verlet, 2 fs timestep, 1500 steps (3 ps), 54-atom cells, forces
from autograd of the trained GNN:

| System | T init | Drift (meV/atom/ps) | E_tot std (meV/atom) |
|---|---|---|---|
| TiZrNb | 300 K | +0.0001 | 0.0014 |
| TiZrNb | 900 K | +0.0003 | 0.0034 |
| Nb | 600 K | +0.0003 | 0.0035 |

Drift is indistinguishable from zero at this timestep, as it must be when
forces are the exact gradient of a smooth energy.

## Physics checks on the trained models

| Check | GNN | ACSF network |
|---|---|---|
| Dimer scan, max energy change per 0.005 A step | 0.107 meV | 0.592 meV |
| Dimer energy beyond cutoff (tail flatness) | 0.0 meV | 0.0 meV |
| In-crystal drag scan, max step change | 5.4 meV | 6.6 meV |
| Rotation / translation / permutation deltas | all 0.0 meV | all 0.0 meV |
| Autograd force vs finite differences | 5.2e-9 eV/A | 5.6e-9 eV/A |

The step-change numbers are slope-dominated (the scans cross steeply
repulsive regions); the discontinuity claim rests on the exactly flat tail
plus the smooth cutoff envelope, and the same properties are enforced as
unit tests. Invariances hold to machine precision because both models are
functions of interatomic distances and angles only.

## Cost accounting

- Dataset: 1,566 frames labeled at 0.095 s/frame; teacher-driven MD
  generation 325 s; full generation about 8 minutes.
- Teacher choice by measured CPU speed (54-atom energy+forces call):
  CHGNet 0.182 s vs MACE-MP-0 small 0.497 s (`data/teacher_choice.json`).
- Benchmark wall times: main 1293 s, data efficiency 2199 s, split-gap
  control 3810 s, EOS 5 s, NVE 57 s, physics 4 s. About two hours end to
  end on CPU, plus generation.
- The executed tutorial notebook runs its own miniature pipeline (144
  frames, no MD batches) in a few minutes; in that gentle, near-harmonic
  regime the ridge baseline is genuinely competitive on forces, which the
  notebook reports as the teaching point it is.
