# Model card: committed TiZrNb potentials

Three trained checkpoints ship with this repository, all trained this
session on the same data and evaluated on the same leakage-safe splits.
They exist to make the demos instant and the benchmark reproducible, not to
be production potentials.

## Checkpoints

| File | Model | Parameters | Test E MAE (meV/atom) | Test F MAE (meV/A) |
|---|---|---|---|---|
| `models/mpnn.pt` | SchNet-style message-passing network | 40,082 | 7.00 | 30.9 |
| `models/acsf_net.pt` | per-element ACSF network (32, 32) | 7,971 | 8.20 | 60.0 |
| `models/ridge.pt` | per-element ridge on ACSF descriptors | 147 | 9.29 | 86.5 |

Test = held-out generation groups, stratified by batch kind, compositions
seen in training. All three models were trained on identical frames with a
matched budget (details below). CPU inference on a 54-atom cell: 11.8 ms
(mpnn), 13.3 ms (acsf_net), 9.7 ms (ridge), including neighbor-list
construction, measured on the build machine.

## Architecture (mpnn.pt)

- Learned element embeddings (width 48), 3 continuous-filter interaction
  blocks, 24 Gaussian radial basis functions under a cosine cutoff envelope
  at 5.0 A, shifted-softplus activations, per-atom energy readout on top of
  fixed per-element reference energies (least squares on train).
- Forces are exact autograd derivatives, F = -dE/dr; float64 throughout.
- After 3 interaction blocks an atom's energy sees structure out to
  approximately 15 A (three hops), even though each hop is local.

## Training data. THE LABELS ARE NOT DFT

Every training label is a surrogate model label produced by CHGNet (pip
package `chgnet` 0.4.2, default pretrained checkpoint CHGNet v0.3.0 trained
on the Materials Project MPtrj dataset, BSD-3-Clause), selected over
MACE-MP-0 small by measured CPU speed (0.18 vs 0.50 s per 54-atom
energy+force call, `data/teacher_choice.json`). The models distill the
teacher; they inherit its biases and add their own fitting error. Nothing
in the training set is DFT. Real DFT appears only as Materials Project
equation-of-state anchors in the EOS benchmark (CC BY 4.0).

Dataset: 1,566 frames in 72 generation groups; BCC Ti/Zr/Nb, three
binaries, equiatomic TiZrNb, plus two off-equiatomic compositions held out
entirely for transfer testing; 16-54 atom supercells; rattle amplitudes
0.05-0.18 A, volumetric strains to +/-6%, shears to 5%, teacher-driven MD
at 600 K and 1400 K. Split: 882 train / 144 val / 192 test frames by whole
generation groups, 348 transfer frames. Recipe and seeds are identical to
the descriptor repo of this project cluster.

## Budget

mpnn: 80 epochs, batch 24, Adam, learning rate 3e-3 chosen by a 15-epoch
validation screen over {1e-3, 3e-3} (both networks got the same grid),
force weight 0.1, best-validation state selected; 663 s train time on CPU.
acsf_net: same budget and grid, 217 s. ridge: closed-form weighted normal
equations with the mean-normalized objective matching the network loss,
alpha tuned on validation over {1e-10 ... 1e-2} (chose 1e-10), 61 s.

## Validation beyond test error

- EOS vs teacher and real-DFT anchors: Nb and TiZrNb near-exact vs teacher
  (a0 within 0.01 A); Ti good; Zr is the honest weak spot (a0 +0.065 A,
  B0 48.5 vs teacher 88.6 GPa). See `results/eos.json`.
- NVE with the mpnn checkpoint: |drift| <= 0.0003 meV/atom/ps over 3 ps at
  300-900 K, 2 fs timestep.
- Physics: energy exactly flat beyond the cutoff (tail flatness < 1e-9 meV),
  smooth across it (max 0.107 meV change per 0.005 A step on a dimer scan);
  rotation, translation, and permutation invariance at machine precision;
  autograd forces match finite differences to < 1e-6 eV/A.

## Intended use and limits

- Scope: bulk BCC TiZrNb solid solutions near equilibrium and up to
  moderate rattles, strains, and temperatures covered by the training
  distribution. Nothing else has been tested: no surfaces, defects, liquid,
  other phases, or other elements.
- The checkpoints reproduce the TEACHER, not experiment. Errors against
  real DFT or experiment are strictly larger than the numbers above.
- The Zr EOS bias means elastic properties of Zr-rich compositions should
  not be trusted without refitting.
- Off-equiatomic transfer (Ti2ZrNb, TiZrNb2) is measured and reported in
  RESULTS.md; extrapolation further off-composition is untested.

## Reproducing

`python scripts/run_all.py` regenerates the dataset (fixed seeds), retrains
all three checkpoints, and rewrites every results JSON and figure.
