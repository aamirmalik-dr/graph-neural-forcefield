# gnff API guide

Everything below is importable from `gnff`. All tensors are float64, all
models run on CPU, and every model exposes the same two-method interface:
`structure_energies(batch, positions)` (differentiable) and
`energy_and_forces(batch)` (forces from autograd, exactly `-dE/dr`).

## Structures and datasets

```python
from gnff import Frame, read_extxyz, write_extxyz

frames = read_extxyz("data/sample_frames.extxyz")
fr = frames[0]
fr.species        # (N,) 0=Ti 1=Zr 2=Nb
fr.positions      # (N, 3) Angstrom
fr.cell           # (3, 3) row lattice vectors
fr.energy         # eV, surrogate teacher label (never DFT)
fr.forces         # (N, 3) eV/A
fr.group          # generation group id, the unit of the leakage-safe split
```

Generate the full dataset (teacher labeling on CPU, roughly 10 minutes):

```python
import yaml
from gnff.dataset import build_dataset
from gnff import write_extxyz

cfg = yaml.safe_load(open("configs/dataset.yaml"))
frames, provenance = build_dataset(cfg)
write_extxyz(frames, "data/full/dataset.extxyz")
```

## Leakage-safe splits

```python
from gnff import group_split, random_frame_split

split = group_split(frames, holdout_compositions=("Ti2ZrNb", "TiZrNb2"), seed=0)
train = [frames[k] for k in split.train]     # whole generation groups only
transfer = [frames[k] for k in split.transfer]  # held-out compositions
```

`random_frame_split` exists only to measure the optimism of frame-level
splitting; never use it to report model quality.

## Graphs

```python
from gnff import build_batch

batch = build_batch(train[:8], cutoff=5.0)              # pair graph (MPNN)
batch = build_batch(train[:8], 5.5, angular_cutoff=4.5)  # + triplets (ACSF)
```

`BatchCache` precomputes per-frame neighbor lists once and assembles
minibatches cheaply during training.

## Models

```python
from gnff import AcsfNet, MpnnConfig, MpnnPotential, RidgePotential

gnn = MpnnPotential(MpnnConfig(hidden=48, n_blocks=3, n_rbf=24, cutoff=5.0))
descriptor_net = AcsfNet(hidden=(32, 32))
ridge = RidgePotential()

energies, forces = gnn.energy_and_forces(batch)  # (S,) eV and (N, 3) eV/A
```

## Training and evaluation

```python
from gnff import TrainSettings, train_potential, evaluate_potential, error_summary

report = train_potential(
    gnn, train, val,
    TrainSettings(epochs=80, batch_size=24, lr=1e-3, force_weight=0.1, seed=0),
)
e_mae, f_mae = evaluate_potential(gnn, test)   # meV/atom, meV/A
summary = error_summary(gnn, test)             # + RMSEs

ridge.fit(train, val, force_weight=0.1)  # alpha tuned on val over a fixed grid
```

## Checkpoints

```python
from gnff import save_checkpoint, load_checkpoint

save_checkpoint(gnn, "models/mpnn.pt")
model = load_checkpoint("models/mpnn.pt")   # dispatches on checkpoint kind
```

## Physics validation

```python
from gnff.eos import model_eos
from gnff.dynamics import run_nve
from gnff.physics import dimer_scan, invariance_report, force_consistency
from gnff.lattice import bcc_supercell

model_eos(model, "Nb")["a0_angstrom"]
run_nve(model, bcc_supercell("TiZrNb", 3, seed=0), 300.0, n_steps=1500)
dimer_scan(model)["max_step_change_mev"]     # continuity across the cutoff
invariance_report(model)                      # translation/rotation/permutation
force_consistency(model)                      # autograd vs finite differences
```

## CLI

```
gnff generate  --config configs/dataset.yaml
gnff train     --data data/full/dataset.extxyz --model mpnn --out models/mpnn.pt
gnff eval      --checkpoint models/mpnn.pt --data data/sample_frames.extxyz
gnff predict   --checkpoint models/mpnn.pt --structure my_structures.extxyz
gnff eos       --checkpoint models/mpnn.pt --composition Nb
gnff md        --checkpoint models/mpnn.pt --composition TiZrNb --temperature 300
```

`gnff train` also accepts `--model acsf_net` and `--model ridge`, and an
external checkpoint produced elsewhere can be evaluated as long as it was
saved by `save_checkpoint` (the file records its own kind and
hyperparameters).
