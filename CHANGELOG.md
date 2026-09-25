# Changelog

## Unreleased

- Package metadata completed: keywords, classifiers, and project URLs in `pyproject.toml`; citation file added.

## 0.1.0 (2026-08-08)

- First public release: from-scratch SchNet-style message-passing neural network potential for BCC TiZrNb in PyTorch, with periodic graph construction, batched graphs, continuous-filter convolutions, autograd forces, and a training loop.
- Descriptor baselines reimplemented in the same code base (compact Behler-Parrinello ACSF network and ridge regression on the same descriptors) and benchmarked head to head at matched data and budget, including the data-efficiency curve.
- Fixed-seed benchmark configs, data pipeline scripts, Materials Project EOS anchors, committed sample frames, trained checkpoints, results, and figures.
- README, RESULTS, model card, API guide, and an executed tutorial notebook.
- Fix: periodic image ranges corrected for strongly sheared cells (row versus column norms of the inverse cell), inert at the shears present in the committed dataset.
- Fix: the tutorial notebook's split-gap statement corrected and the notebook re-executed.
- Docs: training times aligned with `results/metrics.json`; claims tightened.
- Hero figure annotated with the ridge full-data force level.
