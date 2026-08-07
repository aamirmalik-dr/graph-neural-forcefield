"""Split integrity: leakage-safety, stratification, composition holdout."""

from __future__ import annotations

from gnff.lattice import rattle_batch, strain_batch
from gnff.splits import batch_kind, group_split, random_frame_split


def _frames():
    frames = []
    for k, comp in enumerate(("Ti", "Zr", "Nb", "TiZrNb", "Ti2ZrNb")):
        for b, amp in enumerate((0.05, 0.1, 0.18)):
            frames += rattle_batch(comp, 2, amp, 5, seed=10 * k + b, batch=b)
        frames += strain_batch(comp, 2, 6, seed=100 + k, batch=0)
    return frames


def test_batch_kind():
    assert batch_kind("TiZrNb_rattle10_b2", "TiZrNb") == "rattle10"
    assert batch_kind("Ti2ZrNb_strain_vol_b0", "Ti2ZrNb") == "strain_vol"
    assert batch_kind("Nb_md1400_b1", "Nb") == "md1400"


def test_groups_never_straddle_sides():
    frames = _frames()
    split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    sides = {}
    for name, idxs in (("train", split.train), ("val", split.val), ("test", split.test)):
        for k in idxs:
            g = frames[k].group
            assert sides.setdefault(g, name) == name


def test_holdout_compositions_only_in_transfer():
    frames = _frames()
    split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    for k in split.train + split.val + split.test:
        assert frames[k].composition != "Ti2ZrNb"
    assert all(frames[k].composition == "Ti2ZrNb" for k in split.transfer)
    assert len(split.transfer) > 0


def test_test_set_is_stratified_by_kind():
    """Every batch kind must appear in the test set (the repo-1 lesson)."""
    frames = _frames()
    for seed in range(3):
        split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=seed)
        test_kinds = {batch_kind(frames[k].group, frames[k].composition) for k in split.test}
        all_kinds = {
            batch_kind(fr.group, fr.composition) for fr in frames if fr.composition != "Ti2ZrNb"
        }
        assert test_kinds == all_kinds


def test_splits_are_disjoint_and_complete():
    frames = _frames()
    split = group_split(frames, holdout_compositions=("Ti2ZrNb",), seed=1)
    all_idx = sorted(split.train + split.val + split.test + split.transfer)
    assert all_idx == list(range(len(frames)))


def test_random_split_mixes_groups():
    """The optimistic protocol splits siblings; that is exactly its flaw."""
    frames = _frames()
    split = random_frame_split(frames, holdout_compositions=("Ti2ZrNb",), seed=0)
    train_groups = {frames[k].group for k in split.train}
    test_groups = {frames[k].group for k in split.test}
    assert train_groups & test_groups
