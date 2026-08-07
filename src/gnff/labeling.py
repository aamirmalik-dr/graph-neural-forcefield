"""Teacher labeling with a pretrained universal potential (surrogate labels).

Every energy and force label in this project is a SURROGATE MODEL LABEL from
a pip-installed universal potential (CHGNet by default, MACE-MP-0 small as
the alternative), which was itself trained on DFT data. The labels are never
DFT and are never described as DFT. Real DFT enters the project only through
the Materials Project validation anchors (scripts/fetch_mp_anchors.py).
"""

from __future__ import annotations

import time

import numpy as np

from gnff.frames import ELEMENTS, Frame


def frame_to_atoms(frame: Frame):
    """Convert a Frame to an ASE Atoms object (structure handling only)."""
    from ase import Atoms

    return Atoms(
        symbols=[ELEMENTS[int(s)] for s in frame.species],
        positions=frame.positions,
        cell=frame.cell,
        pbc=True,
    )


def get_teacher(name: str):
    """Construct the teacher ASE calculator.

    Args:
        name: "chgnet" or "mace-mp0-small".

    Returns:
        (calculator, info) where info records package, version, checkpoint,
        and license for provenance reporting.
    """
    if name == "chgnet":
        import chgnet
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator

        model = CHGNet.load()
        calc = CHGNetCalculator(model=model, use_device="cpu")
        version = getattr(model, "version", "0.3.0")
        info = {
            "package": "chgnet",
            "version": chgnet.__version__,
            "checkpoint": f"CHGNet v{version} (default pretrained weights, MPtrj-trained)",
            "license": "BSD-3-Clause (code and pretrained weights)",
        }
        return calc, info
    if name == "mace-mp0-small":
        import mace
        from mace.calculators import mace_mp

        calc = mace_mp(model="small", device="cpu", default_dtype="float64")
        info = {
            "package": "mace-torch",
            "version": mace.__version__,
            "checkpoint": "MACE-MP-0 small",
            "license": "MIT (code and MACE-MP-0 weights)",
        }
        return calc, info
    raise ValueError(f"unknown teacher {name}")


def label_frames(frames: list[Frame], calc) -> float:
    """Label frames in place with teacher energies and forces.

    Returns:
        Mean wall-clock seconds per frame.
    """
    start = time.perf_counter()
    for fr in frames:
        atoms = frame_to_atoms(fr)
        atoms.calc = calc
        fr.energy = float(atoms.get_potential_energy())
        fr.forces = np.asarray(atoms.get_forces())
    return (time.perf_counter() - start) / max(1, len(frames))


def time_teacher(calc, frame: Frame, repeats: int = 5) -> float:
    """Mean wall seconds for one energy+forces teacher call on frame.

    Every repeat perturbs the positions slightly; ASE calculators cache
    results for an unchanged structure, which would otherwise time as zero.
    """
    rng = np.random.default_rng(0)

    def one_call() -> None:
        atoms = frame_to_atoms(frame)
        atoms.positions = atoms.positions + rng.normal(0.0, 1e-3, atoms.positions.shape)
        atoms.calc = calc
        atoms.get_potential_energy()
        atoms.get_forces()

    for _ in range(2):
        one_call()
    start = time.perf_counter()
    for _ in range(repeats):
        one_call()
    return (time.perf_counter() - start) / repeats


def teacher_md_batch(
    composition: str,
    reps: int,
    calc,
    temperature_k: float,
    n_steps: int,
    sample_every: int,
    seed: int,
    batch: int,
    timestep_fs: float = 3.0,
) -> list[Frame]:
    """Short teacher-driven Langevin MD; snapshots form one generation group.

    Args:
        composition: Composition tag.
        reps: Supercell repetitions.
        calc: Teacher ASE calculator (drives the dynamics).
        temperature_k: Thermostat temperature.
        n_steps: Total MD steps.
        sample_every: Snapshot stride.
        seed: RNG seed (initial occupancy and velocities).
        batch: Batch index for the group id.
        timestep_fs: Integration timestep in femtoseconds.

    Returns:
        Unlabeled snapshot frames (label separately with label_frames).
    """
    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from gnff.lattice import bcc_supercell

    fr0 = bcc_supercell(composition, reps, seed=seed)
    atoms = frame_to_atoms(fr0)
    atoms.calc = calc
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_k, rng=np.random.default_rng(seed)
    )
    dyn = Langevin(
        atoms,
        timestep=timestep_fs * units.fs,
        temperature_K=temperature_k,
        friction=0.02,
        rng=np.random.default_rng(seed + 1),
    )
    frames: list[Frame] = []

    def snapshot() -> None:
        frames.append(
            Frame(
                species=fr0.species.copy(),
                positions=np.asarray(atoms.get_positions()),
                cell=np.asarray(atoms.get_cell()),
                group=f"{composition}_md{int(temperature_k)}_b{batch}",
                composition=composition,
            )
        )

    dyn.attach(snapshot, interval=sample_every)
    dyn.run(n_steps)
    return frames
