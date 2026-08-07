"""Labeled periodic structures and extxyz persistence.

A Frame is one periodic structure with optional energy and force labels. The
labels in this project are surrogate model labels produced by a pretrained
universal potential (the teacher), never DFT; real DFT enters only through the
Materials Project validation anchors. Every frame carries a generation-group
id and a composition tag so splits can respect data provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ELEMENTS = ("Ti", "Zr", "Nb")
SYMBOL_TO_INDEX = {s: k for k, s in enumerate(ELEMENTS)}


@dataclass
class Frame:
    """One periodic structure with optional labels.

    Attributes:
        species: (N,) int array; 0=Ti, 1=Zr, 2=Nb.
        positions: (N, 3) Cartesian coordinates in Angstrom.
        cell: (3, 3) lattice matrix, rows are lattice vectors.
        energy: Total energy in eV (surrogate teacher label) or None.
        forces: (N, 3) forces in eV/Angstrom (surrogate teacher label) or None.
        group: Generation-group id, e.g. "TiZrNb_md600_b0".
        composition: Composition tag, e.g. "TiZrNb".
    """

    species: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    energy: float | None = None
    forces: np.ndarray | None = None
    group: str = ""
    composition: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    def element_counts(self) -> np.ndarray:
        """Per-element atom counts, shape (3,)."""
        return np.bincount(self.species, minlength=len(ELEMENTS))


def write_extxyz(frames: list[Frame], path: str | Path) -> None:
    """Write frames to an extxyz file (ASE is used for file handling only)."""
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    images = []
    for fr in frames:
        atoms = Atoms(
            symbols=[ELEMENTS[int(s)] for s in fr.species],
            positions=fr.positions,
            cell=fr.cell,
            pbc=True,
        )
        atoms.info["group"] = fr.group
        atoms.info["composition"] = fr.composition
        if fr.energy is not None:
            atoms.calc = SinglePointCalculator(atoms, energy=fr.energy, forces=fr.forces)
        images.append(atoms)
    write(str(path), images, format="extxyz")


def read_extxyz(path: str | Path) -> list[Frame]:
    """Read frames from an extxyz file written by write_extxyz."""
    from ase.io import read

    frames = []
    for atoms in read(str(path), index=":", format="extxyz"):
        energy, forces = None, None
        if atoms.calc is not None:
            try:
                energy = float(atoms.get_potential_energy())
                forces = np.asarray(atoms.get_forces())
            except Exception:
                energy, forces = None, None
        frames.append(
            Frame(
                species=np.array([SYMBOL_TO_INDEX[s] for s in atoms.get_chemical_symbols()]),
                positions=np.asarray(atoms.get_positions()),
                cell=np.asarray(atoms.get_cell()),
                energy=energy,
                forces=forces,
                group=str(atoms.info.get("group", "")),
                composition=str(atoms.info.get("composition", "")),
            )
        )
    return frames
