"""Ridge regression on summed ACSF descriptors, with exact force fitting.

The linear baseline: structure energy is a per-element linear function of the
same 48 ACSF descriptors the compact network uses, so the comparison isolates
what nonlinearity and message passing buy. Both energies (per atom) and
forces (analytic descriptor gradients, computed by autograd) enter the
closed-form normal equations, and the regularization strength is tuned on
the validation split. Prediction forces come from autograd of the linear
energy, so F = -dE/dr holds exactly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gnff.frames import Frame
from gnff.graphs import Batch, build_batch
from gnff.models.acsf import AcsfParams, compute_acsf
from gnff.models.mpnn import N_ELEMENTS


class RidgePotential(nn.Module):
    """Per-element linear model on ACSF descriptors.

    E(structure) = counts . elem_ref + sum_e [sum_{atoms of e} D] . w_e
                   + counts . bias

    Attributes:
        params: ACSF hyperparameters (shared with AcsfNet).
        weight: (3, n_features) per-element weights.
        bias: (3,) per-element energy offsets.
    """

    def __init__(self, params: AcsfParams | None = None, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.params = params or AcsfParams()
        n_feat = self.params.n_features
        self.register_buffer("weight", torch.zeros(N_ELEMENTS, n_feat, dtype=dtype))
        self.register_buffer("bias", torch.zeros(N_ELEMENTS, dtype=dtype))
        self.register_buffer("elem_ref", torch.zeros(N_ELEMENTS, dtype=dtype))
        self.alpha: float | None = None

    def set_reference(self, frames: list[Frame]) -> None:
        """Fit per-element reference energies by least squares on frames."""
        counts = np.stack([fr.element_counts() for fr in frames]).astype(float)
        energies = np.array([fr.energy for fr in frames])
        ref, *_ = np.linalg.lstsq(counts, energies, rcond=None)
        with torch.no_grad():
            self.elem_ref.copy_(torch.tensor(ref, dtype=self.elem_ref.dtype))

    def structure_energies(self, batch: Batch, positions: torch.Tensor) -> torch.Tensor:
        """Total energies per structure, differentiable in positions."""
        desc = compute_acsf(positions, batch, self.params)
        atom_e = (desc * self.weight[batch.species]).sum(dim=1)
        atom_e = atom_e + self.bias[batch.species] + self.elem_ref[batch.species]
        return torch.zeros(batch.n_structs, dtype=desc.dtype).index_add(
            0, batch.struct_of_atom, atom_e
        )

    def energy_and_forces(
        self, batch: Batch, create_graph: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energies (S,) and autograd forces (N, 3) for a batch."""
        positions = batch.positions.detach().clone().requires_grad_(True)
        energy = self.structure_energies(batch, positions)
        (grad,) = torch.autograd.grad(energy.sum(), positions, create_graph=create_graph)
        return energy, -grad

    def n_parameters(self) -> int:
        return int(self.weight.numel() + self.bias.numel())

    def design_rows(
        self, frames: list[Frame], chunk: int = 24
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Energy and force design matrices for the linear fit.

        Args:
            frames: Labeled frames.
            chunk: Frames per autograd chunk.

        Returns:
            (a_e, y_e, a_f, y_f): energy rows (per atom) with targets, and
            force rows with targets. Columns are [w_00..w_2F, bias_0..2].
        """
        n_feat = self.params.n_features
        n_col = N_ELEMENTS * n_feat + N_ELEMENTS
        a_e_l, y_e_l, a_f_l, y_f_l = [], [], [], []
        for c0 in range(0, len(frames), chunk):
            part = frames[c0 : c0 + chunk]
            batch = build_batch(part, self.params.rc_radial, self.params.rc_angular)
            positions = batch.positions.detach().clone().requires_grad_(True)
            desc = compute_acsf(positions, batch, self.params)
            # Per-structure per-element descriptor sums, (S, 3*n_feat).
            row_idx = batch.struct_of_atom * N_ELEMENTS + batch.species
            sums = torch.zeros(batch.n_structs * N_ELEMENTS, n_feat, dtype=desc.dtype)
            sums = sums.index_add(0, row_idx, desc).reshape(batch.n_structs, -1)
            counts = np.stack([fr.element_counts() for fr in part]).astype(float)
            n_atoms = counts.sum(axis=1)
            a_e = np.concatenate([sums.detach().numpy(), counts], axis=1) / n_atoms[:, None]
            ref_e = counts @ self.elem_ref.numpy()
            y_e = (np.array([fr.energy for fr in part]) - ref_e) / n_atoms
            a_e_l.append(a_e)
            y_e_l.append(y_e)
            # Force rows: -(d/dpos) of every weight column; bias columns are zero.
            n_pos = positions.numel()
            a_f = np.zeros((n_pos, n_col))
            for c in range(N_ELEMENTS * n_feat):
                (g,) = torch.autograd.grad(sums[:, c].sum(), positions, retain_graph=True)
                a_f[:, c] = -g.reshape(-1).numpy()
            a_f_l.append(a_f)
            y_f_l.append(np.concatenate([fr.forces for fr in part]).reshape(-1))
        return (
            np.concatenate(a_e_l),
            np.concatenate(y_e_l),
            np.concatenate(a_f_l),
            np.concatenate(y_f_l),
        )

    def fit(
        self,
        train: list[Frame],
        val: list[Frame],
        alphas: tuple[float, ...] = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2),
        force_weight: float = 0.1,
    ) -> dict:
        """Closed-form weighted ridge fit with alpha tuned on the val split.

        The objective is exactly the network training loss, mean-normalized
        per term: mean squared per-atom energy error plus force_weight times
        mean squared force error, plus alpha * |theta|^2. Without the mean
        normalization the roughly 100x more numerous force rows would weight
        forces far more heavily than the networks' loss does, making the
        baseline comparison unfair.

        Args:
            train: Training frames (energy + forces).
            val: Validation frames for alpha selection.
            alphas: Regularization strengths to try.
            force_weight: Relative weight of the mean force term, matching
                the network loss and the alpha-selection score.

        Returns:
            Fit report: chosen alpha and per-alpha validation errors.
        """
        self.set_reference(train)
        a_e, y_e, a_f, y_f = self.design_rows(train)
        ata = a_e.T @ a_e / len(y_e) + force_weight * (a_f.T @ a_f) / len(y_f)
        aty = a_e.T @ y_e / len(y_e) + force_weight * (a_f.T @ y_f) / len(y_f)
        report = {"alphas": [], "force_weight": force_weight}
        best = None
        for alpha in alphas:
            theta = np.linalg.solve(ata + alpha * np.eye(len(ata)), aty)
            self._load_theta(theta)
            from gnff.evaluate import evaluate_potential

            e_mae, f_mae = evaluate_potential(self, val)
            score = (e_mae / 1000.0) ** 2 + force_weight * (f_mae / 1000.0) ** 2
            report["alphas"].append({"alpha": alpha, "val_e_mae": e_mae, "val_f_mae": f_mae})
            if best is None or score < best[0]:
                best = (score, alpha, theta)
        _, alpha, theta = best
        self._load_theta(theta)
        self.alpha = alpha
        report["chosen_alpha"] = alpha
        return report

    def _load_theta(self, theta: np.ndarray) -> None:
        n_feat = self.params.n_features
        with torch.no_grad():
            self.weight.copy_(
                torch.tensor(
                    theta[: N_ELEMENTS * n_feat].reshape(N_ELEMENTS, n_feat),
                    dtype=self.weight.dtype,
                )
            )
            self.bias.copy_(torch.tensor(theta[N_ELEMENTS * n_feat :], dtype=self.bias.dtype))

    def to_checkpoint(self) -> dict:
        from dataclasses import asdict

        return {"kind": "ridge", "acsf": asdict(self.params), "state": self.state_dict()}

    @staticmethod
    def from_checkpoint(ckpt: dict) -> RidgePotential:
        p = ckpt["acsf"]
        params = AcsfParams(
            rc_radial=p["rc_radial"],
            rc_angular=p["rc_angular"],
            radial=[tuple(x) for x in p["radial"]],
            angular=[tuple(x) for x in p["angular"]],
        )
        model = RidgePotential(params)
        model.load_state_dict(ckpt["state"])
        return model
