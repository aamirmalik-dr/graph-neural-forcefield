"""gnff: from-scratch graph neural network potential for BCC TiZrNb.

A SchNet-style continuous-filter message-passing potential with autograd
forces, benchmarked head to head against descriptor baselines (a compact
ACSF network and a ridge regression on the same descriptors) at matched
training data and budget. Training labels are surrogate model labels from a
pretrained universal potential, never DFT; real DFT enters only through
Materials Project validation anchors.
"""

from gnff.checkpoints import load_checkpoint, save_checkpoint
from gnff.evaluate import error_summary, evaluate_potential, predict
from gnff.frames import Frame, read_extxyz, write_extxyz
from gnff.graphs import Batch, BatchCache, build_batch
from gnff.models import AcsfNet, AcsfParams, MpnnConfig, MpnnPotential, RidgePotential
from gnff.splits import Split, group_split, random_frame_split
from gnff.train import TrainSettings, train_potential

__version__ = "0.1.1"

__all__ = [
    "AcsfNet",
    "AcsfParams",
    "Batch",
    "BatchCache",
    "Frame",
    "MpnnConfig",
    "MpnnPotential",
    "RidgePotential",
    "Split",
    "TrainSettings",
    "build_batch",
    "error_summary",
    "evaluate_potential",
    "group_split",
    "load_checkpoint",
    "predict",
    "random_frame_split",
    "read_extxyz",
    "save_checkpoint",
    "train_potential",
    "write_extxyz",
]
