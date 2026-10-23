"""Unified training method registry."""

from thesis_experiments.methods.base import FootballMethod, MethodConfig, MethodResult
from thesis_experiments.methods.registry import available_methods, build_method
from thesis_experiments.methods.trajectory import (
    TrajectoryDatasetSplit,
    TrajectoryMethod,
    TrajectoryTrainingConfig,
    TrajectoryTrainingResult,
)
from thesis_experiments.methods.trajectory_registry import (
    available_trajectory_methods,
    load_trajectory_method,
    train_trajectory_method,
)
from thesis_experiments.methods.trajectory_cql import TrajectoryCQLMethod
from thesis_experiments.methods.trajectory_ctde import TrajectoryCTDEMethod

__all__ = [
    "FootballMethod",
    "MethodConfig",
    "MethodResult",
    "TrajectoryDatasetSplit",
    "TrajectoryMethod",
    "TrajectoryTrainingConfig",
    "TrajectoryTrainingResult",
    "TrajectoryCQLMethod",
    "TrajectoryCTDEMethod",
    "available_methods",
    "available_trajectory_methods",
    "build_method",
    "load_trajectory_method",
    "train_trajectory_method",
]
