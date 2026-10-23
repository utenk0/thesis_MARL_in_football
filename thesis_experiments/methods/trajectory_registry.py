"""Registry for methods that implement the shared trajectory contract."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from thesis_experiments.methods.trajectory import (
    TrajectoryDatasetSplit,
    TrajectoryMethod,
    TrajectoryTrainingConfig,
    TrajectoryTrainingResult,
)


TRAJECTORY_METHODS: dict[str, tuple[str, str]] = {
    "bc": (
        "thesis_experiments.methods.trajectory_bc",
        "MultiHorizonBCMethod",
    ),
    "cql": (
        "thesis_experiments.methods.trajectory_cql",
        "TrajectoryCQLMethod",
    ),
    "ctde": (
        "thesis_experiments.methods.trajectory_ctde",
        "TrajectoryCTDEMethod",
    ),
}


def available_trajectory_methods() -> list[str]:
    return sorted(TRAJECTORY_METHODS)


def trajectory_method_class(name: str) -> type[TrajectoryMethod]:
    try:
        module_name, class_name = TRAJECTORY_METHODS[name]
    except KeyError as error:
        choices = ", ".join(available_trajectory_methods())
        raise ValueError(f"Unknown trajectory method {name!r}. Choose from: {choices}.") from error
    return getattr(import_module(module_name), class_name)


def load_trajectory_method(
    name: str,
    checkpoint: Path,
    *,
    device: str = "cpu",
) -> TrajectoryMethod:
    return trajectory_method_class(name).load(checkpoint, device=device)


def train_trajectory_method(
    name: str,
    dataset: TrajectoryDatasetSplit,
    output_dir: Path,
    config: TrajectoryTrainingConfig,
) -> TrajectoryTrainingResult:
    return trajectory_method_class(name).train(dataset, output_dir, config)
