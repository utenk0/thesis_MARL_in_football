"""Shared interface for methods evaluated as causal team-trajectory forecasts."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from thesis_experiments.training.team_multi_horizon_bc import DEFAULT_HORIZONS


@dataclass(frozen=True, slots=True)
class TrajectoryDatasetSplit:
    """Match-disjoint paths used by every trajectory method."""

    team_id: str
    train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    test_paths: tuple[Path, ...]
    train_matches: tuple[str, ...]
    validation_matches: tuple[str, ...]
    test_matches: tuple[str, ...]

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        contract_path: Path | None = None,
        project_dir: Path | None = None,
        require_files: bool = True,
    ) -> Self:
        manifest_path = Path(manifest_path).resolve()
        root = Path(project_dir).resolve() if project_dir is not None else Path.cwd().resolve()
        manifest = json.loads(manifest_path.read_text())

        def paths(split: str) -> tuple[Path, ...]:
            resolved = []
            for row in manifest["splits"][split]:
                path = Path(row["path"])
                path = path if path.is_absolute() else root / path
                path = path.resolve()
                if require_files and not path.is_file():
                    raise FileNotFoundError(path)
                resolved.append(path)
            if not resolved:
                raise ValueError(f"Trajectory split {split!r} is empty.")
            return tuple(resolved)

        def matches(split: str) -> tuple[str, ...]:
            return tuple(
                sorted({str(row["match_id"]) for row in manifest["splits"][split]})
            )

        result = cls(
            team_id=str(manifest["team_id"]),
            train_paths=paths("train"),
            validation_paths=paths("validation"),
            test_paths=paths("test"),
            train_matches=matches("train"),
            validation_matches=matches("validation"),
            test_matches=matches("test"),
        )
        result.validate_match_disjoint()
        if contract_path is not None:
            result.validate_contract(contract_path)
        return result

    def validate_match_disjoint(self) -> None:
        groups = {
            "train": set(self.train_matches),
            "validation": set(self.validation_matches),
            "test": set(self.test_matches),
        }
        overlaps = {
            f"{left}/{right}": sorted(groups[left] & groups[right])
            for index, left in enumerate(groups)
            for right in tuple(groups)[index + 1 :]
            if groups[left] & groups[right]
        }
        if overlaps:
            raise ValueError(f"Trajectory splits are not match-disjoint: {overlaps}")

    def validate_contract(self, contract_path: Path) -> None:
        from thesis_experiments.evaluation.trajectory_contract import load_contract

        contract = load_contract(Path(contract_path))
        expected = contract["split"]
        errors = []
        if self.team_id != contract["team_id"]:
            errors.append("team_id")
        for name, actual in (
            ("train", self.train_matches),
            ("validation", self.validation_matches),
            ("test", self.test_matches),
        ):
            if list(actual) != list(expected[f"{name}_matches"]):
                errors.append(f"{name}_matches")
        if errors:
            raise ValueError(
                "Dataset manifest violates the trajectory contract: " + ", ".join(errors)
            )

    def paths(self, split: str) -> tuple[Path, ...]:
        try:
            return {
                "train": self.train_paths,
                "validation": self.validation_paths,
                "test": self.test_paths,
            }[split]
        except KeyError as error:
            raise ValueError("split must be train, validation, or test") from error


@dataclass(frozen=True, slots=True)
class TrajectoryTrainingConfig:
    """Hyperparameters shared by trajectory-method adapters."""

    seed: int = 0
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    history_steps: int = 10
    observation_variant: str = "temporal_context_no_slot"
    epochs: int = 10
    batch_size: int = 2048
    learning_rate: float = 3e-4
    hidden_size: int = 128
    early_stopping_patience: int = 3
    minimum_epochs: int = 5
    device: str = "cpu"
    cql_action_count: int = 32
    cql_alpha: float = 1.0
    cql_gamma: float = 0.99
    cql_target_update_interval: int = 500
    cql_prototype_samples: int = 100_000
    cql_kmeans_iterations: int = 15
    ctde_bc_checkpoint: Path | None = None
    ctde_critic_epochs: int = 20
    ctde_gamma: float = 0.99
    ctde_temperature: float = 1.0
    ctde_max_weight: float = 20.0
    ctde_bc_coefficient: float = 0.5
    ctde_critic_hidden_size: int = 256

    def __post_init__(self) -> None:
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and increasing")
        if any(value < 1 for value in self.horizons):
            raise ValueError("horizons must be positive")
        positive = (
            self.history_steps,
            self.epochs,
            self.batch_size,
            self.hidden_size,
            self.early_stopping_patience,
        )
        if any(value < 1 for value in positive):
            raise ValueError("trajectory training parameters must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.minimum_epochs < 0 or self.minimum_epochs > self.epochs:
            raise ValueError("minimum_epochs must be between zero and epochs")
        if self.cql_action_count < 2:
            raise ValueError("cql_action_count must be at least two")
        if self.cql_alpha < 0:
            raise ValueError("cql_alpha must be non-negative")
        if not 0 <= self.cql_gamma <= 1:
            raise ValueError("cql_gamma must be between zero and one")
        if min(
            self.cql_target_update_interval,
            self.cql_prototype_samples,
            self.cql_kmeans_iterations,
        ) < 1:
            raise ValueError("CQL update and prototype parameters must be positive")
        if min(self.ctde_critic_epochs, self.ctde_critic_hidden_size) < 1:
            raise ValueError("CTDE critic parameters must be positive")
        if not 0 <= self.ctde_gamma <= 1:
            raise ValueError("ctde_gamma must be between zero and one")
        if min(
            self.ctde_temperature,
            self.ctde_max_weight,
            self.ctde_bc_coefficient,
        ) <= 0:
            raise ValueError("CTDE weighting parameters must be positive")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["horizons"] = list(self.horizons)
        if self.ctde_bc_checkpoint is not None:
            result["ctde_bc_checkpoint"] = str(self.ctde_bc_checkpoint)
        return result


@dataclass(frozen=True, slots=True)
class TrajectoryTrainingResult:
    method: str
    checkpoint: Path
    training: dict[str, Any]
    validation: dict[str, Any]
    test_loaded: bool = False
    future_observations_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "checkpoint": str(self.checkpoint),
            "training": self.training,
            "validation": self.validation,
            "test_loaded": self.test_loaded,
            "future_observations_used": self.future_observations_used,
        }


class TrajectoryMethod(ABC):
    """Contract shared by BC and every later trajectory method.

    ``predict`` receives one current 61-feature observation per player and its
    causal motion-history vector.  It returns displacement from the forecast
    origin in metres with shape ``(samples, horizons, 2)``.  No method receives
    an observation, action, velocity, or simulator state after the origin.
    """

    name: str

    @property
    @abstractmethod
    def horizons(self) -> tuple[int, ...]:
        """Direct forecast horizons implemented by the loaded checkpoint."""

    @property
    @abstractmethod
    def history_steps(self) -> int:
        """Number of causal history steps required by ``predict``."""

    @classmethod
    @abstractmethod
    def train(
        cls,
        dataset: TrajectoryDatasetSplit,
        output_dir: Path,
        config: TrajectoryTrainingConfig,
    ) -> TrajectoryTrainingResult:
        """Fit on train, select on validation, and never load test data."""

    @classmethod
    @abstractmethod
    def load(cls, checkpoint: Path, *, device: str = "cpu") -> Self:
        """Load a saved trajectory method for inference."""

    @abstractmethod
    def predict(
        self,
        observations: np.ndarray,
        causal_history: np.ndarray,
    ) -> np.ndarray:
        """Return direct relative-position forecasts in metres."""

    @abstractmethod
    def evaluate(
        self,
        paths: tuple[Path, ...] | list[Path],
        *,
        team_id: str,
    ) -> dict[str, Any]:
        """Evaluate ADE/FDE and formation metrics at the shared horizons."""
