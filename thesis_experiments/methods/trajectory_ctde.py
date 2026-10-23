"""Trajectory-method adapter for centralized training/decentralized execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import numpy as np

from thesis_experiments.methods.trajectory import (
    TrajectoryDatasetSplit,
    TrajectoryMethod,
    TrajectoryTrainingConfig,
    TrajectoryTrainingResult,
)
from thesis_experiments.training.team_displacement_bc import METERS_PER_GRF
from thesis_experiments.training.team_multi_horizon_bc import _predict_corrections
from thesis_experiments.training.team_trajectory_ctde import (
    _load_model,
    evaluate_team_trajectory_ctde,
    train_team_trajectory_ctde,
)
from thesis_experiments.transitions.observations import (
    LOCAL_FEATURES,
    LOCAL_FEATURE_SLICES,
)


class TrajectoryCTDEMethod(TrajectoryMethod):
    """BC-initialized trajectory actor with a training-only global critic."""

    name = "ctde"

    def __init__(self, checkpoint: Path, *, device: str = "cpu"):
        self.checkpoint = Path(checkpoint).resolve()
        self.device = device
        self.actor, self.critic, self.input_mask, self.saved = _load_model(
            self.checkpoint, device
        )
        self._horizons = tuple(int(value) for value in self.saved["horizons"])
        self._history_steps = int(self.saved["history_steps"])

    @property
    def horizons(self) -> tuple[int, ...]:
        return self._horizons

    @property
    def history_steps(self) -> int:
        return self._history_steps

    @classmethod
    def train(
        cls,
        dataset: TrajectoryDatasetSplit,
        output_dir: Path,
        config: TrajectoryTrainingConfig,
    ) -> TrajectoryTrainingResult:
        if config.ctde_bc_checkpoint is None:
            raise ValueError(
                "Trajectory CTDE requires a frozen trajectory-BC checkpoint."
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = output_dir / "single_team_trajectory_ctde.pt"
        training = train_team_trajectory_ctde(
            list(dataset.train_paths),
            config.ctde_bc_checkpoint,
            checkpoint,
            team_id=dataset.team_id,
            validation_paths=list(dataset.validation_paths),
            horizons=config.horizons,
            actor_epochs=config.epochs,
            critic_epochs=config.ctde_critic_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            actor_hidden_size=config.hidden_size,
            critic_hidden_size=config.ctde_critic_hidden_size,
            history_steps=config.history_steps,
            seed=config.seed,
            device=config.device,
            observation_variant=config.observation_variant,
            early_stopping_patience=config.early_stopping_patience,
            minimum_epochs=config.minimum_epochs,
            gamma=config.ctde_gamma,
            temperature=config.ctde_temperature,
            max_weight=config.ctde_max_weight,
            bc_coefficient=config.ctde_bc_coefficient,
        )
        validation = evaluate_team_trajectory_ctde(
            checkpoint,
            list(dataset.validation_paths),
            team_id=dataset.team_id,
            device=config.device,
        )
        result = TrajectoryTrainingResult(
            method=cls.name,
            checkpoint=checkpoint,
            training=training,
            validation=validation,
        )
        payload = result.to_dict()
        payload["config"] = config.to_dict()
        payload["split_matches"] = {
            "train": list(dataset.train_matches),
            "validation": list(dataset.validation_matches),
            "test_declared_but_not_loaded": list(dataset.test_matches),
        }
        (output_dir / "trajectory_result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        return result

    @classmethod
    def load(cls, checkpoint: Path, *, device: str = "cpu") -> Self:
        return cls(checkpoint, device=device)

    def predict(
        self,
        observations: np.ndarray,
        causal_history: np.ndarray,
    ) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        causal_history = np.asarray(causal_history, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != LOCAL_FEATURES:
            raise ValueError(
                f"observations must have shape (samples, {LOCAL_FEATURES}), "
                f"found {observations.shape}"
            )
        expected_history = self.input_mask.shape[0] - LOCAL_FEATURES
        if causal_history.ndim != 2 or causal_history.shape != (
            len(observations),
            expected_history,
        ):
            raise ValueError(
                "causal_history must align with observations and contain "
                f"{expected_history} features, found {causal_history.shape}"
            )
        inputs = np.concatenate([observations, causal_history], axis=-1)
        corrections = _predict_corrections(
            self.actor,
            inputs,
            self.input_mask,
            batch_size=16_384,
            device=self.device,
        )
        velocity = (
            observations[:, LOCAL_FEATURE_SLICES["player_velocity"]]
            * METERS_PER_GRF
        )
        horizon_scale = np.asarray(self.horizons, dtype=np.float32)[None, :, None]
        return (velocity[:, None, :] * horizon_scale + corrections).astype(np.float32)

    def evaluate(
        self,
        paths: tuple[Path, ...] | list[Path],
        *,
        team_id: str,
    ) -> dict[str, object]:
        return evaluate_team_trajectory_ctde(
            self.checkpoint,
            list(paths),
            team_id=team_id,
            device=self.device,
        )
