"""Shared method interface for football learning backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from thesis_experiments.simulation.gymnasium_replay import PlayerTransitionBatch, load_player_transitions


class OptionalBackendUnavailable(RuntimeError):
    """Raised when a configured method backend is not installed."""


@dataclass(slots=True)
class MethodConfig:
    method: str
    data_paths: list[Path]
    output_dir: Path
    seed: int = 0
    total_timesteps: int = 10_000
    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 3e-4
    hidden_size: int = 128
    backend: str = "auto"
    device: str = "cpu"
    bc_checkpoint: Path | None = None
    eval_episodes: int = 5
    env_backend: str = "grf"
    grf_python: Path | None = None
    rollout_steps: int = 128
    episode_steps: int = 128
    context_length: int = 16
    extra: dict[str, Any] = field(default_factory=dict)

    def method_dir(self) -> Path:
        return self.output_dir / self.method


@dataclass(slots=True)
class MethodResult:
    method: str
    status: str
    backend: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FootballMethod(ABC):
    """Common train/evaluate/imitate surface for all selected methods."""

    name: str
    backend_name: str

    def __init__(self, config: MethodConfig):
        for name in ("epochs", "batch_size", "total_timesteps", "hidden_size", "rollout_steps", "episode_steps", "context_length"):
            if getattr(config, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if config.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.config = config
        self.output_dir = config.method_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self) -> MethodResult:
        """Train or prepare the method and return metrics/artifacts."""

    def evaluate(self) -> dict[str, float]:
        """Evaluate the method if it has a directly callable policy."""

        data = self.load_player_data()
        return action_classification_metrics(self.imitate(data.observations), data.actions)

    def imitate(self, observations: np.ndarray) -> np.ndarray:
        """Predict GRF actions for a batch of local observations."""

        from thesis_experiments.policies.loading import load_method_policy
        if not hasattr(self, "_policy"):
            self._policy = load_method_policy(self.output_dir / "result.json", device=self.config.device)
        if hasattr(self._policy, "reset"):
            self._policy.reset()
        return self._policy.act(observations)

    def load_player_data(self) -> PlayerTransitionBatch:
        return load_player_transitions(self.config.data_paths)

    def write_result(self, result: MethodResult) -> MethodResult:
        config_path = self.output_dir / "config.json"
        config_path.write_text(json.dumps(asdict(self.config), default=str, indent=2), encoding="utf-8")
        result.artifacts["config"] = str(config_path)
        path = self.output_dir / "result.json"
        result.artifacts.setdefault("result", str(path))
        path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result


def action_classification_metrics(predicted: np.ndarray, expert: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted).reshape(-1)
    expert = np.asarray(expert).reshape(-1)
    if predicted.shape != expert.shape:
        raise ValueError(f"Predicted and expert actions must align, got {predicted.shape} and {expert.shape}.")
    return {
        "action_accuracy": float((predicted == expert).mean()),
        "samples": float(len(expert)),
    }


def require_import(module_name: str, install_hint: str):
    try:
        return __import__(module_name, fromlist=["*"])
    except ImportError as exc:
        raise OptionalBackendUnavailable(f"Missing optional backend '{module_name}'. {install_hint}") from exc
