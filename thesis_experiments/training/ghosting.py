"""Small PyTorch ghosting baseline over player-local observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from thesis_experiments.transitions.observations import LOCAL_FEATURES

from thesis_experiments.datasets.movement import movement_actions
from thesis_experiments.training.data import load_arrays


class GhostingNetwork(nn.Module):
    """Predict the next player position from a local player observation."""

    def __init__(self, observation_size: int = LOCAL_FEATURES, hidden_size: int = 128):
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.network = nn.Sequential(
            nn.Linear(observation_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, :2] + self.network(observations)


def train_ghosting(
    paths: list[Path],
    output: Path,
    *,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    hidden_size: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Train a next-position ghosting model and save a checkpoint."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    arrays = load_arrays(paths)
    observations = torch.from_numpy(
        arrays["local_observations"].reshape(-1, arrays["local_observations"].shape[-1])
    ).float()
    target_xy = torch.from_numpy(
        arrays["next_local_observations"].reshape(-1, arrays["next_local_observations"].shape[-1])[:, :2]
    ).float()
    model = GhostingNetwork(observations.shape[-1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(observations))
        for start in range(0, len(observations), batch_size):
            indices = permutation[start:start + batch_size]
            loss = F.mse_loss(model(observations[indices]), target_xy[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    with torch.no_grad():
        predictions = model(observations)
        mse = float(F.mse_loss(predictions, target_xy).detach())
        predicted_actions = ghosting_actions(observations.numpy(), predictions.numpy())
        expert_actions = arrays["actions"].reshape(-1)
        action_accuracy = float((predicted_actions == expert_actions).mean())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "observation_size": model.observation_size,
            "hidden_size": hidden_size,
            "stage": "ghosting",
            "predicts_displacement": True,
        },
        output,
    )
    return {
        "loss": final_loss,
        "position_mse": mse,
        "movement_action_accuracy": action_accuracy,
        "samples": float(len(observations)),
    }


def load_ghosting_policy(checkpoint_path: Path) -> GhostingNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = GhostingNetwork(checkpoint["observation_size"], checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    if not checkpoint.get("predicts_displacement", False):
        # Preserve compatibility with the earlier absolute-position baseline.
        model.forward = model.network.forward
    model.eval()
    return model


def ghosting_actions(observations: np.ndarray, predicted_next_xy: np.ndarray, *, threshold: float = 1e-4) -> np.ndarray:
    """Convert predicted next positions into GRF movement actions."""

    observations = np.asarray(observations, dtype=np.float32)
    predicted_next_xy = np.asarray(predicted_next_xy, dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] < 2:
        raise ValueError(f"Expected observations with shape [N,L>=2], got {observations.shape}.")
    if predicted_next_xy.shape != (len(observations), 2):
        raise ValueError(f"Expected predicted_next_xy shape {(len(observations), 2)}, got {predicted_next_xy.shape}.")
    delta = predicted_next_xy - observations[:, :2]
    if observations.shape[1] >= 7:
        delta[observations[:, 6] < 0] *= -1  # GRF right-side controls are rotated.
    return movement_actions(np.zeros_like(delta), delta, threshold=threshold).astype(np.int64)
