"""Behavioural-cloning pretraining for the shared actor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from thesis_experiments.policies.networks import SharedActor
from thesis_experiments.training.data import load_arrays


def train_bc(paths: list[Path], output: Path, *, epochs: int = 20, batch_size: int = 512, learning_rate: float = 3e-4, hidden_size: int = 128, seed: int = 0) -> dict[str, float]:
    torch.manual_seed(seed); np.random.seed(seed)
    arrays = load_arrays(paths)
    observations = torch.from_numpy(arrays["local_observations"].reshape(-1, arrays["local_observations"].shape[-1])).float()
    actions = torch.from_numpy(arrays["actions"].reshape(-1)).long()
    actor = SharedActor(observations.shape[-1], hidden_size=hidden_size)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    final_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(actions))
        for start in range(0, len(actions), batch_size):
            indices = permutation[start:start + batch_size]
            loss = F.cross_entropy(actor(observations[indices]), actions[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            final_loss = float(loss.detach())
    with torch.no_grad():
        accuracy = float((actor(observations).argmax(-1) == actions).float().mean())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor.state_dict(), "observation_size": actor.observation_size, "action_size": actor.action_size, "hidden_size": hidden_size, "stage": "bc"}, output)
    return {"loss": final_loss, "accuracy": accuracy, "samples": float(len(actions))}
