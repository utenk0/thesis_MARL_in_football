"""Shared decentralized actor and centralized two-team value critic."""

from __future__ import annotations

import torch
from torch import nn


def _mlp(input_size: int, output_size: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class SharedActor(nn.Module):
    """One policy network reused for every player during decentralized execution."""

    def __init__(self, observation_size: int = 24, action_size: int = 19, hidden_size: int = 128):
        super().__init__()
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.network = _mlp(observation_size, action_size, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)

    @torch.no_grad()
    def act(self, observations: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        logits = self(observations)
        if deterministic:
            return logits.argmax(dim=-1)
        return torch.distributions.Categorical(logits=logits).sample()


class CentralizedCritic(nn.Module):
    """Estimate left- and right-team values from the complete 22-player state."""

    def __init__(self, state_size: int = 46, hidden_size: int = 256):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.network = _mlp(state_size, 2, hidden_size)

    def forward(self, global_states: torch.Tensor) -> torch.Tensor:
        return self.network(global_states)
