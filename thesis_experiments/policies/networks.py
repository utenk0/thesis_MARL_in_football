"""Shared decentralized actor and centralized two-team value critic."""

from __future__ import annotations

import torch
from torch import nn

from thesis_experiments.transitions.observations import LOCAL_FEATURES


def _mlp(input_size: int, output_size: int, hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, hidden_size), nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )


class SharedActor(nn.Module):
    """One policy network reused for every player during decentralized execution."""

    def __init__(self, observation_size: int = LOCAL_FEATURES, action_size: int = 19, hidden_size: int = 128):
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


class SharedDisplacementPredictor(nn.Module):
    """One causal next-displacement regressor shared by all team players."""

    def __init__(self, observation_size: int = LOCAL_FEATURES, hidden_size: int = 128):
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.network = _mlp(observation_size, 2, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict the next ``(dx, dy)`` in metres for each observation."""

        return self.network(observations)


class TemporalResidualDisplacementPredictor(nn.Module):
    """Predict a correction to causal previous-velocity persistence."""

    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.network = _mlp(input_size, 2, hidden_size)
        self.reset_correction_head()

    def reset_correction_head(self) -> None:
        """Make the initial model exactly reproduce the persistence baseline."""

        output = self.network[-1]
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, observations_and_history: torch.Tensor) -> torch.Tensor:
        """Return an additive ``(dx, dy)`` correction in metres."""

        return self.network(observations_and_history)


class CentralizedCritic(nn.Module):
    """Estimate left- and right-team values from the complete 22-player state."""

    def __init__(self, state_size: int = 46, hidden_size: int = 256):
        super().__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.network = _mlp(state_size, 2, hidden_size)

    def forward(self, global_states: torch.Tensor) -> torch.Tensor:
        return self.network(global_states)
