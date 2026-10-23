"""Team-level policy composition for 22-player football control."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np
import torch

from thesis_experiments.policies.networks import SharedActor

PLAYERS_PER_TEAM = 11
PLAYERS_PER_MATCH = 22


class PlayerPolicy(Protocol):
    """Policy object that maps local player observations to GRF actions."""

    def act(self, observations: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        """Return one discrete GRF action per observation."""


@dataclass(slots=True)
class TeamPolicyOutput:
    """Actions and optional team-level message produced by a team policy."""

    actions: np.ndarray
    message: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TeamPolicy:
    """A team policy owns the 11 player policies for one side.

    The player policies are implementation details of the team policy. Training
    and simulation should treat the team policy as the communicating unit.
    """

    name: str
    player_policies: Sequence[PlayerPolicy]

    def __post_init__(self) -> None:
        if len(self.player_policies) != PLAYERS_PER_TEAM:
            raise ValueError(f"{self.name} must contain {PLAYERS_PER_TEAM} player policies.")

    @classmethod
    def shared(cls, name: str, policy: PlayerPolicy) -> "TeamPolicy":
        """Create a team where all players reuse the same player policy."""

        return cls(name=name, player_policies=[policy] * PLAYERS_PER_TEAM)

    def act(
        self,
        observations: np.ndarray,
        *,
        deterministic: bool = True,
        opponent_message: dict[str, Any] | None = None,
    ) -> TeamPolicyOutput:
        """Return 11 actions for one team's local observations."""

        team_observations = np.asarray(observations, dtype=np.float32)
        if team_observations.ndim != 2 or team_observations.shape[0] != PLAYERS_PER_TEAM:
            raise ValueError(
                f"{self.name} observations must have shape [{PLAYERS_PER_TEAM}, features], "
                f"got {team_observations.shape}."
            )
        unique_policy_ids = {id(policy) for policy in self.player_policies}
        if len(unique_policy_ids) == 1:
            actions = self.player_policies[0].act(team_observations, deterministic=deterministic)
        else:
            actions = np.asarray([
                policy.act(team_observations[index:index + 1], deterministic=deterministic)[0]
                for index, policy in enumerate(self.player_policies)
            ])
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape != (PLAYERS_PER_TEAM,):
            raise ValueError(f"{self.name} produced {actions.shape}, expected ({PLAYERS_PER_TEAM},).")
        return TeamPolicyOutput(actions=actions, message={"team": self.name, "opponent_message": opponent_message or {}})


@dataclass(slots=True)
class TwoTeamPolicy:
    """Compose left and right team policies into one 22-player joint policy."""

    left: TeamPolicy
    right: TeamPolicy

    def act(self, observations: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        joint_observations = np.asarray(observations, dtype=np.float32)
        if joint_observations.ndim != 2 or joint_observations.shape[0] != PLAYERS_PER_MATCH:
            raise ValueError(
                f"Joint observations must have shape [{PLAYERS_PER_MATCH}, features], "
                f"got {joint_observations.shape}."
            )
        left_output = self.left.act(joint_observations[:PLAYERS_PER_TEAM], deterministic=deterministic)
        right_output = self.right.act(
            joint_observations[PLAYERS_PER_TEAM:],
            deterministic=deterministic,
            opponent_message=left_output.message,
        )
        return np.concatenate([left_output.actions, right_output.actions]).astype(np.int64)


class TorchActorPolicy:
    """Wrap a project PyTorch actor behind the player-policy protocol."""

    def __init__(
        self,
        actor: SharedActor,
        *,
        device: str = "cpu",
        feature_mask: np.ndarray | torch.Tensor | None = None,
    ):
        self.actor = actor.to(device)
        self.actor.eval()
        self.device = device
        if feature_mask is None:
            feature_mask = np.ones(actor.observation_size, dtype=np.float32)
        mask = torch.as_tensor(feature_mask, dtype=torch.float32, device=device).reshape(-1)
        if mask.shape != (actor.observation_size,):
            raise ValueError(
                f"Feature mask has shape {tuple(mask.shape)}, expected "
                f"({actor.observation_size},)."
            )
        self.feature_mask = mask

    @torch.no_grad()
    def act(self, observations: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        tensor = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        tensor = tensor * self.feature_mask
        actions = self.actor.act(tensor, deterministic=deterministic)
        return actions.cpu().numpy().astype(np.int64)


def load_shared_actor_policy(checkpoint_path: Path, *, device: str = "cpu") -> TorchActorPolicy:
    """Load a BC or CTDE shared-actor checkpoint as a player policy."""

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hidden_size = int(checkpoint.get("hidden_size", checkpoint.get("actor_hidden_size", 128)))
    actor = SharedActor(
        observation_size=int(checkpoint["observation_size"]),
        action_size=int(checkpoint["action_size"]),
        hidden_size=hidden_size,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    return TorchActorPolicy(
        actor,
        device=device,
        feature_mask=checkpoint.get("feature_mask"),
    )


def load_two_team_shared_actor_policy(checkpoint_path: Path, *, device: str = "cpu") -> TwoTeamPolicy:
    """Use one trained shared actor for both teams and all players."""

    player_policy = load_shared_actor_policy(checkpoint_path, device=device)
    return TwoTeamPolicy(
        left=TeamPolicy.shared("left", player_policy),
        right=TeamPolicy.shared("right", player_policy),
    )
