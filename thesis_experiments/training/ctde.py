"""Offline centralized-critic training and BC-anchored actor fine-tuning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from thesis_experiments.policies.networks import CentralizedCritic, SharedActor
from thesis_experiments.training.data import load_arrays


def train_ctde(paths: list[Path], bc_checkpoint: Path, output: Path, *, critic_epochs: int = 30, actor_epochs: int = 10, batch_size: int = 256, learning_rate: float = 3e-4, gamma: float = 0.99, temperature: float = 1.0, max_weight: float = 20.0, bc_coefficient: float = 0.5, critic_hidden_size: int = 256, seed: int = 0) -> dict[str, float]:
    torch.manual_seed(seed); np.random.seed(seed)
    arrays = load_arrays(paths)
    states = torch.from_numpy(arrays["global_states"]).float()
    next_states = torch.from_numpy(arrays["next_global_states"]).float()
    rewards = torch.from_numpy(arrays["team_rewards"]).float()
    dones = torch.from_numpy(arrays["dones"].astype(np.float32)).float().unsqueeze(-1)
    checkpoint = torch.load(bc_checkpoint, map_location="cpu", weights_only=True)
    actor = SharedActor(checkpoint["observation_size"], checkpoint["action_size"], checkpoint["hidden_size"])
    actor.load_state_dict(checkpoint["actor_state_dict"])
    critic = CentralizedCritic(states.shape[-1], hidden_size=critic_hidden_size)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    final_critic_loss = 0.0
    for _ in range(critic_epochs):
        permutation = torch.randperm(len(states))
        for start in range(0, len(states), batch_size):
            indices = permutation[start:start + batch_size]
            with torch.no_grad():
                target = rewards[indices] + gamma * (1 - dones[indices]) * critic(next_states[indices])
            loss = F.mse_loss(critic(states[indices]), target)
            critic_optimizer.zero_grad(); loss.backward(); critic_optimizer.step()
            final_critic_loss = float(loss.detach())
    with torch.no_grad():
        advantages = rewards + gamma * (1 - dones) * critic(next_states) - critic(states)
        agent_advantages = torch.cat([advantages[:, 0:1].expand(-1, 11), advantages[:, 1:2].expand(-1, 11)], dim=1).reshape(-1)
        weights = torch.exp(agent_advantages / temperature).clamp(max=max_weight)
    observations = torch.from_numpy(arrays["local_observations"].reshape(-1, arrays["local_observations"].shape[-1])).float()
    actions = torch.from_numpy(arrays["actions"].reshape(-1)).long()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    final_actor_loss = 0.0
    for _ in range(actor_epochs):
        permutation = torch.randperm(len(actions))
        for start in range(0, len(actions), batch_size):
            indices = permutation[start:start + batch_size]
            nll = F.cross_entropy(actor(observations[indices]), actions[indices], reduction="none")
            loss = (weights[indices] * nll).mean() + bc_coefficient * nll.mean()
            actor_optimizer.zero_grad(); loss.backward(); actor_optimizer.step()
            final_actor_loss = float(loss.detach())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor_state_dict": actor.state_dict(), "critic_state_dict": critic.state_dict(), "observation_size": actor.observation_size, "action_size": actor.action_size, "actor_hidden_size": actor.hidden_size, "state_size": critic.state_size, "critic_hidden_size": critic_hidden_size, "stage": "ctde", "gamma": gamma}, output)
    return {"critic_loss": final_critic_loss, "actor_loss": final_actor_loss, "mean_advantage": float(advantages.mean()), "transitions": float(len(states))}
