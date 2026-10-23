"""Discrete, return-conditioned causal Decision Transformer (Chen et al., 2021)."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from thesis_experiments.simulation.gymnasium_replay import flatten_player_transitions, player_episode_indices
from thesis_experiments.training.data import load_arrays
from thesis_experiments.transitions.observations import LOCAL_FEATURES


class TrajectoryWindows(Dataset):
    """Non-overlapping windows never cross a match/clip or change player."""

    def __init__(self, paths, context_length=16):
        arrays = load_arrays(paths)
        self.batch = flatten_player_transitions(arrays)
        self.episodes = player_episode_indices(arrays)
        self.context_length = context_length
        self.returns = np.zeros(len(self.batch.actions), dtype=np.float32)
        self.windows = []
        for indices in self.episodes:
            self.returns[indices] = np.cumsum(self.batch.rewards[indices][::-1])[::-1]
            for start in range(0, len(indices), context_length):
                self.windows.append((indices[start:start + context_length], start))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        indices, start = self.windows[index]
        count, length = len(indices), self.context_length
        obs = np.zeros((length, self.batch.observation_size), dtype=np.float32)
        actions = np.zeros(length, dtype=np.int64)
        returns = np.zeros((length, 1), dtype=np.float32)
        mask = np.arange(length) < count
        obs[:count] = self.batch.observations[indices]
        actions[:count] = self.batch.actions[indices]
        returns[:count, 0] = self.returns[indices]
        return obs, actions, returns, np.arange(start, start + length), mask


class DecisionTransformer(nn.Module):
    def __init__(self, observation_size=LOCAL_FEATURES, hidden_size=128, context_length=16, max_timestep=16384):
        super().__init__()
        if hidden_size % 4:
            raise ValueError("Decision Transformer hidden_size must be divisible by 4.")
        self.config = dict(observation_size=observation_size, hidden_size=hidden_size,
                           context_length=context_length, max_timestep=max_timestep)
        self.state_embedding = nn.Linear(observation_size, hidden_size)
        self.return_embedding = nn.Linear(1, hidden_size)
        self.action_embedding = nn.Embedding(19, hidden_size)
        self.time_embedding = nn.Embedding(max_timestep, hidden_size)
        layer = nn.TransformerEncoderLayer(hidden_size, 4, hidden_size * 4,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, 19)

    def forward(self, observations, actions, returns, timesteps):
        time = self.time_embedding(timesteps.clamp(max=self.config["max_timestep"] - 1))
        # R_t, s_t, a_t: predicting from s_t cannot see a_t or future tokens.
        tokens = torch.stack((self.return_embedding(returns) + time,
                              self.state_embedding(observations) + time,
                              self.action_embedding(actions) + time), dim=2)
        tokens = tokens.flatten(1, 2)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1],
                                    dtype=torch.bool, device=tokens.device), diagonal=1)
        encoded = self.transformer(self.norm(tokens), mask=mask)
        return self.action_head(encoded[:, 1::3])


def train_decision_transformer(paths, output, *, epochs=20, batch_size=64,
                               learning_rate=3e-4, hidden_size=128, seed=0,
                               context_length=16, device="cpu"):
    torch.manual_seed(seed)
    dataset = TrajectoryWindows(paths, context_length)
    model = DecisionTransformer(dataset.batch.observation_size, hidden_size, context_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = []
    for _ in range(epochs):
        total_loss = count = 0
        model.train()
        for batch in loader:
            obs, actions, returns, timesteps, valid = [value.to(device) for value in batch]
            logits = model(obs, actions, returns, timesteps)
            loss = F.cross_entropy(logits[valid], actions[valid])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * valid.sum().item()
            count += valid.sum().item()
        history.append(total_loss / count)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = float(np.quantile([dataset.returns[i[0]] for i in dataset.episodes], 0.75))
    torch.save({"model_state_dict": model.state_dict(), "config": model.config,
                "target_return": target, "loss_history": history, "stage": "dt"}, output)
    return {"initial_loss": history[0], "loss": history[-1],
            "samples": float(len(dataset.batch.actions)), "target_return": target}


class DecisionTransformerPolicy:
    def __init__(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        self.model = DecisionTransformer(**checkpoint["config"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(device).eval()
        self.device = device
        self.target_return = checkpoint["target_return"]
        self.reset()

    def reset(self):
        self.observations, self.actions, self.returns = [], [], []
        self.remaining = None
        self.timestep = 0

    def observe_rewards(self, rewards):
        if self.remaining is not None:
            self.remaining -= np.asarray(rewards, dtype=np.float32)

    @torch.no_grad()
    def act(self, observations, *, deterministic=True):
        observations = np.asarray(observations, dtype=np.float32)
        if self.remaining is None:
            self.remaining = np.full(len(observations), self.target_return, dtype=np.float32)
        if len(observations) != len(self.remaining):
            raise ValueError("Reset the sequence policy when the number of controlled players changes.")
        self.observations.append(observations.copy())
        self.returns.append(self.remaining.copy())
        self.actions.append(np.zeros(len(observations), dtype=np.int64))
        length = self.model.config["context_length"]
        self.observations = self.observations[-length:]
        self.returns = self.returns[-length:]
        self.actions = self.actions[-length:]
        n = len(self.actions)
        obs = torch.as_tensor(np.stack(self.observations, axis=1), device=self.device)
        actions = torch.as_tensor(np.stack(self.actions, axis=1), device=self.device)
        returns = torch.as_tensor(np.stack(self.returns, axis=1)[..., None], device=self.device)
        times = torch.arange(self.timestep - n + 1, self.timestep + 1, device=self.device).expand(len(obs), -1)
        logits = self.model(obs, actions, returns, times)[:, -1]
        chosen = logits.argmax(-1) if deterministic else torch.distributions.Categorical(logits=logits).sample()
        self.actions[-1] = chosen.cpu().numpy()
        self.timestep += 1
        return self.actions[-1].copy()
