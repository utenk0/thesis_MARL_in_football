"""Behavioural-cloning pretraining for the shared actor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from thesis_experiments.policies.networks import SharedActor
from thesis_experiments.training.data import load_arrays
from thesis_experiments.transitions.observations import (
    ACTION_COUNT,
    LOCAL_FEATURES,
    observation_feature_mask,
)


def train_bc(
    paths: list[Path],
    output: Path,
    *,
    epochs: int = 20,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    hidden_size: int = 128,
    seed: int = 0,
    device: str = "cpu",
    observation_variant: str = "full",
    class_weighting: str = "none",
    max_class_weight: float = 20.0,
) -> dict[str, object]:
    """Train BC with reproducible observation and class-balance controls."""

    if class_weighting not in {"none", "inverse_frequency"}:
        raise ValueError("class_weighting must be 'none' or 'inverse_frequency'.")
    if max_class_weight <= 0:
        raise ValueError("max_class_weight must be positive.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    arrays = load_arrays(paths)
    flat_observations = arrays["local_observations"].reshape(
        -1, arrays["local_observations"].shape[-1]
    )
    if flat_observations.shape[-1] == LOCAL_FEATURES:
        feature_mask = observation_feature_mask(observation_variant)
    elif observation_variant == "full":
        # Preserve compatibility with legacy 24-feature fixtures/checkpoints.
        feature_mask = np.ones(flat_observations.shape[-1], dtype=bool)
    else:
        raise ValueError(
            f"Observation variant {observation_variant!r} requires "
            f"{LOCAL_FEATURES}-feature data."
        )
    observations = torch.from_numpy(flat_observations).float().to(device)
    mask_tensor = torch.from_numpy(feature_mask.astype(np.float32)).to(device)
    observations = observations * mask_tensor
    actions = torch.from_numpy(arrays["actions"].reshape(-1)).long().to(device)
    actor = SharedActor(observations.shape[-1], hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    class_counts = torch.bincount(actions, minlength=ACTION_COUNT)
    class_weights = torch.ones(ACTION_COUNT, dtype=torch.float32, device=device)
    if class_weighting == "inverse_frequency":
        observed = class_counts > 0
        class_weights.zero_()
        class_weights[observed] = len(actions) / (
            observed.sum() * class_counts[observed].float()
        )
        class_weights.clamp_(max=max_class_weight)
    final_loss = 0.0
    for _ in range(epochs):
        permutation = torch.randperm(len(actions), device=device)
        for start in range(0, len(actions), batch_size):
            indices = permutation[start:start + batch_size]
            loss = F.cross_entropy(
                actor(observations[indices]),
                actions[indices],
                weight=class_weights if class_weighting == "inverse_frequency" else None,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach())
    with torch.no_grad():
        accuracy = float((actor(observations).argmax(-1) == actions).float().mean())
    output.parent.mkdir(parents=True, exist_ok=True)
    counts_list = class_counts.detach().cpu().tolist()
    weights_list = class_weights.detach().cpu().tolist()
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "observation_size": actor.observation_size,
            "action_size": actor.action_size,
            "hidden_size": hidden_size,
            "stage": "bc",
            "observation_variant": observation_variant,
            "feature_mask": torch.from_numpy(feature_mask),
            "class_weighting": class_weighting,
            "max_class_weight": float(max_class_weight),
            "class_counts": torch.tensor(counts_list, dtype=torch.int64),
            "class_weights": torch.tensor(weights_list, dtype=torch.float32),
        },
        output,
    )
    return {
        "loss": final_loss,
        "accuracy": accuracy,
        "samples": float(len(actions)),
        "observation_variant": observation_variant,
        "retained_features": int(feature_mask.sum()),
        "class_weighting": class_weighting,
        "max_effective_class_weight": float(max(weights_list)),
        "observed_action_classes": int(sum(count > 0 for count in counts_list)),
    }
