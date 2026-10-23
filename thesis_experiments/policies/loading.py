"""Reload every method's saved policy through one batch-action interface."""

import json
from pathlib import Path

import numpy as np


class PredictPolicy:
    def __init__(self, model, *, sb3=True):
        self.model, self.sb3 = model, sb3

    def act(self, observations, *, deterministic=True):
        if self.sb3:
            return self.model.predict(observations, deterministic=deterministic)[0].astype(np.int64)
        return self.model.predict(observations).astype(np.int64)


class GhostingPolicy:
    def __init__(self, path):
        from thesis_experiments.training.ghosting import load_ghosting_policy
        self.model = load_ghosting_policy(path)

    def act(self, observations, *, deterministic=True):
        import torch
        from thesis_experiments.training.ghosting import ghosting_actions
        obs = np.asarray(observations, dtype=np.float32)
        with torch.no_grad():
            positions = self.model(torch.from_numpy(obs)).numpy()
        return ghosting_actions(obs, positions)


def _load_imitation_actor_critic(path, *, device="cpu"):
    """Load the trusted policy file written by ``imitation.bc.BC``.

    Stable-Baselines3 2.2 calls ``torch.load`` without specifying
    ``weights_only``. PyTorch 2.6 changed that default to true, but this policy
    format also stores its Gymnasium spaces and constructor data. The artifact
    is produced locally by this project, so explicitly loading the complete
    trusted payload is required for this older SB3 serialization format.
    """

    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.utils import get_device

    resolved_device = get_device(device)
    try:
        saved = torch.load(path, map_location=resolved_device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        saved = torch.load(path, map_location=resolved_device)
    model = ActorCriticPolicy(**saved["data"])
    model.load_state_dict(saved["state_dict"])
    model.to(resolved_device)
    return model


def load_method_policy(result_path, *, device="cpu"):
    result_path = Path(result_path).resolve()
    result = json.loads(result_path.read_text())
    if result["status"] != "trained":
        raise ValueError("Only trained method results can be loaded")
    artifact = result["artifacts"].get("checkpoint", result["artifacts"].get("policy"))
    # Checkpoints are stored beside result.json so directories remain portable.
    path = result_path.parent / Path(artifact).name
    if not path.exists():
        raise FileNotFoundError(path)
    method = result["method"]
    if method in {"bc", "ctde"} and result["backend"] != "imitation":
        from thesis_experiments.policies.team import load_shared_actor_policy
        return load_shared_actor_policy(path, device=device)
    if method == "bc":
        return PredictPolicy(_load_imitation_actor_critic(path, device=device))
    if method in {"ppo", "gail"}:
        from stable_baselines3 import PPO
        return PredictPolicy(PPO.load(path, device=device))
    if method == "cql":
        from d3rlpy import load_learnable
        return PredictPolicy(load_learnable(str(path), device=device), sb3=False)
    if method == "dt":
        from thesis_experiments.training.decision_transformer import DecisionTransformerPolicy
        return DecisionTransformerPolicy(path, device=device)
    if method == "ghosting":
        return GhostingPolicy(path)
    raise ValueError(f"Unsupported method: {method}")
