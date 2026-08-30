"""Neural networks used by the focused CTDE pipeline."""

from thesis_experiments.policies.networks import CentralizedCritic, SharedActor

__all__ = ["SharedActor", "CentralizedCritic"]
