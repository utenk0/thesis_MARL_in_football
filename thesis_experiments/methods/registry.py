"""Registry for the unified football method interface."""

from __future__ import annotations

from importlib import import_module

from thesis_experiments.methods.base import FootballMethod, MethodConfig


METHODS: dict[str, str] = {
    "bc": "BehavioralCloningMethod",
    "gail": "GAILMethod",
    "ppo": "PPOMethod",
    "cql": "CQLMethod",
    "dt": "DecisionTransformerMethod",
    "ghosting": "GhostingMethod",
    "ctde": "CTDEMethod",
}


def available_methods() -> list[str]:
    return sorted(METHODS)


def build_method(config: MethodConfig) -> FootballMethod:
    try:
        class_name = METHODS[config.method]
    except KeyError as exc:
        raise ValueError(f"Unknown method '{config.method}'. Choose from: {', '.join(available_methods())}.") from exc
    method_class = getattr(import_module("thesis_experiments.methods.implementations"), class_name)
    return method_class(config)
