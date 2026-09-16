"""Provider coordinate conversions to and from the GRF pitch system."""

from __future__ import annotations

from typing import Any

import numpy as np

GRF_X_BOUNDS = (-1.0, 1.0)
GRF_Y_BOUNDS = (-0.42, 0.42)


def dataset_a_to_grf(x_value: Any, y_value: Any) -> np.ndarray:
    """Map Dataset A's centred 105 x 68 metre pitch to GRF coordinates."""
    x, y = _numbers(x_value, y_value)
    if not np.isfinite([x, y]).all():
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    return np.asarray([np.clip(x / 52.5, -1, 1), np.clip(0.42 * y / 34, -0.42, 0.42)], dtype=np.float32)


def grf_to_dataset_a(x_value: Any, y_value: Any) -> np.ndarray:
    """Map GRF coordinates to Dataset A's centred pitch metres."""
    x, y = _numbers(x_value, y_value)
    if not np.isfinite([x, y]).all():
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    return np.asarray([52.5 * x, 34 * y / 0.42], dtype=np.float32)


def metrica_to_grf(x_value: Any, y_value: Any) -> np.ndarray:
    """Map Metrica's top-left [0,1] pitch to GRF coordinates."""
    x, y = _numbers(x_value, y_value)
    if not np.isfinite([x, y]).all():
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    return np.asarray([np.clip(2 * x - 1, -1, 1), np.clip(0.42 * (2 * y - 1), -0.42, 0.42)], dtype=np.float32)


def grf_to_metrica(x_value: Any, y_value: Any) -> np.ndarray:
    """Map GRF coordinates to Metrica's top-left [0,1] pitch."""
    x, y = _numbers(x_value, y_value)
    if not np.isfinite([x, y]).all():
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    return np.asarray([(x + 1) * 0.5, (y / 0.42 + 1) * 0.5], dtype=np.float32)


def _numbers(x_value: Any, y_value: Any) -> tuple[float, float]:
    try:
        return float(x_value), float(y_value)
    except (TypeError, ValueError):
        return float("nan"), float("nan")


def provider_to_grf(source: str, x_value: Any, y_value: Any) -> np.ndarray:
    """Convert a provider-native tracking position to shared GRF world space."""
    if source == "metrica":
        return metrica_to_grf(x_value, y_value)
    if source == "dataset_a":
        return dataset_a_to_grf(x_value, y_value)
    raise ValueError(f"Unsupported coordinate source: {source}.")


def provider_velocity_to_grf(source: str, x_value: Any, y_value: Any) -> np.ndarray:
    """Scale a provider-native velocity without applying position offsets."""
    x, y = _numbers(x_value, y_value)
    if not np.isfinite([x, y]).all():
        return np.asarray([np.nan, np.nan], dtype=np.float32)
    if source == "metrica":
        return np.asarray([2.0 * x, 0.84 * y], dtype=np.float32)
    if source == "dataset_a":
        return np.asarray([x / 52.5, 0.42 * y / 34.0], dtype=np.float32)
    raise ValueError(f"Unsupported coordinate source: {source}.")


def to_attacking_frame(position: Any, *, attacks_positive_x: bool) -> np.ndarray:
    """Express a GRF world position in a team's positive-x attacking frame.

    GRF exposes right-team controls in a 180-degree rotated egocentric frame,
    so a team attacking toward negative world x uses ``(-x, -y)``.
    """
    values = np.asarray(position, dtype=np.float32)
    return values.copy() if attacks_positive_x else -values
