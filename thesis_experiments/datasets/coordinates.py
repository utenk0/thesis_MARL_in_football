"""Provider coordinate conversions to and from the GRF pitch system."""

from __future__ import annotations

from typing import Any

import numpy as np


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
