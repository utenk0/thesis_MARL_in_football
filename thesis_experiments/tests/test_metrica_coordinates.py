"""Coordinate-system tests for Metrica-to-GRF conversion."""

import numpy as np

from thesis_experiments.datasets.coordinates import (
    grf_to_metrica,
    metrica_to_grf,
)


def test_metrica_pitch_landmarks_map_to_grf() -> None:
    assert np.allclose(metrica_to_grf(0.0, 0.0), [-1.0, -0.42])
    assert np.allclose(metrica_to_grf(0.5, 0.5), [0.0, 0.0])
    assert np.allclose(metrica_to_grf(1.0, 1.0), [1.0, 0.42])


def test_metrica_grf_coordinate_round_trip() -> None:
    source = np.asarray([0.237, 0.814], dtype=np.float32)
    assert np.allclose(grf_to_metrica(*metrica_to_grf(*source)), source)
