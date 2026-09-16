"""Coordinate-system tests for Metrica-to-GRF conversion."""

import numpy as np

from thesis_experiments.datasets.coordinates import (
    grf_to_metrica,
    metrica_to_grf,
    to_attacking_frame,
)


def test_metrica_pitch_landmarks_map_to_grf() -> None:
    assert np.allclose(metrica_to_grf(0.0, 0.0), [-1.0, -0.42])
    assert np.allclose(metrica_to_grf(0.5, 0.5), [0.0, 0.0])
    assert np.allclose(metrica_to_grf(1.0, 1.0), [1.0, 0.42])


def test_metrica_grf_coordinate_round_trip() -> None:
    source = np.asarray([0.237, 0.814], dtype=np.float32)
    assert np.allclose(grf_to_metrica(*metrica_to_grf(*source)), source)


def test_metrica_outside_pitch_is_clipped() -> None:
    assert np.allclose(metrica_to_grf(-0.1, 1.1), [-1.0, 0.42])


def test_negative_x_attacking_team_uses_rotated_view() -> None:
    np.testing.assert_allclose(
        to_attacking_frame((0.4, -0.2), attacks_positive_x=False),
        (-0.4, 0.2),
    )
