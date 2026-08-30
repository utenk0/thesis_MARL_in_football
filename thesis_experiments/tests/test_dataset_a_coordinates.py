"""Coordinate tests for Dataset A's centred metre pitch."""

import numpy as np

from thesis_experiments.datasets.coordinates import dataset_a_to_grf, grf_to_dataset_a


def test_dataset_a_pitch_landmarks_map_to_grf() -> None:
    np.testing.assert_allclose(dataset_a_to_grf(-52.5, -34), [-1, -0.42])
    np.testing.assert_allclose(dataset_a_to_grf(0, 0), [0, 0])
    np.testing.assert_allclose(dataset_a_to_grf(52.5, 34), [1, 0.42])


def test_dataset_a_grf_coordinate_round_trip() -> None:
    source = np.asarray([17.25, -12.5], dtype=np.float32)
    np.testing.assert_allclose(grf_to_dataset_a(*dataset_a_to_grf(*source)), source, atol=1e-5)
