import unittest
import numpy as np

from gravity import calc_distance_matrix, calc_diff_matrix, calc_direction_matrix


def get_pos():
    pos = np.array([
        [0, 0],
        [3, 4],
        [4, 3],
    ])
    return pos


class TestGravity(unittest.TestCase):
    tol = 1e-10

    def test_calc_distance_and_direction_matrix(self):
        pos = get_pos()

        diff_matrix = calc_diff_matrix(pos)
        distance_matrix = calc_distance_matrix(diff_matrix)

        expected = np.array([
            [0, 5, 5],
            [5, 0, np.sqrt(2)],
            [5, np.sqrt(2), 0],
        ])
        self.assertTrue(np.sum(distance_matrix - expected, axis=None) < self.tol)

        direction_matrix = calc_direction_matrix(diff_matrix, distance_matrix)
        tmp = calc_distance_matrix(direction_matrix)

        self.assertTrue(sum(tmp[np.where(~np.eye(tmp.shape[0], dtype=bool))] - 1) < self.tol)
