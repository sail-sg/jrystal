"""Test for crystal.py"""

from absl.testing import absltest, parameterized
import numpy as np
from .crystal import Crystal


class _TestModules(parameterized.TestCase):

  def setUp(self):
    self.file_path = "../geometry/diamond.xyz"
    self.positions = np.array(
      [
        [-0.84251071, -0.84251071, -0.84251071],
        [0.84251071, 0.84251071, 0.84251071]
      ]
    )
    self.charges = np.array([6, 6])
    self.cell_vectors = np.array(
      [
        [0., 3.37004284, 3.37004284], [3.37004284, 0., 3.37004284],
        [3.37004284, 3.37004284, 0.]
      ]
    )
    self.vol = 76.5484253352856
    self.scaled_positions = np.array(
      [[-0.125, -0.125, -0.125], [0.125, 0.125, 0.125]]
    )
    self.reciprocal_vectors = np.array(
      [
        [-0.93221149, 0.93221149, 0.93221149],
        [0.93221149, -0.93221149, 0.93221149],
        [0.93221149, 0.93221149, -0.93221149]
      ]
    )

  def test_create_from_file(self):
    crystal = Crystal.create_from_file(self.file_path)
    np.testing.assert_almost_equal(crystal.positions, self.positions)
    np.testing.assert_almost_equal(crystal.charges, self.charges)
    np.testing.assert_almost_equal(crystal.cell_vectors, self.cell_vectors)
    self.assertEqual(crystal.vol, self.vol)
    np.testing.assert_almost_equal(
      crystal.scaled_positions, self.scaled_positions
    )
    np.testing.assert_almost_equal(
      crystal.reciprocal_vectors, self.reciprocal_vectors
    )


if __name__ == "__main__":
  absltest.main()
