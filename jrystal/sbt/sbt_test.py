"""Test cases for spherical Bessel transform implementations.

This module tests the consistency between different SBT implementations:
- batch_sbt from sbt.py (uses pysbt library)
- sbt from sbt_numerical.py (pure numerical implementation)
"""

import numpy as np
from absl.testing import absltest

from jrystal import Crystal, get_pkg_path
from jrystal.pseudopotential.dataclass import (
  NormConservingPseudopotential as NCPP
)

# Import the functions to test
from .sbt import batch_sbt
from .sbt_numerical import sbt as numerical_sbt
from scipy.interpolate import CubicSpline


class TestSBTConsistency(absltest.TestCase):
  """Test consistency between different SBT implementations."""

  def setUp(self):
    """Set up test data for each test method."""
    self.crystal = Crystal.create_from_symbols(
      symbols="Si",
      positions=[[0, 0, 0]],
      cell_vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    self.pseudo_file = get_pkg_path() + "/pseudopotential/normconserving/"
    # Create logarithmic r grid (as required by pysbt)
    self.pseudo = NCPP.create(self.crystal, self.pseudo_file)
    self.beta = np.array(self.pseudo.nonlocal_beta_grid[0], dtype=np.float64)
    self.r_ab = np.array(self.pseudo.r_ab[0], dtype=np.float64)
    self.r_grid = np.array(self.pseudo.r_grid[0], dtype=np.float64)
    self.angmom = np.array(
      self.pseudo.nonlocal_angular_momentum[0], dtype=np.int64
    )

  def test_sbt(self):
    gg1, beta_g1 = batch_sbt(
      self.r_grid,
      self.beta,
      l=self.angmom,
      kmax=100,
      norm=False
    )

    xx = np.linspace(0, 100, 2000, dtype=np.float64)
    beta_g1 = CubicSpline(gg1, beta_g1, axis=-1)(xx)

    # Get result from numerical_sbt
    gg2, beta_g2 = numerical_sbt(
      self.r_grid,
      self.beta,
      l=self.angmom,
      kmax=100,
      delta_r=self.r_ab
    )

    beta_g2 = CubicSpline(gg2, beta_g2, axis=-1)(xx)
    # Compare results (allowing for some numerical differences)
    np.testing.assert_allclose(
      beta_g1,
      beta_g2,
      rtol=0,
      atol=5e-3,
      err_msg="SBT results should match between implementations"
    )

  def test_sbt_single_l(self):
    angmom = 1
    gg1, beta_g1 = batch_sbt(
      self.r_grid,
      self.beta,
      l=angmom,
      kmax=100,
      norm=False
    )

    xx = np.linspace(0, 100, 2000, dtype=np.float64)
    beta_g1 = CubicSpline(gg1, beta_g1, axis=-1)(xx)

    # Get result from numerical_sbt
    gg2, beta_g2 = numerical_sbt(
      self.r_grid,
      self.beta,
      l=angmom,
      kmax=100,
      delta_r=self.r_ab
    )

    beta_g2 = CubicSpline(gg2, beta_g2, axis=-1)(xx)

    # Compare results (allowing for some numerical differences)
    np.testing.assert_allclose(
      beta_g1,
      beta_g2,
      rtol=0,
      atol=5e-5,
      err_msg="SBT results should match between implementations"
    )


if __name__ == "__main__":
  absltest.main()
