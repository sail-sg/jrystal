"""Tests of occupation module.  """

import jax.numpy as jnp
from absl.testing import absltest, parameterized
from jrystal._src.occupation import fermi_dirac
import numpy as np


class _TestOccupation(parameterized.TestCase):

  def setUp(self):
    return super().setUp()

  def test_fermi_direc(self):
    num = 100
    eigenvalues = np.linspace(-2, 2, num)
    fermi_level = 0
    w = 0.001
    occ = fermi_dirac(eigenvalues, fermi_level, w)
    np.testing.assert_almost_equal(num / 2, jnp.sum(occ))


if __name__ == "__main__":
  absltest.main()
