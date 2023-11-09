import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax.config import config
from jrystal._src import energy
from jrystal._src.grid import get_ewald_vector_grid

config.update("jax_enable_x64", False)


class _Test_energy(parameterized.TestCase):

  def setUp(self):
    key = jax.random.PRNGKey(123)

    key, subkey = jax.random.split(key)
    self.ng = jax.random.uniform(subkey, [10, 10, 10]) + 0.1

    key, subkey = jax.random.split(key)
    self.gvec = jax.random.uniform(subkey, [10, 10, 10, 3]) + 0.1

    key, subkey = jax.random.split(key)
    self.positions = jax.random.normal(subkey, [10, 3])
    self.charges = jnp.arange(10)

    key, subkey = jax.random.split(key)
    self.kpts = jnp.ones([1, 3])
    self.coeff = jax.random.normal(subkey, [3, 1, 10, 10, 10, 10])
    self.vol = 1
    self.occupation = 1

  def test_hartree(self):
    eh = energy.hartree(self.ng, self.gvec, self.vol)
    np.testing.assert_almost_equal(eh, 2962.0739493371193, 5)

  def test_external(self):
    ee = energy.external(
      self.ng, self.positions, self.charges, self.gvec, self.vol
    )
    np.testing.assert_almost_equal(ee, -296984.46581626206, 5)

  def test_kinetic(self):
    ek = energy.kinetic(self.gvec, self.kpts, self.coeff, self.occupation)
    np.testing.assert_almost_equal(ek, 117683.2072496425, 5)

  def test_lda(self):
    ex = energy.xc_lda(self.ng, self.vol)
    np.testing.assert_almost_equal(ex, -0.38135464743073577, 5)

  def test_ewald(self):
    ew_grid = get_ewald_vector_grid(jnp.eye(3), 1e2)
    ew = energy.ewald_coulomb_repulsion(
      self.positions, self.charges, self.gvec, self.vol, 0.1, ew_grid
    )
    np.testing.assert_almost_equal(ew, 215765.68074046407, 5)


if __name__ == '__main__':
  absltest.main()
