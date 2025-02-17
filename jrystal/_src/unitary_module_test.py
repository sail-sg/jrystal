import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from .unitary_module import UnitaryMatrix

from jrystal._src.grid import cubic_mask, k_vectors

jax.config.update("jax_enable_x64", True)


class _TestModules(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)

    self.ni = 4
    self.grid_sizes = [7, 8, 9]
    self.k_grid_sizes = [2, 2, 2]
    self.mask = cubic_mask(self.grid_sizes)
    self.ng = int(jnp.sum(self.mask).item())
    self.a = jnp.eye(3)
    self.k_grid = k_vectors(np.eye(3), self.k_grid_sizes)
    self.nk = self.k_grid.shape[0]

  def test_qr_shape(self):
    shape = [2, 1, self.ng, 4]
    qr = UnitaryMatrix(shape, True)
    params = qr.init(self.key)
    np.testing.assert_array_equal(params['w_re'].shape, shape)
    x = qr(params)
    np.testing.assert_array_equal(x.shape, shape)


if __name__ == '__main__':
  absltest.main()
