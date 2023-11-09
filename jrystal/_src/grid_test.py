import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src.grid import get_ewald_vector_grid, grid_1d


class _Test_grid(parameterized.TestCase):

  def setUp(self):

    def get_ewald_vector_grid(a, ew_cut=1e4):
      n = jnp.ceil(ew_cut / jnp.linalg.norm(jnp.sum(a, axis=0))**2)
      n = jnp.arange(n) - n // 2
      n = n[:, None, None, None] * a[None, None, None, 0] + \
        n[None, :, None, None] * a[None, None, None, 1] + \
        n[None, None, :, None] * a[None, None, None, 2]
      return jnp.reshape(n, [-1, 3])

    self.get_ewald_vector_grid = get_ewald_vector_grid

  def test_grid(self):
    key = jax.random.PRNGKey(123)
    a = jax.random.uniform(key, [3, 3])
    ec = 1e2
    l1 = self.get_ewald_vector_grid(a, ec)
    l2 = get_ewald_vector_grid(a, ec)

    np.testing.assert_allclose(l1.shape, l2.shape, 1e-8)

  def test_fftfreq(self):
    np.testing.assert_array_equal(
      grid_1d(9, normalize=True), jnp.fft.fftfreq(9, 1)
    )
    np.testing.assert_array_equal(
      grid_1d(9, normalize=False), jnp.fft.fftfreq(9, 1 / 9)
    )
    np.testing.assert_array_equal(
      grid_1d(10, normalize=True), jnp.fft.fftfreq(10, 1)
    )
    np.testing.assert_array_equal(
      grid_1d(10, normalize=False), jnp.fft.fftfreq(10, 1 / 10)
    )


if __name__ == '__main__':
  absltest.main()
