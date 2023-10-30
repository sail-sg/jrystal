import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src.grid import _get_ewald_lattice


class _Test_grid(parameterized.TestCase):
  
  def setUp(self):
    def get_ewald_lattice(a, ew_cut=1e4):
      n = jnp.ceil(ew_cut / jnp.linalg.norm(jnp.sum(a, axis=0))**3)
      n = jnp.arange(n) - n // 2
      n = n[:, None, None, None] * a[None, None, None, 0] + \
        n[None, :, None, None] * a[None, None, None, 1] + \
        n[None, None, :, None] * a[None, None, None, 2]
      return jnp.reshape(n, [-1, 3])
    self.get_ewald_lattice = get_ewald_lattice
  
  def test_grid(self):
    key = jax.random.PRNGKey(123)
    a = jax.random.uniform(key, [3, 3])
    ec = 2e-3
    l1 = self.get_ewald_lattice(a, ec)
    l2 = _get_ewald_lattice(a, ec)
    
    np.testing.assert_allclose(l1, l2, 1e-8)


if __name__ == '__main__':
  absltest.main()
