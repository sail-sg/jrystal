import jax.numpy as jnp
from absl.testing import absltest, parameterized
import numpy as np
from jrystal._src.utils import vmapstack, vmapstack_reverse
from jax.config import config

config.update("jax_enable_x64", True)


class _Test_utils(parameterized.TestCase):

  def setUp(self):
    self.f = jnp.mean
    self.g = jnp.sin

  @parameterized.parameters(([3, 4, 5, 6, 7],), ([2, 3, 4, 5],))
  def test_vmapstack_mean(self, shape):
    _vmap_f = vmapstack(times=3)(self.f)
    x = jnp.ones(shape=shape)
    np.testing.assert_array_equal(_vmap_f(x).shape, shape[:3])

  @parameterized.parameters(([3, 4, 5, 6, 7],), ([2, 3, 4, 5],))
  def test_vmapstack_sin(self, shape):
    _vmap_f = vmapstack(times=3)(self.g)
    x = jnp.ones(shape=shape)
    np.testing.assert_array_equal(_vmap_f(x).shape, shape)

  @parameterized.parameters(
    {
      'shape': [3, 4, 5, 6, 7],
      'results': [3, 4, 5]
    }, {
      'shape': [2, 3, 4, 5],
      'results': [2, 3, 4]
    }
  )
  def test_vmapstack_reverse_mean(self, shape, results):
    _vmap_f = vmapstack_reverse(times=3)(self.f)
    x = jnp.ones(shape=shape)
    np.testing.assert_array_equal(_vmap_f(x).shape, results)

  @parameterized.parameters(
    {
      'shape': [3, 4, 5, 6, 7],
      'results': [5, 6, 7, 3, 4]
    }, {
      'shape': [2, 3, 4, 5],
      'results': [4, 5, 2, 3]
    }
  )
  def test_vmapstack_reverse_sin(self, shape, results):
    _vmap_f = vmapstack_reverse(times=2)(self.g)
    x = jnp.ones(shape=shape)
    np.testing.assert_array_equal(_vmap_f(x).shape, results)


if __name__ == "__main__":
  absltest.main()
