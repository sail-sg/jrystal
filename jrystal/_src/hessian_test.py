import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src.hessian import complex_hessian


jax.config.update("jax_enable_x64", True)


class _TestHessian(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)

  def test_quadratic_form(self):
    dim = 10
    key1, key2 = jax.random.split(self.key, 2)
    a = jax.random.normal(key1, shape=(10, 10))
    b = jax.random.normal(key2, shape=(10, 10))

    A = a + b*1.j
    H = A+A.conj().T

    def f(x):
      return 1/2 * x.conj().T @ H @ x

    x = jnp.ones(dim, dtype=A.dtype)
    hessian_matrix = complex_hessian(f, x)
    # print(hessian_matrix)
    # print(H)
    np.testing.assert_allclose(hessian_matrix, H)


if __name__ == '__main__':
  absltest.main()
