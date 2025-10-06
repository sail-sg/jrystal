# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src.hessian import (
  complex_hessian, hvp, hessian_diag_pytree, hessian_diag_vector
)

jax.config.update("jax_enable_x64", True)


class _TestHessian(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(12341)

  def test_quadratic_form(self):
    dim = 10
    key1, key2 = jax.random.split(self.key, 2)
    a = jax.random.normal(key1, shape=(10, 10))
    b = jax.random.normal(key2, shape=(10, 10))

    A = a + b * 1.j
    H = A + A.conj().T

    def f(x):
      return 1 / 2 * x.conj().T @ H @ x

    x = jnp.ones(dim, dtype=A.dtype)
    hessian_matrix = complex_hessian(f, x)
    # print(hessian_matrix)
    # print(H)
    np.testing.assert_allclose(hessian_matrix, H)

  def test_hvp(self):
    dim = 10
    key1, key2 = jax.random.split(self.key, 2)
    a = jax.random.normal(key1, shape=(10, 10))
    b = jax.random.normal(key2, shape=(10, 10))
    v = jax.random.uniform(key2, shape=(10)) + 1.j

    A = a + b * 1.j
    H = A + A.conj().T

    def f(x):
      return jnp.real(1 / 2 * x.conj().T @ H @ x)

    x = jnp.ones(dim, dtype=A.dtype)

    hv = hvp(f, x, v)
    np.testing.assert_allclose(hv, H@v)

  def test_hessian_diag_pytree(self):
    dim = 10
    key1, key2 = jax.random.split(self.key, 2)
    a = jax.random.normal(key1, shape=(10, 10))
    b = jax.random.normal(key2, shape=(10, 10))

    A = a + b * 1.j
    H = A + A.conj().T

    def f(x):
      return jnp.real(1. / 2 * x.conj().T @ H @ x)

    x = jnp.ones(dim, dtype=A.dtype)
    hdiag = hessian_diag_pytree(f)(x)
    np.testing.assert_allclose(hdiag, jnp.diag(H))

  def test_hessian_diag_vector(self):
    dim = 10
    key1, key2 = jax.random.split(self.key, 2)
    a = jax.random.normal(key1, shape=(10, 10))
    b = jax.random.normal(key2, shape=(10, 10))

    A = a + b * 1.j
    H = A + A.conj().T

    def f(x):
      return jnp.real(1 / 2 * x.conj().T @ H @ x)

    x = jnp.ones(dim, dtype=A.dtype)
    hdiag = hessian_diag_vector(f)(x)
    np.testing.assert_allclose(hdiag, jnp.diag(H))


if __name__ == '__main__':
  absltest.main()
