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

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from .unitary_module import unitary_matrix, unitary_matrix_param_init

jax.config.update("jax_enable_x64", True)


class _TestModules(unittest.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    self.ng = 32

  def test_qr_shape(self):
    shape = [2, 1, self.ng, 4]
    params = unitary_matrix_param_init(self.key, shape, complex=True)
    np.testing.assert_array_equal(params['w_re'].shape, shape)
    x = unitary_matrix(params, complex=True)
    np.testing.assert_array_equal(x.shape, shape)
    gram = jnp.einsum("...gi,...gj->...ij", jnp.conj(x), x)
    np.testing.assert_allclose(
      gram,
      jnp.broadcast_to(jnp.eye(shape[-1]), gram.shape),
      atol=1e-10,
      rtol=1e-10,
    )
