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
