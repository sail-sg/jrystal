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
import jrystal as jr
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src import braket

jax.config.update("jax_enable_x64", True)


class _TestEnergy(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    self.crystal = jr.Crystal.create_from_file("../geometry/diamond.xyz")
    self.num_bands = self.crystal.num_electron

    self.g_vecs = jr.grid.g_vectors(self.crystal.A, [7, 8, 9])
    self.freq_mask = jr.grid.cubic_mask([7, 8, 9])
    self.kpts = jr.grid.k_vectors(self.crystal.A, [1, 1, 1])
    self.params = jr.pw.param_init(
      self.key, self.num_bands, self.kpts.shape[0], self.freq_mask
    )
    self.coeff = jr.pw.coeff(self.params, self.freq_mask)

    self.occ = jr.occupation.gamma(
      self.kpts.shape[0], self.crystal.num_electron
    )
    self.wave_grid = jr.pw.wave_grid(self.coeff, self.crystal.vol)
    self.density_grid = jr.pw.density_grid(
      self.coeff, self.crystal.vol, self.occ
    )
    self.density_grid_reciprocal = jnp.fft.fftn(
      self.density_grid, axes=range(-3, 0)
    )

  def test_lda_energy(self):
    e1 = jr.energy.xc_lda(self.density_grid, self.crystal.vol, kohn_sham=True)
    e2 = jr.potential.xc_lda(self.density_grid, kohn_sham=True)

    e2 = jnp.sum(self.density_grid * e2)
    e2 = e2 * self.crystal.vol / jnp.prod(jnp.array([7, 8, 9]))
    print(e1)
    print(e2)
    np.testing.assert_allclose(e1, e2, atol=1e-7)

  def test_effective_potential(self):

    kohn_sham = False

    v_h, v_e, v_xc = jr.potential.effective(
      self.density_grid, self.crystal.positions, self.crystal.charges, self.g_vecs, self.crystal.vol, split=True, kohn_sham=kohn_sham
    )

    # e_h1 = jnp.sum(v_h * self.density_grid) * self.crystal.vol / jnp.prod(jnp.array([7, 8, 9]))
    # e_e1 = jnp.sum(v_e * self.density_grid) * self.crystal.vol / jnp.prod(jnp.array([7, 8, 9]))
    # e_xc1 = jnp.sum(v_xc * self.density_grid) * self.crystal.vol / jnp.prod(jnp.array([7, 8, 9]))

    e_h1 = braket.expectation(
      self.wave_grid, v_h, self.crystal.vol, diagonal=True, mode="real"
    )
    e_e1 = braket.expectation(
      self.wave_grid, v_e, self.crystal.vol, diagonal=True, mode="real"
    )
    e_xc1 = braket.expectation(
      self.wave_grid, v_xc, self.crystal.vol, diagonal=True, mode="real"
    )

    e_h1 = jnp.sum(e_h1 * self.occ).real
    e_e1 = jnp.sum(e_e1 * self.occ).real
    e_xc1 = jnp.sum(e_xc1 * self.occ).real

    e_h2 = jr.energy.hartree(
      self.density_grid_reciprocal,
      self.g_vecs,
      self.crystal.vol,
      kohn_sham=kohn_sham
    )
    e_e2 = jr.energy.external(
      self.density_grid_reciprocal,
      self.crystal.positions,
      self.crystal.charges,
      self.g_vecs,
      self.crystal.vol
    )
    e_xc2 = jr.energy.xc_lda(
      self.density_grid, self.crystal.vol, kohn_sham=kohn_sham
    )
    print(jnp.sum(e_h1).real)
    print(jnp.sum(e_h2).real)
    print(jnp.sum(e_e1).real)
    print(jnp.sum(e_e2).real)
    print(jnp.sum(e_xc1).real)
    print(jnp.sum(e_xc2).real)

    np.testing.assert_allclose(e_h1, e_h2, atol=1e-7)
    np.testing.assert_allclose(e_e1, e_e2, atol=1e-7)
    np.testing.assert_allclose(e_xc1, e_xc2, atol=1e-7)

  # def test_kinetic(self):
  #   e1 = jr.energy.kinetic(self.g_vecs, self.kpts, self.coeff, self.occ)
  #   e2 = jr.kinetic(self.g_vecs, self.kpts, self.coeff, self.occ)
  #   np.testing.assert_allclose(e1, e2, atol=1e-7)


if __name__ == '__main__':
  absltest.main()
