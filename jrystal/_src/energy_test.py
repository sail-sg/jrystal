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

from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jrystal as jr
from jrystal._src import braket

jax.config.update("jax_enable_x64", True)


class _TestEnergy(unittest.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    geometry_path = Path(__file__).resolve().parents[2] / "geometry" / "diamond.xyz"
    self.crystal = jr.Crystal.create_from_file(str(geometry_path))
    self.num_bands = self.crystal.num_electron

    self.g_vecs = jr.grid.g_vectors(self.crystal.A, [7, 8, 9])
    self.freq_mask = jr.grid.cubic_mask([7, 8, 9])
    self.kpts, self.kpts_weights = jr.grid.k_vectors(
      self.crystal.A,
      [1, 1, 1],
      scaled_positions=self.crystal.scaled_positions,
      charges=self.crystal.charges,
    )
    self.params = jr.pw.param_init(
      self.key, self.num_bands, self.kpts.shape[0], self.freq_mask
    )
    self.coeff = jr.pw.coeff(self.params, self.freq_mask)

    self.occ = jnp.zeros(
      (1, self.kpts.shape[0], self.num_bands),
      dtype=jnp.float64,
    )
    self.occ = self.occ.at[:, :, :self.crystal.num_electron // 2].set(2.0)
    self.wave_grid = jr.pw.wave_grid(self.coeff, self.crystal.vol)
    self.density_grid = jr.pw.density_grid(
      self.coeff, self.crystal.vol, self.occ
    )
    self.density_grid_reciprocal = jnp.fft.fftn(
      self.density_grid, axes=range(-3, 0)
    )

  def test_lda_energy(self):
    e_xc = jr.energy.xc_energy(
      self.density_grid,
      self.g_vecs,
      self.crystal.vol,
      xc_type="lda_x",
      kohn_sham=False,
    )
    _, _, _, e_xc_split = jr.energy.total_energy(
      self.coeff,
      self.crystal.positions,
      self.crystal.charges,
      self.g_vecs,
      self.kpts,
      self.crystal.vol,
      occupation=self.occ,
      xc="lda_x",
      split=True,
    )
    np.testing.assert_allclose(e_xc, e_xc_split, atol=1e-7)

  def test_effective_potential(self):

    kohn_sham = False

    v_h, v_e, v_xc = jr.potential.effective(
      self.density_grid, self.crystal.positions, self.crystal.charges,
      self.g_vecs, self.crystal.vol, split=True, kohn_sham=kohn_sham
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
    e_h1 = jnp.sum(e_h1 * self.occ).real
    e_e1 = jnp.sum(e_e1 * self.occ).real

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

    np.testing.assert_allclose(e_h1, e_h2, atol=1e-7)
    np.testing.assert_allclose(e_e1, e_e2, atol=1e-7)
    self.assertTrue(jnp.isfinite(v_xc).all())

  # def test_kinetic(self):
  #   e1 = jr.energy.kinetic(self.g_vecs, self.kpts, self.coeff, self.occ)
  #   e2 = jr.kinetic(self.g_vecs, self.kpts, self.coeff, self.occ)
  #   np.testing.assert_allclose(e1, e2, atol=1e-7)
