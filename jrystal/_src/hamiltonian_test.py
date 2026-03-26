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
from jrystal._src.hamiltonian import (
  _hamiltonian_matrix,
  hamiltonian_matrix,
  hamiltonian_matrix_trace,
)

jax.config.update("jax_enable_x64", True)


class _TestHamiltonian(unittest.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    diamond_file_path = Path(__file__).resolve().parents[2] / "geometry" / "diamond.xyz"
    self.crystal = jr.Crystal.create_from_file(str(diamond_file_path))
    self.num_bands = self.crystal.num_electron
    self.key = jax.random.PRNGKey(123)
    self.kpts, self.kpts_weights = jr.grid.k_vectors(
      self.crystal.A,
      [2, 2, 1],
      scaled_positions=self.crystal.scaled_positions,
      charges=self.crystal.charges,
    )
    self.g_vecs = jr.grid.g_vectors(self.crystal.A, [7, 8, 9])
    self.freq_mask = jr.grid.cubic_mask([7, 8, 9])

    self.params = jr.pw.param_init(
      self.key, self.num_bands, self.kpts.shape[0], self.freq_mask
    )
    self.coeff = jr.pw.coeff(self.params, self.freq_mask)

    self.occ = jnp.zeros(
      (1, self.kpts.shape[0], self.num_bands),
      dtype=jnp.float64,
    )
    self.occ = self.occ.at[:, :, :self.crystal.num_electron // 2].set(2.0)
    self.density_grid = jr.pw.density_grid(
      self.coeff, self.crystal.vol, self.occ
    )

  def test_hamiltonian_matrix(self):
    h1 = _hamiltonian_matrix(
      self.coeff,
      self.crystal.positions,
      self.crystal.charges,
      self.density_grid,
      self.g_vecs,
      self.kpts,
      self.crystal.vol,
      kohn_sham=True
    )

    h2 = hamiltonian_matrix(
      self.coeff,
      self.crystal.positions,
      self.crystal.charges,
      self.density_grid,
      self.g_vecs,
      self.kpts,
      self.crystal.vol,
      kohn_sham=True
    )

    np.testing.assert_allclose(h1, h2, atol=1e-7)

  def test_hamiltonian_matrix_trace(self):
    e1 = hamiltonian_matrix_trace(
      self.coeff,
      self.crystal.positions,
      self.crystal.charges,
      self.density_grid,
      self.crystal.vol,
      self.g_vecs,
      self.kpts,
      kohn_sham=True
    ).real
    h = hamiltonian_matrix(
      self.coeff,
      self.crystal.positions,
      self.crystal.charges,
      self.density_grid,
      self.g_vecs,
      self.kpts,
      self.crystal.vol,
      kohn_sham=True
    )
    e2 = jnp.trace(h, axis1=-2, axis2=-1).sum().real
    np.testing.assert_allclose(e1, e2, atol=1e-7)
