# Copyright 2026 Garena Online Private Limited
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
"""Tests for pw.py."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from . import energy
from .grid import cubic_mask, g_vectors, r_vectors
from .pw import (
  coeff,
  density_r,
  grad_density_grid,
  lapl_grid,
  nabla_density_grid,
  nabla_density_r,
  param_init,
  sigma_grid,
  tau_grid,
  wave_grid,
  wave_r,
)
from .utils import volume

jax.config.update("jax_enable_x64", True)


class _TestModules(unittest.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    self.cell_vectors = jnp.array(
      [[3.0, 0.0, 0.0], [0.0, 3.2, 0.0], [0.0, 0.0, 3.4]],
      dtype=jnp.float64,
    )
    self.vol = volume(self.cell_vectors)
    self.grid_size = (5, 6, 7)
    self.freq_mask = cubic_mask(self.grid_size)
    self.num_bands = 3
    self.num_kpts = 1

    pw_param = param_init(
      self.key,
      num_bands=self.num_bands,
      num_kpts=self.num_kpts,
      freq_mask=self.freq_mask,
      spin_restricted=True,
    )
    self.coeff = coeff(pw_param, self.freq_mask)
    self.g_vecs = g_vectors(self.cell_vectors, self.grid_size)
    self.occupation = jnp.ones(
      (self.coeff.shape[0], self.coeff.shape[1], self.coeff.shape[2]),
      dtype=jnp.float64,
    )
    self.kpts = jnp.zeros((self.num_kpts, 3), dtype=jnp.float64)

  def test_wave_grid_shape(self):
    w = wave_grid(self.coeff, self.vol)
    self.assertEqual(w.shape, self.coeff.shape)

  def test_wave_r_shape(self):
    r = jnp.array([0.1, 0.2, -0.3], dtype=jnp.float64)
    psi = wave_r(r, self.coeff, self.cell_vectors, self.g_vecs)
    self.assertEqual(psi.shape, (1, 1, self.num_bands))

  def test_density_r_shape(self):
    r = jnp.array([0.11, -0.07, 0.22], dtype=jnp.float64)
    den_state = density_r(r, self.coeff, self.cell_vectors, self.g_vecs)
    den_total = density_r(
      r,
      self.coeff,
      self.cell_vectors,
      self.g_vecs,
      self.occupation,
    )
    self.assertEqual(den_state.shape, (1, 1, self.num_bands))
    self.assertEqual(den_total.shape, ())

  def test_nabla_density_grid_matches_autodiff_total_density(self):
    r = jnp.array([0.2, 0.05, -0.1], dtype=jnp.float64)

    grad_analytic = nabla_density_grid(
      r,
      self.coeff,
      self.cell_vectors,
      self.g_vecs,
      self.occupation,
    )

    def total_density(r):
      return density_r(
        r,
        self.coeff,
        self.cell_vectors,
        self.g_vecs,
        self.occupation,
      )

    grad_autodiff = jax.grad(total_density)(r)
    np.testing.assert_allclose(
      grad_analytic, grad_autodiff, atol=1e-8, rtol=1e-8
    )

  def test_nabla_density_r_matches_nabla_density_grid(self):
    r = jnp.array([-0.3, 0.12, 0.07], dtype=jnp.float64)
    grad_r = nabla_density_r(
      r,
      self.coeff,
      self.cell_vectors,
      self.g_vecs,
      self.occupation,
    )
    grad_grid = nabla_density_grid(
      r,
      self.coeff,
      self.cell_vectors,
      self.g_vecs,
      self.occupation,
    )
    np.testing.assert_allclose(grad_r, grad_grid, atol=1e-10, rtol=1e-10)

  def test_grad_density_grid_shape(self):
    grad = grad_density_grid(
      self.coeff, self.vol, self.g_vecs, self.occupation
    )
    self.assertEqual(grad.shape, (1,) + self.grid_size + (3,))

  def test_sigma_grid_shape(self):
    sigma = sigma_grid(
      self.coeff, self.vol, self.g_vecs, self.occupation
    )
    self.assertEqual(sigma.shape, (1,) + self.grid_size)

  def test_tau_grid_shape(self):
    tau = tau_grid(
      self.coeff, self.vol, self.g_vecs, self.kpts, self.occupation
    )
    self.assertEqual(tau.shape, (1,) + self.grid_size)

  def test_lapl_grid_shape(self):
    lapl = lapl_grid(
      self.coeff, self.vol, self.g_vecs, self.occupation
    )
    self.assertEqual(lapl.shape, (1,) + self.grid_size)

  def test_grad_density_grid_matches_single_point(self):
    """Full-grid gradient at a grid point should match single-point formula."""
    r_grid = r_vectors(self.cell_vectors, self.grid_size)
    idx = (1, 2, 1)
    r = r_grid[idx]

    grad_full = grad_density_grid(
      self.coeff, self.vol, self.g_vecs, self.occupation
    )
    grad_at_point = grad_full[0, idx[0], idx[1], idx[2]]

    grad_single = nabla_density_grid(
      r, self.coeff, self.cell_vectors, self.g_vecs, self.occupation
    )
    np.testing.assert_allclose(grad_at_point, grad_single, atol=1e-10)

  def test_sigma_equals_grad_squared(self):
    """sigma should equal the squared norm of the density gradient."""
    grad = grad_density_grid(
      self.coeff, self.vol, self.g_vecs, self.occupation
    )
    sigma_expected = jnp.sum(grad ** 2, axis=-1)
    sigma_actual = sigma_grid(
      self.coeff, self.vol, self.g_vecs, self.occupation
    )
    np.testing.assert_allclose(sigma_actual, sigma_expected, atol=1e-12)

  def test_tau_integral_equals_kinetic_energy(self):
    """Integral of tau over the cell should equal kinetic energy."""
    tau = tau_grid(
      self.coeff, self.vol, self.g_vecs, self.kpts, self.occupation
    )
    num_grid = np.prod(self.grid_size)
    tau_integral = jnp.sum(tau) * self.vol / num_grid

    e_kin = energy.kinetic(
      self.coeff, self.g_vecs, self.kpts, None, self.occupation
    )
    np.testing.assert_allclose(tau_integral, e_kin, atol=1e-10)

  def test_tau_non_negative(self):
    """Kinetic energy density should be non-negative everywhere."""
    tau = tau_grid(
      self.coeff, self.vol, self.g_vecs, self.kpts, self.occupation
    )
    self.assertTrue(jnp.all(tau >= -1e-15))

  def test_lapl_integrates_to_zero(self):
    """Integral of Laplacian over the periodic cell should vanish."""
    lapl = lapl_grid(
      self.coeff, self.vol, self.g_vecs, self.occupation
    )
    num_grid = np.prod(self.grid_size)
    lapl_integral = jnp.sum(lapl) * self.vol / num_grid
    np.testing.assert_allclose(lapl_integral, 0.0, atol=1e-12)
