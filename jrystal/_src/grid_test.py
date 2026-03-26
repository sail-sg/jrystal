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
"""Tests for grid.py."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from .grid import (
  cubic_mask,
  estimate_max_cutoff_energy,
  g2cell_vectors,
  g2r_vector_grid,
  g_vectors,
  grid_vector_radius,
  k_vectors,
  proper_grid_size,
  r2cell_vectors,
  r2g_vector_grid,
  r_vectors,
  spherical_mask,
  translation_vectors,
)

jax.config.update("jax_enable_x64", True)


class _TestModules(unittest.TestCase):

  def setUp(self):
    self.cell_vectors = jnp.array(
      [
        [3.1, 0.0, 0.0],
        [0.2, 2.9, 0.0],
        [0.1, 0.3, 3.3],
      ],
      dtype=jnp.float64,
    )
    self.grid_sizes = jnp.array([6, 8, 10], dtype=jnp.int32)
    self.simple_cell = jnp.eye(3, dtype=jnp.float64)
    self.scaled_positions = jnp.array([[0.25, 0.25, 0.25]])
    self.charges = jnp.array([6], dtype=jnp.int32)

  def test_g_vectors_and_r_vectors_shapes(self):
    g = g_vectors(self.cell_vectors, self.grid_sizes)
    r = r_vectors(self.cell_vectors, self.grid_sizes)
    self.assertEqual(g.shape, (6, 8, 10, 3))
    self.assertEqual(r.shape, (6, 8, 10, 3))
    np.testing.assert_allclose(g[0, 0, 0], jnp.zeros(3), atol=1e-12)
    np.testing.assert_allclose(r[0, 0, 0], jnp.zeros(3), atol=1e-12)

  def test_k_vectors_shape_and_gamma_for_111(self):
    k, w = k_vectors(
      self.simple_cell, [1, 1, 1],
      scaled_positions=self.scaled_positions,
      charges=self.charges,
    )
    self.assertEqual(k.shape, (1, 3))
    np.testing.assert_allclose(k[0], jnp.zeros(3), atol=1e-12)

  def test_proper_grid_size_scalar_and_list(self):
    scalar = proper_grid_size(12)
    vec = proper_grid_size([12, 12, 12])
    np.testing.assert_array_equal(scalar, vec)
    self.assertEqual(scalar.shape, (3,))
    self.assertTrue(jnp.all(scalar > 0))

  def test_proper_grid_size_validation(self):
    with self.assertRaisesRegex(ValueError, "must be positive"):
      proper_grid_size([8, 0, 8])
    with self.assertRaisesRegex(TypeError, "mesh should contain"):
      proper_grid_size(["a", 8, 8])

  def test_translation_vectors_and_validation(self):
    t = translation_vectors(self.cell_vectors, cutoff=10.0)
    self.assertEqual(t.shape[1], 3)
    self.assertGreater(t.shape[0], 0)
    with self.assertRaisesRegex(TypeError, "scalar float"):
      translation_vectors(
        self.cell_vectors, cutoff=jnp.array([10.0, 10.0, 10.0])
      )
    with self.assertRaisesRegex(ValueError, "must be positive"):
      translation_vectors(self.cell_vectors, cutoff=0.0)

  def test_spherical_and_cubic_masks(self):
    sph = spherical_mask(self.simple_cell, [5, 5, 5], cutoff_energy=0.0)
    cub = cubic_mask([7, 7, 7])
    self.assertEqual(sph.shape, (5, 5, 5))
    self.assertEqual(cub.shape, (7, 7, 7))
    self.assertEqual(int(jnp.sum(jnp.array(sph))), 1)
    self.assertEqual(int(jnp.sum(jnp.array(cub))), 64)

  def test_estimate_max_cutoff_energy(self):
    mask = cubic_mask([7, 7, 7])
    emax = estimate_max_cutoff_energy(self.simple_cell, mask)
    self.assertGreater(emax, 0.0)

  def test_grid_vector_radius(self):
    g = g_vectors(self.cell_vectors, self.grid_sizes)
    r1 = grid_vector_radius(g)
    r2 = jnp.linalg.norm(jnp.array(g), axis=-1)
    np.testing.assert_allclose(r1, r2, atol=1e-6, rtol=1e-6)

  def test_g2r_and_r2g_conversion(self):
    g = g_vectors(self.cell_vectors, self.grid_sizes)
    r = g2r_vector_grid(g, self.cell_vectors)
    g_back = r2g_vector_grid(r, self.cell_vectors)
    np.testing.assert_allclose(
      r, r_vectors(self.cell_vectors, self.grid_sizes), atol=1e-8
    )
    np.testing.assert_allclose(g_back, g, atol=1e-10)

  def test_infer_cell_vectors_from_grids(self):
    g = g_vectors(self.cell_vectors, self.grid_sizes)
    r = r_vectors(self.cell_vectors, self.grid_sizes)
    cell_from_g = g2cell_vectors(g)
    cell_from_r = r2cell_vectors(r)
    r_reconstructed = r_vectors(cell_from_r, self.grid_sizes)

    # r2cell_vectors should reconstruct an equivalent real-space grid.
    np.testing.assert_allclose(r_reconstructed, r, atol=1e-5, rtol=1e-5)

    # g2cell_vectors should at least return a valid finite non-singular cell.
    self.assertEqual(cell_from_g.shape, (3, 3))
    self.assertTrue(jnp.isfinite(jnp.array(cell_from_g)).all())
    self.assertNotEqual(float(jnp.linalg.det(jnp.array(cell_from_g))), 0.0)

  def test_get_irreducible_k_mesh_basic(self):
    kpts_frac, w = k_vectors(
      cell_vectors=self.simple_cell,
      grid_sizes=[2, 2, 2],
      scaled_positions=self.scaled_positions,
      charges=self.charges,
    )
    self.assertEqual(kpts_frac.shape[1], 3)
    self.assertEqual(w.ndim, 1)
    np.testing.assert_allclose(jnp.sum(jnp.array(w)), 1.0, atol=1e-12)
