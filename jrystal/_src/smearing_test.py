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
"""Tests for smearing.py."""

import unittest

import jax.numpy as jnp
import numpy as np

from .smearing import (
  _bisect_root,
  fermi_dirac,
  find_chemical_potential,
  occupations_from_eigenvalues,
)


class _TestSmearing(unittest.TestCase):

  def test_bisect_root_supports_decreasing_residual(self):
    root = _bisect_root(
      lambda x: 1.0 - x,
      jnp.array(0.0),
      jnp.array(2.0),
      max_iter=60,
    )

    self.assertAlmostEqual(float(root), 1.0, places=10)

  def test_bisect_root_raises_for_unbracketed_interval(self):
    with self.assertRaisesRegex(ValueError, "not bracketed"):
      _bisect_root(
        lambda x: x * x + 1.0,
        jnp.array(-1.0),
        jnp.array(1.0),
        max_iter=20,
      )

  def test_find_chemical_potential_tiny_smearing_uniform_occ_per_k(self):
    eigenvalues = jnp.array(
      [
        [
          [-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, 1.0, 2.0, 3.0, 4.0],
          [-4.8, -3.8, -2.8, -2.1, -1.4, -0.9, 1.2, 2.1, 3.1, 4.1],
          [-5.2, -4.1, -3.1, -2.2, -1.6, -1.1, 0.8, 1.9, 2.8, 3.8],
        ]
      ],
      dtype=jnp.float32,
    )
    k_weights = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
    num_electrons = 12.0

    mu = find_chemical_potential(
      eigenvalues,
      num_electrons,
      smearing=1e-9,
      k_weights=k_weights,
    )
    occupation = fermi_dirac(eigenvalues, mu, smearing=1e-9)

    occupied_per_k = np.asarray(jnp.sum(occupation[0] > 1.0, axis=-1))
    np.testing.assert_array_equal(occupied_per_k, np.array([6, 6, 6]))
    self.assertAlmostEqual(
      float(jnp.sum(occupation * k_weights[None, :, None])),
      num_electrons,
      places=8,
    )

  def test_find_chemical_potential_finite_smearing_weighted_count(self):
    eigenvalues = jnp.array(
      [[
        [-1.0, -0.2, 0.1, 0.8],
        [-0.9, 0.05, 0.2, 1.0],
      ]],
      dtype=jnp.float32,
    )
    k_weights = jnp.array([0.5, 0.5], dtype=jnp.float32)
    num_electrons = 4.0

    mu = find_chemical_potential(
      eigenvalues,
      num_electrons,
      smearing=0.1,
      k_weights=k_weights,
    )
    occupation = fermi_dirac(eigenvalues, mu, smearing=0.1)

    self.assertAlmostEqual(
      float(jnp.sum(occupation * k_weights[None, :, None])),
      num_electrons,
      places=6,
    )

  def test_occupations_from_eigenvalues_zero_smearing_sorted_fill(self):
    eigenvalues = jnp.array(
      [
        [
          [-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, 1.0, 2.0, 3.0, 4.0],
          [-4.8, -3.8, -2.8, -2.1, -1.4, -0.9, 1.2, 2.1, 3.1, 4.1],
          [-5.2, -4.1, -3.1, -2.2, -1.6, -1.1, 0.8, 1.9, 2.8, 3.8],
        ]
      ],
      dtype=jnp.float32,
    )
    k_weights = jnp.array([0.2, 0.3, 0.5], dtype=jnp.float32)
    num_electrons = 12.0

    mu, occupation = occupations_from_eigenvalues(
      eigenvalues,
      num_electrons,
      smearing=0.0,
      k_weights=k_weights,
    )
    del mu

    occupied_per_k = np.asarray(jnp.sum(occupation[0] > 1.0, axis=-1))
    np.testing.assert_array_equal(occupied_per_k, np.array([6, 6, 6]))
    self.assertAlmostEqual(
      float(jnp.sum(occupation * k_weights[None, :, None])),
      num_electrons,
      places=8,
    )

  def test_occupations_from_eigenvalues_small_smearing_sorted_fill(self):
    eigenvalues = jnp.array(
      [[
        [-2.0, -1.0, 0.5, 1.5],
        [-1.9, -0.9, 0.6, 1.6],
      ]],
      dtype=jnp.float32,
    )
    k_weights = jnp.array([0.5, 0.5], dtype=jnp.float32)
    num_electrons = 4.0

    mu, occupation = occupations_from_eigenvalues(
      eigenvalues,
      num_electrons,
      smearing=1e-9,
      k_weights=k_weights,
      small_smearing_ratio=1e-6,
    )
    self.assertTrue(np.isfinite(float(mu)))
    np.testing.assert_array_equal(
      np.asarray(jnp.sum(occupation[0] > 1.0, axis=-1)),
      np.array([2, 2]),
    )
    self.assertAlmostEqual(
      float(jnp.sum(occupation * k_weights[None, :, None])),
      num_electrons,
      places=8,
    )
