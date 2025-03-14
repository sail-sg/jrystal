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
"""Ewald summation for periodic systems. """

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def ewald_coulomb_repulsion(
  positions: Float[Array, 'atom d'],
  charges: Float[Array, 'atom'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  ewald_eta: Float,
  ewald_grid: Float[Array, 'x y z 3'],
) -> Float:
  """ Calculate the nuclei repulsion energy using Ewald summation method.

  This function computes the nuclei repulsion energy for a periodic system using the Ewald summation technique, which splits the calculation into real and reciprocal space contributions. The method provides an efficient way to handle long-range Coulomb interactions in periodic systems.

  .. note::
    Further reading:

    - Textbook: Martin, R. M. (2020). Electronic structure: basic theory and practical methods. Cambridge university press. (Appendix F.2)
    - Our tutorial: :doc:`Ewald Summation </tutorial/ewald>`

  Args:
    positions (Float[Array, 'atom d']): Array of shape (atom, d) containing atomic positions in d-dimensional space.
    charges (Float[Array, 'atom']): Array of shape (atom,) containing the charges of each atom.
    g_vector_grid (Float[Array, 'x y z 3']): Array of shape (x, y, z, 3) containing the reciprocal lattice vectors.
    vol (Float): Float representing the volume of the unit cell.
    ewald_eta (Float): Float controlling the split between real and reciprocal space contributions. Also known as the Ewald convergence parameter.
    ewald_grid (Float[Array, 'x y z 3']): Array of shape (x, y, z, 3) containing real-space translation vectors for the Ewald sum. Can be generated using :func:`jrystal.grid.translation_vectors`.

  Returns:
    Float: The total Coulomb repulsion energy computed using Ewald summation. Includes both real-space and reciprocal-space contributions, as well as self-interaction corrections.
  """
  dim = positions.shape[-1]

  tau = jnp.expand_dims(positions, 0) - jnp.expand_dims(positions, 1)
  tau_t = jnp.expand_dims(tau, 2) - jnp.expand_dims(ewald_grid, axis=(0, 1))
  # [na, na, nt, 3]
  tau_t_norm = jnp.sqrt(jnp.sum(tau_t**2, axis=-1) + 1e-20)  # [na, na, nt]
  tau_t_norm = jnp.where(tau_t_norm <= 1e-9, 1e20, tau_t_norm)

  #  atom-atom part:
  ew_ovlp = jnp.sum(
    jax.scipy.special.erfc(ewald_eta * tau_t_norm) / tau_t_norm, axis=2
  )

  # the reciprocal space part:
  gvec_norm_sq = jnp.sum(g_vector_grid**2, axis=3)  # [x y z]
  gvec_norm_sq = gvec_norm_sq.at[(0,) * dim].set(1e16)

  ew_rprcl = jnp.exp(-gvec_norm_sq / 4 / ewald_eta**2) / gvec_norm_sq
  ew_rprcl1 = jnp.expand_dims(ew_rprcl, range(dim, dim + 2))
  ew_rprcl2 = jnp.cos(
    jnp.sum(
      jnp.expand_dims(g_vector_grid, axis=(-2, -3)) *
      jnp.expand_dims(tau, range(dim)),
      axis=-1
    )
  )  # [x y z, na, na, nt]
  ew_rprcl2 = ew_rprcl2.at[(0,) * dim].set(0)  # this is to exclude G = 0
  ew_rprcl = jnp.sum(ew_rprcl1 * ew_rprcl2, axis=range(dim))  # [na, na]
  ew_rprcl = ew_rprcl * 4 * jnp.pi / vol
  ew_aa = jnp.einsum('i,ij->j', charges, ew_ovlp + ew_rprcl)
  ew_aa = jnp.dot(ew_aa, charges) / 2

  # single atom part
  ew_a = -jnp.sum(charges**2) * 2 * ewald_eta / jnp.sqrt(jnp.pi) / 2
  ew_a -= jnp.sum(charges)**2 * jnp.pi / ewald_eta**2 / vol / 2

  return ew_aa + ew_a
