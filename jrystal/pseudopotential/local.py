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
"""Local potential functions for plane waves.

This module provides functions for calculating local pseudopotential
contributions in plane wave basis sets, including reciprocal space
transformations and Hamiltonian matrix elements.
"""
from typing import List

import jax.numpy as jnp
from interpax import CubicSpline
from jax.tree_util import tree_map
from jaxtyping import Array, Complex, Float

from .._src import braket
from ..grid import g2r_vector_grid


def potential_local_reciprocal(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  local_potential_grid: List[Float[Array, "r"]],
  local_potential_charge: List[int],
  vol: float,
) -> Float[Array, "x y z"]:
  """Calculate the local potential in reciprocal space.

  This function computes the local pseudopotential contribution in
  reciprocal space by transforming the real-space local potentials
  and applying structure factors.

  .. math::

    v_{\text{loc}}(\mathbf{r}) =
    \sum_I v^{\text{loc}}_{I} \bigg( \overline{\mathbf{r} - \mathbf{R}_I} \bigg)

    \tilde{v}_{\text{loc}}(\mathbf{G}) = \sum_I \exp(-\text{i} \mathbf{G}
    \cdot \mathbf{R}_I) \mathcal{F}[v^{\text{loc}}_{I}](\mathbf{G})

  Args:
    positions (Float[Array, "atom 3"]): Atomic positions in Cartesian coordinates.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal space grid vectors.
    r_grid (List[Float[Array, "r"]]): Real-space radial grids for each atom type.
    local_potential_grid (List[Float[Array, "r"]]): Local potential values on radial grids.
    local_potential_charge (List[int]): Nuclear charges for each atom type.
    vol (float): Unit cell volume.

  Returns:
    Float[Array, "x y z"]: Local potential in reciprocal space grid.
  """
  # Convert reciprocal to real space grid
  r_vector_grid = g2r_vector_grid(g_vector_grid)
  r_radius = jnp.sqrt(jnp.sum(r_vector_grid**2, axis=-1))
  g_radius = jnp.sqrt(jnp.sum(g_vector_grid**2, axis=-1))
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))

  def _g(r, v_r, z):
    """Com  pute the modified potential component for a single atom type."""
    # Add Coulomb tail: v_r + z/r
    v_r_modified = v_r + z / r
    # Interpolate to 3D grid
    cs = CubicSpline(r, v_r_modified)
    v_r_3d = cs(r_radius)
    # Transform to reciprocal space
    v_g_3d = jnp.fft.fftn(v_r_3d, axes=range(-3, 0)) / num_grids
    return v_g_3d

  # Compute v1: interpolated local potential in reciprocal space
  v1_g = tree_map(_g, r_grid, local_potential_grid, local_potential_charge)
  v1_g = jnp.stack(v1_g)  # shape: [na x y z]

  # Compute v2: Coulomb tail correction in reciprocal space
  # Avoid division by zero at G=0
  g_radius_safe = g_radius.at[0, 0, 0].set(1e10)
  charges_expanded = jnp.expand_dims(
    jnp.array(local_potential_charge), axis=(1, 2, 3)
  )
  v2_g = charges_expanded / jnp.expand_dims(g_radius_safe**2, 0)
  v2_g *= 4 * jnp.pi
  # Set G=0 component to zero (neutralizing background)
  v2_g = v2_g.at[:, 0, 0, 0].set(0)

  # Combine v1 and v2 components
  v_g = v1_g - v2_g
  v_g = v_g.at[:, 0, 0, 0].set(0)  # Neutralize G=0 component

  # Apply structure factor: exp(-i G · R)
  structure_factor = jnp.exp(
    -1.j * jnp.matmul(g_vector_grid, positions.transpose())
  )  # shape: [x y z na]
  structure_factor = jnp.transpose(structure_factor, axes=(3, 0, 1, 2))
  v_g *= structure_factor
  v_g = jnp.sum(v_g, axis=0)  # Sum over atoms

  # Apply normalization factors
  v_g = v_g * num_grids / vol
  return v_g / 2  # Factor of 1/2 due to unit conversion


def hamiltonian_local(
  wave_grid: Complex[Array, "spin kpt band x y z"],
  potential_local_grid_reciprocal: Complex[Array, "x y z"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  """Compute the local potential Hamiltonian matrix elements.

  This function calculates the matrix elements of the local potential
  operator in the basis of wave functions. The potential is transformed
  from reciprocal to real space before computing the matrix elements.


  Args:
    wave_grid: Wave functions in real space grid.
    potential_local_grid_reciprocal: Local potential in reciprocal space.
        Can be obtained from :func:`potential_local_reciprocal`.
    vol: Unit cell volume.

  Returns:
    Local potential Hamiltonian matrix elements.
  """
  # Transform potential from reciprocal to real space
  v_loc_real = jnp.fft.ifftn(potential_local_grid_reciprocal, axes=range(-3, 0))
  # Compute matrix elements
  hamiltonian = braket.expectation(
    wave_grid, v_loc_real, vol, diagonal=False, mode="real"
  )
  return hamiltonian


def energy_local(
  reciprocal_density_grid: Complex[Array, "spin kpt band x y z"],
  potential_local_grid_reciprocal: Complex[Array, "x y z"],
  vol: Float,
) -> Float:
  """Compute the local potential energy in reciprocal space.

  This function calculates the energy contribution from the local potential
  by computing the inner product between the potential and density in
  reciprocal space.

  Args:
    reciprocal_density_grid (Complex[Array, "spin kpt band x y z"]): Electron
    density in reciprocal space.
    potential_local_grid_reciprocal (Complex[Array, "x y z"]): Local potential
    in reciprocal space. Can be obtained from `potential_local_reciprocal`.
    vol (Float): Unit cell volume.

  Returns:
    Float: Local potential energy contribution.
  """
  return braket.reciprocal_braket(
    potential_local_grid_reciprocal, reciprocal_density_grid, vol
  )
