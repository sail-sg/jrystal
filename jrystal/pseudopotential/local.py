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
import numpy as np
import jax.numpy as jnp
from interpax import CubicSpline
from jaxtyping import Array, Complex, Float

from .._src import braket
from ..sbt import sbt, sbt_numerical
from ..grid import g2r_vector_grid
from .utils import map_over_atoms


def potential_local_reciprocal(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  local_potential_grid: List[Float[Array, "r"]],
  local_potential_charge: List[int],
  vol: float,
  fourier_transform_method: str = "sbt"
) -> Float[Array, "x y z"]:
  """Calculate the local potential in reciprocal space.

  This function computes the local pseudopotential contribution in
  reciprocal space by transforming the real-space local potentials
  and applying structure factors.

  .. math::

    < v_loc | G >

  Args:
    positions (Float[Array, "atom 3"]): Atomic positions in Cartesian
    coordinates.
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal space grid vectors.
    r_grid (List[Float[Array, "r"]]): Real-space radial grids for each atom
    type.
    local_potential_grid (List[Float[Array, "r"]]): Local potential values on
    radial grids.
    local_potential_charge (List[int]): Nuclear charges for each atom type.
    vol (float): Unit cell volume.
    fourier_transform (str): Fourier transform method. Can be either "sbt" or
    "fft".

  Returns:
    Float[Array, "x y z"]: Local potential in reciprocal space grid.
  """
  # Convert reciprocal to real space grid
  # First part v1 = (v_loc + Z/r)
  # shape: [na, nr]
  r_vector_grid = g2r_vector_grid(g_vector_grid)
  g_radius = jnp.sqrt(jnp.sum(g_vector_grid**2, axis=-1))
  r_radius = jnp.sqrt(jnp.sum(r_vector_grid**2, axis=-1))
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))
  local_potential_charge = jnp.array(local_potential_charge)

  #############################################################
  if fourier_transform_method == "sbt":
    # V_r + Z/r - Z/r, where V_r + Z/r is numerical and Z/r is analytical
    @map_over_atoms
    def g(r, v_r, z):
      v_r_prime = v_r + z / r
      v_r_prime = jnp.expand_dims(v_r_prime, axis=0)
      # kk, f_k = sbt(r, v_r_prime, l=0, kmax=np.max(g_radius), norm=False)
      kk, f_k = sbt_numerical(r, v_r_prime, l=0, kmax=np.max(g_radius))
      f_k = CubicSpline(kk, f_k[0], axis=1)(g_radius)
      f_k *= 4 * jnp.pi
      #  factor of 4 * pi is due to the fourier transform of Yukawa potential
      return f_k

  elif fourier_transform_method == "fft":
    @map_over_atoms
    def g(r, v_r, z):
      v_r_modified = v_r + z / r
      cs = CubicSpline(r, v_r_modified)
      v_r_3d = cs(r_radius)
      v_g_3d = jnp.fft.fftn(v_r_3d, axes=range(-3, 0)) / num_grids
      return v_g_3d * 4 * jnp.pi

  else:
    raise ValueError(
      f"Invalid fourier transform method: {fourier_transform_method}."
      f"Can only be 'sbt' or 'fft'."
    )

  v1_g = g(r_grid, local_potential_grid, local_potential_charge)
  v1_g = jnp.stack(v1_g)  # shape: [na x y z]

  # Compute v2: Coulomb tail correction in reciprocal space
  # Avoid division by zero at G=0
  g_radius_safe = g_radius.at[0, 0, 0].set(1e10)
  charges_expanded = jnp.expand_dims(
    jnp.array(local_potential_charge), axis=(1, 2, 3)
  )
  v2_g = charges_expanded / jnp.expand_dims(g_radius_safe**2, 0)
  v2_g *= 4 * jnp.pi

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
  return v_g


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
