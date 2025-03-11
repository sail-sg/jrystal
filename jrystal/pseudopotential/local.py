"""Local potential functions for plane waves. """
import numpy as np
import jax.numpy as jnp
from typing import List
from jaxtyping import Float, Array, Complex

from .utils import map_over_atoms
from ..sbt import sbt
from .._src import braket
from .interpolate import cubic_spline


def _potential_local_reciprocal(
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: List[Float[Array, "r"]],
  local_potential_grid: List[Float[Array, "r"]],
  local_potential_charge: List[int],
  vol: float,
) -> Complex[Array, "x y z"]:
  """Calculate the local potential in reciprocal space.
  
  .. note::
    This function is for split the potential and density from energy calculation such that it is differentiable with respect to the density, and can be jitted.

  Args:
      positions (Float[Array, "atom 3"]): The positions of the atoms.
      g_vector_grid (Float[Array, "x y z 3"]): The grid of the reciprocal vectors.
      r_grid (List[Float[Array, "r"]]): The grid of the real space.
      local_potential_grid (List[Float[Array, "r"]]): The local potential in real space.
      local_potential_charge (List[int]): The charge of the local potential.
      vol (float): The volume of the unit cell.

  Returns:
      Complex[Array, "x y z"]: The local potential in reciprocal space.
  """
  
  # v_loc = (v_loc + Z/r) - Z/r   where v_loc is always negative

  # First part v1 = (v_loc + Z/r)
  # shape: [na, nr]
  g_radius = jnp.sqrt(jnp.sum(g_vector_grid**2, axis=-1))
  num_grids = jnp.prod(jnp.array(g_vector_grid.shape[:-1]))
  local_potential_charge = jnp.array(local_potential_charge)

  #############################################################
  # V_r + Z/r - Z/r, where V_r + Z/r is numerical and Z/r is analytical
  @map_over_atoms
  def g(r, v_r, z):
    v_r_prime = v_r + z / r
    kk, f_k = sbt(r, v_r_prime, l=0, kmax=np.max(g_radius), norm=False)
    f_k = cubic_spline(kk, f_k, g_radius)
    f_k *= 4 * np.pi
    return f_k

  v1_g = g(r_grid, local_potential_grid, local_potential_charge)
  v1_g = jnp.stack(v1_g)  # shape [na x y z]

  # Second part v2 = - Z/r in reciprocal space
  g_radius = g_radius.at[0, 0, 0].set(1e10)
  v2_g = jnp.expand_dims(
    local_potential_charge, axis=(1, 2, 3)
  ) / jnp.expand_dims(g_radius**2, 0)  # [natom x y z]
  v2_g *= 4 * jnp.pi
  v2_g = v2_g.at[:, 0, 0, 0].set(0)

  # sum up
  v_g = v1_g - v2_g
  v_g = v_g.at[:, 0, 0, 0].set(0)

  ################################################################
  # structure factor
  structure_factor = jnp.exp(
    -1.j * jnp.matmul(g_vector_grid, positions.transpose())
  )  # shape: [x y z na]
  structure_factor = jnp.transpose(structure_factor, axes=(3, 0, 1, 2))
  v_g *= structure_factor
  v_g = jnp.sum(v_g, axis=0)  # reduce over atoms

  v_g = v_g * num_grids / vol
  return v_g / 2  # factor of 1/2 is due to the conversion of unit.


def _hamiltonian_local(
  wave_grid: Complex[Array, "spin kpt band x y z"],
  potential_local_grid_reciprocal: Complex[Array, "x y z"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  """The local potential hamiltonian in real space.
  
  .. note::
    This function is for split the potential and density from energy calculation such that it is differentiable with respect to the density, and can be jitted.

  
  Args:
    wave_grid (Complex[Array, "spin kpt band x y z"]): The wave function in real space.
    potential_local_grid_reciprocal (Complex[Array, "x y z"]): The local potential in reciprocal space.
    vol (Float): The volume of the unit cell.

  Returns:
      Complex[Array, "spin kpt band band"]: The local potential hamiltonian in real space.
  """
  v_loc_r = jnp.fft.ifftn(potential_local_grid_reciprocal, axes=range(-3, 0))
  hamil = braket.expectation(
    wave_grid, v_loc_r, vol, diagonal=False, mode="real"
  )
  return hamil


def hamiltonian_local(
  wave_grid: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: Float[Array, "r"],
  local_potential_grid: Float[Array, "r"],
  local_potential_charge: Float[Array, "atom"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  """The local potential hamiltonian in reciprocal space.
  
  Args:
    wave_grid (Complex[Array, "spin kpt band x y z"]): The wave function in real space.
    positions (Float[Array, "atom 3"]): The positions of the atoms.
    g_vector_grid (Float[Array, "x y z 3"]): The grid of the reciprocal vectors.
    r_grid (Float[Array, "r"]): The grid of the real space.
    local_potential_grid (Float[Array, "r"]): The local potential in real space.
    local_potential_charge (Float[Array, "atom"]): The charge of the local potential.
    vol (Float): The volume of the unit cell.

  Returns:
      Complex[Array, "spin kpt band band"]: The local potential hamiltonian in reciprocal space.
  """
  v_loc_g = _potential_local_reciprocal(
    positions,
    g_vector_grid,
    r_grid,
    local_potential_grid,
    local_potential_charge,
    vol
  )
  v_loc_r = jnp.fft.ifftn(v_loc_g, axes=range(-3, 0))
  hamil = braket.expectation(
    wave_grid, v_loc_r, vol, diagonal=False, mode="real"
  )
  return hamil


def _energy_local(
  reciprocal_density_grid: Complex[Array, "spin kpt band x y z"],
  v_local_reciprocal: Complex[Array, "x y z"],
  vol: Float,
) -> Float:
  """The local potential energy in reciprocal space.

  .. note::
    This function is for split the potential and density from energy calculation such that it is differentiable with respect to the density, and can be jitted.

  Args:
      reciprocal_density_grid (Complex[Array, "spin kpt band x y z"]): The reciprocal density grid.
      v_local_reciprocal (Complex[Array, "x y z"]): The local potential in reciprocal space.
      vol (Float): The volume of the unit cell.

  Returns:
      Float: The local potential energy in reciprocal space.
  """

  return braket.reciprocal_braket(
    v_local_reciprocal, reciprocal_density_grid, vol
  )


def energy_local(
  reciprocal_density_grid: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  r_grid: Float[Array, "r"],
  local_potential_grid: Float[Array, "r"],
  local_potential_charge: Float[Array, "atom"],
  vol: Float,
) -> Float:
  """The local potential energy in reciprocal space.

  Args:
    reciprocal_density_grid (Complex[Array, "spin kpt band x y z"]): The reciprocal density grid.
    positions (Float[Array, "atom 3"]): The positions of the atoms.
    g_vector_grid (Float[Array, "x y z 3"]): The grid of the reciprocal vectors.
  
  Returns:
    Float: The local potential energy in reciprocal space.
  """
  v_external_reciprocal = _potential_local_reciprocal(
    positions,
    g_vector_grid,
    r_grid,
    local_potential_grid,
    local_potential_charge,
    vol
  )
  external_energy = braket.reciprocal_braket(
    v_external_reciprocal, reciprocal_density_grid, vol
  )
  return external_energy.real
