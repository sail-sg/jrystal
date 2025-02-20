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
  positions: Float[Array, "na 3"],
  r_vector_grid: Float[Array, "*nd d"],
  g_vector_grid: Float[Array, "*nd d"],
  r_grid: List[Float[Array, "num_r"]],
  local_potential_grid: List[Float[Array, "num_r"]],
  local_potential_charge: List[int],
  vol: float,
) -> Complex[Array, "n1 n2 n3"]:
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
  v1_g = jnp.stack(v1_g)  # shape [na n1 n2 n3]

  # Second part v2 = - Z/r in reciprocal space
  g_radius = g_radius.at[0, 0, 0].set(1e10)
  v2_g = jnp.expand_dims(
    local_potential_charge, axis=(1, 2, 3)
  ) / jnp.expand_dims(g_radius**2, 0)  # [natom n1 n2 n3]
  v2_g *= 4 * jnp.pi
  v2_g = v2_g.at[:, 0, 0, 0].set(0)

  # sum up
  v_g = v1_g - v2_g
  v_g = v_g.at[:, 0, 0, 0].set(0)

  ################################################################
  # structure factor
  structure_factor = jnp.exp(
    -1.j * jnp.matmul(g_vector_grid, positions.transpose())
  )  # shape: [n1 n2 n3 na]
  structure_factor = jnp.transpose(structure_factor, axes=(3, 0, 1, 2))
  v_g *= structure_factor
  v_g = jnp.sum(v_g, axis=0)  # reduce over atoms

  v_g = v_g * num_grids / vol
  return v_g / 2  # factor of 1/2 is due to the conversion of unit.


def _hamiltonian_local(
  wave_grid: Complex[Array, "*batchs n1 n2 n3"],
  potential_local_grid_reciprocal,
  vol: Float,
) -> Complex[Array, "ns nk nb nb"]:
  v_loc_r = jnp.fft.ifftn(potential_local_grid_reciprocal, axes=range(-3, 0))
  hamil = braket.expectation(
    wave_grid, v_loc_r, vol, diagonal=False, mode="real"
  )
  return hamil


def hamiltonian_local(
  wave_grid: Complex[Array, "*batchs n1 n2 n3"],
  positions: Float[Array, "na 3"],
  r_vector_grid: Float[Array, "*nd d"],
  g_vector_grid: Float[Array, "*nd d"],
  r_grid: Float[Array, "nr"],
  local_potential_grid: Float[Array, "nr"],
  local_potential_charge: Float[Array, "na"],
  vol: Float,
) -> Complex[Array, "ns nk nb nb"]:
  v_loc_g = _potential_local_reciprocal(
    positions,
    r_vector_grid,
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
  reciprocal_density_grid: Complex[Array, "*batchs n1 n2 n3"],
  v_local_reciprocal: Complex[Array, "n1 n2 n3"],
  vol: Float,
) -> Float:
  # this function is for split the potential and density from energy
  # calculation such that it is differentiable with respect to the density,
  # and can be jitted.

  return braket.reciprocal_braket(
    v_local_reciprocal, reciprocal_density_grid, vol
  )


def energy_local(
  reciprocal_density_grid: Complex[Array, "*batchs n1 n2 n3"],
  positions: Float[Array, "na 3"],
  r_vector_grid: Float[Array, "*nd d"],
  g_vector_grid: Float[Array, "*nd d"],
  r_grid: Float[Array, "nr"],
  local_potential_grid: Float[Array, "nr"],
  local_potential_charge: Float[Array, "na"],
  vol: Float,
) -> Float:
  v_external_reciprocal = _potential_local_reciprocal(
    positions,
    r_vector_grid,
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
