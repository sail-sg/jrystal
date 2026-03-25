"""Hartree-Fock helper functions."""

from typing import Optional

import jax.numpy as jnp
import numpy as np
from einops import einsum
from jaxtyping import Array, Complex, Float

from . import pw
from .utils import absolute_square


def paired_density_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
) -> Complex[Array, "spin k1 k2 band1 band2  x y z"]:
  r"""Compute paired real-space densities :math:`\rho_{ik,jk'}`.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.

  Returns:
    Complex[Array, "spin k1 k2 band1 band2  x y z"]: Paired density tensor.
  """
  wave_grid = pw.wave_grid(coeff, vol)
  paired_den = einsum(
    wave_grid,
    wave_grid.conj(),
    "s k1 b1 x y z, s k2 b2 x y z -> s k1 k2 b1 b2 x y z"
  )

  return paired_den


def paired_density_grid_reciprocal(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
) -> Complex[Array, "spin k1 k2 band1 band2  x y z"]:
  paired_den = paired_density_grid(coeff, vol)
  return jnp.fft.fftn(paired_den, axes=range(-3, 0))


def exchange_energy(
  paired_density_grid_reciporcal: Complex[Array, "s k1 k2 b1 b2 x y z"],
  g_vector_grid: Float[Array, 'x y z 3'],
  occupation: Optional[Float[Array, 'spin kpt band']],
  vol: Float,
) -> Float:
  dim = 3
  spin_restricted = paired_density_grid_reciporcal.shape[0] == 1
  g_vec_square = jnp.sum(g_vector_grid**2, axis=-1)  # [x y z]
  g_vec_square = g_vec_square.at[(0,) * dim].set(1)
  inv_g_sqaure = 1. / (g_vec_square)
  inv_g_sqaure = inv_g_sqaure.at[(0,) * dim].set(0)  # remove G = 0

  e_x = einsum(
    absolute_square(paired_density_grid_reciporcal),
    inv_g_sqaure,
    "s k1 k2 b1 b2 x y z, x y z -> s k1 k2 b1 b2"
  ) * 2 * jnp.pi  # E = 1/2 * 4 pi |n(G)|^2 / |G|^2

  e_x = einsum(
    e_x, occupation, occupation, "s k1 k2 b1 b2, s k1 b1, s k2 b2 -> "
  )

  num_grids = np.prod(np.array(g_vec_square.shape))
  parseval_factor = 1 / num_grids
  numerical_integral_weight = vol / num_grids

  e_x = e_x * parseval_factor * numerical_integral_weight
  if spin_restricted:
    e_x /= 2  # exchange does not count different spin channel.

  return -e_x


def gygi_baldereschi(
  paired_density_grid_reciporcal: Complex[Array, "s k1 k2 b1 b2 x y z"],
  occupation: Optional[Float[Array, 'spin kpt band']],
  g_vector_grid: Float[Array, "x y z 3"],
  vol: Float,
  alpha: float = 0.1,
) -> Float:

  assert g_vector_grid.shape[-1] == 3
  num_grids = np.prod(np.array(g_vector_grid.shape[:3]))

  def fun(x):
    return jnp.pi * 4 * jnp.exp(-alpha * x) / x

  g_vec_square = jnp.sum(g_vector_grid**2, axis=-1)  # [x y z]
  g_vec_square = g_vec_square.at[(0,) * 3].set(1)
  fun_gvec_grid = fun(g_vec_square)
  fun_gvec_grid = fun_gvec_grid.at[(0,) * 3].set(0)

  e_x = einsum(
    absolute_square(paired_density_grid_reciporcal),
    fun_gvec_grid,
    "s k k b1 b2 x y z, x y z -> s k b1 b2"
  ) * 2 * jnp.pi  # E = 1/2 * 4 pi |n(G)|^2 / |G|^2

  e_x = einsum(e_x, occupation, occupation, "s k b1 b2, s k b1, s k b2 -> ")

  parseval_factor = 1 / num_grids
  numerical_integral_weight = vol / num_grids

  e_x *= parseval_factor * numerical_integral_weight
  spin_restricted = paired_density_grid_reciporcal.shape[0] == 1
  if spin_restricted:
    e_x /= 2  # exchange does not count different spin channel.

  div = e_x / vol - 1 / jnp.sqrt(alpha * jnp.pi
                                ) * jnp.sum(occupation) / 2 / jnp.pi

  # return e_x  * 4 * jnp.pi
  return div
