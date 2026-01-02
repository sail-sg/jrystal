"""Function for hartree fock method. """

from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int
from einops import einsum

from . import braket, potential, pw, xc
from .ewald import ewald_coulomb_repulsion
from .grid import translation_vectors
from .utils import (
  absolute_square, safe_real, wave_to_density, wave_to_density_reciprocal
)


def paired_density_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
) -> Complex[Array, "spin k1 k2 band1 band2  x y z"]:
  r"""Calculate the paired density :math:`\rho_{ik, jk'}`.

  Computes the (ik, jk') paired density tensor, where :math:`i` and :math:`j`
  denote the band indices,
  and :math:`k` and :math:`k'` the k-point indices.

  The paired density is defined as:

    .. math::

      \rho_{ik, jk'} = \psi^*_{ik}(\mathbf{r}) \psi_{jk'}(\mathbf{r})

  where :math:`\psi_{ik}(\mathbf{r})` is the wave function for band :math:`i`
  and k-point :math:`k` evaluated at position :math:`\mathbf{r}`.
  """
  wave_grid = pw.wave_grid(coeff, vol)
  paired_den = einsum(
    wave_grid.conj(),
    wave_grid,
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
  inv_g_sqaure = 1. / g_vec_square
  inv_g_sqaure.at[(0,) * dim].set(0)     # remove G = 0

  e_x = einsum(
    absolute_square(paired_density_grid_reciporcal),
    inv_g_sqaure,
    "s k1 k2 b1 b2 x y z, x y z -> s k1 k2 b1 b2"
  ) * 2 * jnp.pi   # E = 1/2 * 4 pi |n(G)|^2 / |G|^2

  e_x = einsum(
    e_x,
    occupation,
    occupation,
    "s k1 k2 b1 b2, s k1 b1, s k2 b2 -> "
  )

  num_grids = np.prod(np.array(g_vec_square.shape))
  parseval_factor = 1 / num_grids
  numerical_integral_weight = vol / num_grids

  e_x = e_x * parseval_factor * numerical_integral_weight
  if spin_restricted:
    e_x /= 2      # exchange does not count different spin channel.

  return -e_x
