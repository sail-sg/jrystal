"""Kinetic operator."""
import jax.numpy as jnp
from jaxtyping import Float, Array

import einops
from .typing import VectorGrid


def kinetic(
  g_vector_grid: VectorGrid[Float, 3],
  k_points: Float[Array, "num_k 3"] = None,
) -> Array:
  """The Kinetic operator in reciprocal space.

    See. Eq. (12.5). Martin, Richard M. 2020.

    .. math::
      \hat T \psi> = - \dfrac12 \nabla^2 \psi>

    Usage:

      >>> from .module import PlaneWave
      >>> from .ops import expecation

      >>> pw = PlaneWave(...)
      >>> coefficient = pw.coefficient()    # real space
      >>> kinetic = hamiltonian.kinetic(...)
      >>> kinetic_energy = expectation(coefficient, kinetic, mode='kinetic')

  Args:
      g_vector_grid (VectorGrid[Float, 3]): _description_
      k_points (Float[Array, 'num_k 3'], optional): _description_.
        Defaults to None.

  Returns:
      Array: _description_
  """
  k_points = jnp.zeros([1, 3]) if k_points is None else k_points
  k_points = einops.rearrange(k_points, "nk d -> nk 1 1 1 d")
  return jnp.sum((g_vector_grid + k_points)**2, axis=-1) / 2
