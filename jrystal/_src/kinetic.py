"""Kinetic operator."""
import jax.numpy as jnp
from jaxtyping import Float, Array

import einops
from ._typing import VectorGrid


def kinetic(
  g_vector_grid: VectorGrid[Float, 3],
  k_points: Float[Array, "num_k 3"] = None,
) -> Array:
  """Compute Kinetic energy in reciprocal space.

  This function calculates the kinetic energy matrix element evaluated at
  each k-point and reciprocal lattice vector which will be contracted with
  the coefficient matrix to obtain the kinetic energy.
    
  Args:
      g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
      k_points (Float[Array, 'num_k 3'], optional): k-points. Defaults to None.

  Returns:
      Float[Array, 'num_k ... 3']: kinetic energy matrix element evaluated at
      each k-point and reciprocal lattice vector. 
  """
  k_points = jnp.zeros([1, 3]) if k_points is None else k_points
  k_points = einops.rearrange(k_points, "nk d -> nk 1 1 1 d")
  return jnp.sum((g_vector_grid + k_points)**2, axis=-1) / 2
