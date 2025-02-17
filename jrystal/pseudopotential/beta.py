"""Functions for dealing with beta functions. """
import jax.numpy as jnp
from typing import List
from jaxtyping import Float, Array
from ..sbt import batched_sbt
from .interpolate import cubic_spline


def beta_sbt_grid_single_atom(
  r_grid: Float[Array, "num_r"],
  nonlocal_beta_grid: Float[Array, "num_beta num_r"],
  nonlocal_angular_momentum: List[int],
  g_vector_grid: Float[Array, "n1 n2 n3 3"],
) -> Float[Array, "num_beta n1 n2 n3"]:
  """calculate the spherical bessel transform of the beta functions for a
    single atom.

  ..math::
    beta_l(g) = int_0^\infty  beta(r) j_l(gr) r^2 dr

  Return the beta function value of angular momentum values l at the
    g_vector_grid per atom

  Args:
      r_grid (Float[Array, "num_r"]): the r grid corresponding to the beta
        functions.
      nonlocal_beta_grid (Float[Array, "num_beta num_r"]): beta values.
      nonlocal_angular_momentum (List[int]): angular momentum corresponding
        to the beta functions.
      g_vector_grid (Float[Array, "n1 n2 n3 3"]): reciprocal vectors to
        interpolate.

  Returns:
      Float[Array, "num_beta n1 n2 n3"]: the beta functions in reciprocal space.

  Warning: Cubic spline interpolation is not implemented in JAX. This function
    uses NumPy and is not differentiable.

  """
  assert len(nonlocal_angular_momentum) == nonlocal_beta_grid.shape[0]
  assert r_grid.shape[0] == nonlocal_beta_grid.shape[1]

  radius = jnp.sqrt(jnp.sum(g_vector_grid**2, axis=-1))
  k, beta_k = batched_sbt(
    r_grid, nonlocal_beta_grid, l=nonlocal_angular_momentum,
    kmax=jnp.max(radius)
  )

  beta_sbt = cubic_spline(k, beta_k, radius)
  return beta_sbt


def beta_sbt_grid_multi_atoms(
  r_grid: List[Float[Array, "num_r"]],
  nonlocal_beta_grid: List[Float[Array, "num_beta num_r"]],
  nonlocal_angular_momentum: List[List[int]],
  g_vector_grid: Float[Array, "n1 n2 n3 3"],
) -> List[Float[Array, "num_beta n1 n2 n3"]]:

  output = []
  for r, b, l in zip(r_grid, nonlocal_beta_grid, nonlocal_angular_momentum):
    output.append(beta_sbt_grid_single_atom(r, b, l, g_vector_grid))
  return jnp.stack(output)
