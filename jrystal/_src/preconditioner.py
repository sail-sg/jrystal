"""Preconditioner for total energy in the plane wave basis.
"""
import jax
import jax.numpy as jnp
from typing import Callable, Any
from jaxtyping import Complex, Array, Int, Float
from jrystal import pw
from .hamiltonian import hamiltonian_matrix_trace
from .hessian import hessian_diag_pytree, get_hvp_fn
from einops import einsum


def preconditioner_hessian_diag(
  params: Complex[Array, "s k b x y z"],
  freq_mask: Int[Array, "x y z"],
  occupation: Float[Array, "s k b"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = True,
) -> Complex[Array, "s k b x y z"]:

  """Preconditioner for total energy in the plane wave basis.

  The preconditioner is the inverse of the diagonal elements of the Hessian
  matrix of the total energy with respect to the plane wave parameters.

  It returns an array or a Pytree that has the same shape and structure as the
  plane wave parameters.

  """

  def _f(prm):
    # prm is a dict, and the values are real-valued arrays.
    # the shape the leaves is (s, k, g, b)
    coeff = pw.coeff(prm, freq_mask)  # [s k b x y z]
    density = pw.density_grid(coeff, vol, occupation)  # [s k b x y z]
    energy = hamiltonian_matrix_trace(
      coeff,
      positions,
      charges,
      density,
      g_vector_grid,
      kpts,
      vol,
      xc,
      kohn_sham,
    )

    diag_identity = einsum(
      coeff, coeff.conj(), "s k b x y z, s k b x y z -> s k b"
    ) * 100

    return jnp.sum(energy) + jnp.sum(diag_identity)

  output = hessian_diag_pytree(_f)(params)
  return output


def preconditioner_neumann(
  fun: Callable[[Any], float],
  primal: Any,
  max_iter: int = 50,
  max_eigval: float = 300,
) -> Callable[[Any], Any]:
  """
  Calculate the preconditioner via the Neumann series.

  .. math::
    A^-1 = a \sum_{i = 0}^{k} (I - aA)^i

  """
  hvp_fn = get_hvp_fn(fun, primal)

  alpha = 1. / max_eigval

  def precond(cotangent):
    def scan_fn(carry, _):
      sum, x = carry
      diff = jax.tree.map(lambda a: alpha * a, hvp_fn(x))
      x = jax.tree.map(lambda a, b: a - b, x, diff)
      sum = jax.tree.map(lambda a, b: a + b, sum, x)
      return (sum, x), _

    # sum = jnp.copy(cotangent)
    carry, _ = jax.lax.scan(
      scan_fn, (cotangent, cotangent), length=max_iter
    )
    return jax.tree.map(lambda a: a * alpha, carry[0])

  return precond
