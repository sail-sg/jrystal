"""Hessian for Complex-Valued Functions. """
import jax
import jax.numpy as jnp
from typing import Callable


def complex_hessian(
  f: Callable[[jnp.ndarray], jnp.ndarray], 
  primal: jnp.ndarray,
) -> jnp.ndarray:
  """Compute the Hessian of a complex-valued function at a point. 

  .. warning::

    Only for :math:`x` being a vector is tested.

  Args:
    f: A function that takes a complex-valued array and returns a complex-valued array.
    primal: The point at which to compute the Hessian.

  Returns:
    The Hessian of the function at the point.
  """
  dtype = primal.dtype
  dim = primal.shape[-1]

  def grad_f(x):
    value, vjp_fn = jax.vjp(f, x)
    return vjp_fn(jnp.ones(value.shape, dtype=dtype))[0]

  def hessian_vec_prod(x, v):
    _, vjp_fn = jax.vjp(grad_f, x)
    return vjp_fn(v)[0]

  # x = jnp.ones(dim, dtype=dtype)
  return jnp.array(
    [hessian_vec_prod(primal, jnp.eye(dim, dtype=dtype)[i]) for i in range(dim)]
    )
