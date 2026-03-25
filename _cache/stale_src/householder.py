import jax
import jax.numpy as jnp

from .utils import vmapstack


def householder(x: jnp.ndarray) -> jnp.ndarray:
  """Construct orthonormal columns using Householder reflections.

  Args:
    x (jnp.ndarray): Input tensor with shape ``(..., n, m)`` and ``n >= m``.

  Returns:
    jnp.ndarray: Tensor with shape ``(..., n, m)`` whose columns are
    orthonormal.
  """
  n, m = x.shape[-2:]
  assert n >= m, "n should be greater than or equal to m"
  batch_dims = len(x.shape) - 2

  @vmapstack(times=batch_dims)
  def _householder(x):

    def householder_transform(u):
      u = u / jnp.linalg.norm(u)
      return jnp.eye(len(u)) - 2 * jnp.outer(u, u.conj())

    def fn(P, u):
      return P @ householder_transform(u), None

    P, _ = jax.lax.scan(fn, jnp.eye(x.shape[0], dtype=x.dtype), x.T)

    return P.at[:, :x.shape[1]].get()

  return _householder(x)
