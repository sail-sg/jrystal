import jax
import jax.numpy as jnp


def diis_init(max_hist, density_shape, dtype=jnp.float32):
  """Initialize DIIS state.

  Args:
    max_hist: int. Maximum number of densities/errors to store.
    density_shape: tuple. Shape of a single density array.
    dtype: dtype. Array dtype for densities/errors.

  Returns:
    state: dict with 'densities', 'errors', 'size', 'head'.
  """
  densities = jnp.zeros((max_hist,) + tuple(density_shape), dtype=dtype)
  errors = jnp.zeros_like(densities)
  size = jnp.array(0, dtype=jnp.int32)
  head = jnp.array(0, dtype=jnp.int32)
  return {
    "densities": densities,
    "errors": errors,
    "size": size,
    "head": head,
  }


def _diis_mix(densities, errors, size, eps=1e-12):
  m = densities.shape[0]
  mask = (jnp.arange(m) < size).astype(errors.dtype)
  mask_mat = mask[:, None] * mask[None, :]

  E = errors.reshape((m, -1))
  B = jnp.matmul(jnp.conj(E), jnp.transpose(E))
  B = B * mask_mat
  B = B + jnp.eye(m, dtype=B.dtype) * (1.0 - mask)
  B = B + jnp.eye(m, dtype=B.dtype) * eps

  a = mask
  A = jnp.zeros((m + 1, m + 1), dtype=B.dtype)
  A = A.at[:m, :m].set(B)
  A = A.at[:m, m].set(a)
  A = A.at[m, :m].set(a)
  rhs = jnp.zeros((m + 1,), dtype=B.dtype)
  rhs = rhs.at[m].set(1.0)
  sol = jnp.linalg.solve(A, rhs)
  coeff = sol[:m]
  mixed = jnp.tensordot(coeff, densities, axes=(0, 0))
  return mixed


@jax.jit
def diis_update(state, density, error, eps=1e-12):
  """Update DIIS state and return mixed density (jittable).

  Args:
    state: dict. Output of diis_init.
    density: ndarray. New density.
    error: ndarray. New error/residual associated with density.
    eps: float. Diagonal jitter for stability.

  Returns:
    new_state: dict. Updated state.
    mixed_density: ndarray. DIIS-mixed density.
  """
  densities = state["densities"]
  errors = state["errors"]
  size = state["size"]
  head = state["head"]

  densities = jax.lax.dynamic_update_index_in_dim(
    densities, density, head, axis=0
  )
  errors = jax.lax.dynamic_update_index_in_dim(
    errors, error, head, axis=0
  )

  max_hist = densities.shape[0]
  head = (head + 1) % max_hist
  size = jnp.minimum(size + 1, max_hist)

  def _mix(_):
    return _diis_mix(densities, errors, size, eps=eps)

  def _fallback(_):
    return density

  mixed = jax.lax.cond(size < 2, _fallback, _mix, operand=None)
  new_state = {
    "densities": densities,
    "errors": errors,
    "size": size,
    "head": head,
  }
  return new_state, mixed


__all__ = ["diis_init", "diis_update"]
