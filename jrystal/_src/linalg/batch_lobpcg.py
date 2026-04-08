from __future__ import annotations

from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array


def _apply_operator(operator, X):
  X2 = X if X.ndim >= 2 else X[..., None]
  if operator is None:
    return X2
  Y = operator(X2)
  return Y if Y.ndim == X2.ndim else Y[..., None]


def _symmetrize_hermitian(matrix):
  return 0.5 * (matrix + jnp.swapaxes(jnp.conj(matrix), -1, -2))


def _gram_matrix(X, Y=None, b_matmul=None):
  if Y is None:
    Y = X
  BY = _apply_operator(b_matmul, Y)
  return jnp.einsum("...ni,...nj->...ij", jnp.conj(X), BY)


def _hermitian_inv_sqrt(matrix):
  matrix = _symmetrize_hermitian(matrix)
  evals, evecs = jnp.linalg.eigh(matrix)
  real_dtype = jnp.real(matrix).dtype
  eps = jnp.finfo(real_dtype).eps
  scale = jnp.max(jnp.abs(evals), axis=-1, keepdims=True)
  floor = jnp.maximum(scale * eps * matrix.shape[-1], eps)
  inv_sqrt = jnp.reciprocal(jnp.sqrt(jnp.maximum(evals, floor)))
  return jnp.einsum(
    "...ik,...k,...jk->...ij",
    evecs,
    inv_sqrt,
    jnp.conj(evecs),
  )


def block_mgs(W, V=None, reorth=True, b_matmul=None):
  """Orthonormalize columns under the B-inner product."""
  if W.ndim == 1:
    W = W[:, None]
  if W.shape[-1] == 0:
    return jnp.zeros_like(W)

  def _project_out(vector, basis):
    if basis is None or basis.size == 0:
      return vector
    coeff = _gram_matrix(basis, vector, b_matmul)
    return vector - jnp.matmul(basis, coeff)

  real_dtype = jnp.real(W).dtype
  eps = jnp.asarray(jnp.finfo(real_dtype).eps * W.shape[-2], dtype=real_dtype)
  columns = []

  for col_idx in range(W.shape[-1]):
    vector = W[..., :, col_idx:col_idx + 1]
    vector = _project_out(vector, V)

    if columns:
      basis = jnp.concatenate(columns, axis=-1)
      vector = _project_out(vector, basis)
      if reorth:
        vector = _project_out(vector, V)
        vector = _project_out(vector, basis)

    norm_sq = jnp.real(_gram_matrix(vector, b_matmul=b_matmul)[..., 0, 0])
    safe_norm = jnp.sqrt(jnp.maximum(norm_sq, eps))
    vector = vector / safe_norm[..., None, None]
    vector = jnp.where(
      norm_sq[..., None, None] <= eps,
      jnp.zeros_like(vector),
      vector,
    )
    columns.append(vector)

  return jnp.concatenate(columns, axis=-1)


def block_mgs_batch(W, V=None, reorth=True, b_matmul=None):
  """Alias kept for compatibility with the old LOBPCG implementation."""
  return block_mgs(W, V=V, reorth=reorth, b_matmul=b_matmul)


def _projected_generalized_eigh(T, B, k, which):
  T = _symmetrize_hermitian(T)
  if B is None:
    evals, vecs = jnp.linalg.eigh(T)
  else:
    B = _symmetrize_hermitian(B)
    eigvals_B, eigvecs_B = jnp.linalg.eigh(B)
    real_dtype = jnp.real(B).dtype
    eps = jnp.finfo(real_dtype).eps
    scale = jnp.max(jnp.abs(eigvals_B), axis=-1, keepdims=True)
    floor = jnp.maximum(scale * eps * B.shape[-1], eps)
    valid = eigvals_B > floor
    safe_eigvals_B = jnp.maximum(eigvals_B, floor)
    inv_sqrt = jnp.where(
      valid,
      jnp.reciprocal(jnp.sqrt(safe_eigvals_B)),
      jnp.zeros_like(eigvals_B),
    )
    inv_sqrt_B = jnp.einsum(
      "...ik,...k,...jk->...ij",
      eigvecs_B,
      inv_sqrt,
      jnp.conj(eigvecs_B),
    )
    null_proj = jnp.einsum(
      "...ik,...k,...jk->...ij",
      eigvecs_B,
      jnp.where(valid, 0.0, 1.0).astype(real_dtype),
      jnp.conj(eigvecs_B),
    )
    T_whitened = jnp.matmul(
      jnp.swapaxes(jnp.conj(inv_sqrt_B), -1, -2),
      jnp.matmul(T, inv_sqrt_B),
    )
    T_whitened = _symmetrize_hermitian(T_whitened)
    penalty = (
      jnp.max(jnp.abs(T_whitened), axis=(-2, -1), keepdims=True) + 1.0
    ) * T.shape[-1]
    if which == "smallest":
      T_whitened = T_whitened + penalty * null_proj
    else:
      T_whitened = T_whitened - penalty * null_proj
    evals, whitened_vecs = jnp.linalg.eigh(T_whitened)
    vecs = jnp.matmul(inv_sqrt_B, whitened_vecs)

  evals = jnp.real(evals)
  order = jnp.argsort(evals, axis=-1)
  order = order[..., ::-1] if which == "largest" else order
  idx = order[..., :k]
  idx_expanded = jnp.broadcast_to(idx[..., None, :], vecs.shape[:-1] + (k,))
  eigvecs = jnp.take_along_axis(vecs, idx_expanded, axis=-1)
  eigvals = jnp.take_along_axis(evals, idx, axis=-1)
  return eigvals, eigvecs


def batch_lobpcg_matrix_free(
  matmul: Callable,
  k: int,
  n=None,
  tol=1e-8,
  maxit=200,
  v0=None,
  batch_size=None,
  preconditioner=None,
  b_matmul: Callable | None = None,
  which="largest",
  reorth=True,
  seed=0,
  return_history=False,
) -> Union[Tuple[Array, Array], Tuple[Array, Array, Array]]:
  """LOBPCG for batched extremal eigenpairs, optionally generalized by B."""
  if v0 is None and n is None:
    raise ValueError("Provide n when v0 is None.")
  if v0 is None and batch_size is None:
    raise ValueError("Provide batch_size when v0 is None for batched mode.")
  if which not in ("largest", "smallest"):
    raise ValueError('which must be "largest" or "smallest"')

  def _matmul_cols(X):
    return _apply_operator(matmul, X)

  def _b_matmul_cols(X):
    return _apply_operator(b_matmul, X)

  def _apply_precond(X):
    if preconditioner is None:
      return X
    X2 = X if X.ndim >= 2 else X[..., None]
    if callable(preconditioner):
      Y = preconditioner(X2)
      return Y if Y.ndim == X2.ndim else Y[..., None]
    P = jnp.asarray(preconditioner)
    if P.ndim == 1:
      P = P.reshape((1,) * (X2.ndim - 2) + (P.shape[0], 1))
    elif P.ndim == 2 and P.shape[-1] == 1 and P.shape[0] == X2.shape[-2]:
      P = P.reshape((1,) * (X2.ndim - 2) + P.shape)
    elif P.ndim == X2.ndim - 1:
      P = P[..., None]
    return P * X2

  if v0 is not None:
    X = jnp.asarray(v0)
    if X.ndim == 1:
      X = X[:, None]
    if X.shape[-1] > k:
      X = X[..., :k]
    n = X.shape[-2]
  else:
    key = jax.random.PRNGKey(seed)
    X = jax.random.normal(key, (batch_size, n, k))

  AX0 = _matmul_cols(X)
  if AX0.dtype != X.dtype:
    X = X.astype(AX0.dtype)
  X = block_mgs_batch(X, V=None, reorth=reorth, b_matmul=b_matmul)
  P = jnp.zeros_like(X)

  def _update(X, P):
    AX = _matmul_cols(X)
    BX = _b_matmul_cols(X)
    theta = jnp.real(jnp.sum(jnp.conj(X) * AX, axis=-2))
    R = AX - BX * theta[..., None, :]
    Z = _apply_precond(R)

    basis = jnp.concatenate([X, P, Z], axis=-1)
    basis = block_mgs_batch(basis, V=None, reorth=reorth, b_matmul=b_matmul)
    AB = _matmul_cols(basis)
    T = jnp.einsum("...ni,...nj->...ij", jnp.conj(basis), AB)
    Bproj = None
    if b_matmul is not None:
      BB = _b_matmul_cols(basis)
      Bproj = jnp.einsum("...ni,...nj->...ij", jnp.conj(basis), BB)

    _, vecs = _projected_generalized_eigh(T, Bproj, k, which)
    X_new = jnp.einsum("...ni,...ik->...nk", basis, vecs)
    X_new = block_mgs_batch(X_new, V=None, reorth=reorth, b_matmul=b_matmul)

    overlap_x = _gram_matrix(X, X_new, b_matmul=b_matmul)
    delta = X_new - jnp.matmul(X, overlap_x)
    P_new = block_mgs_batch(delta, V=None, reorth=reorth, b_matmul=b_matmul)

    AX_new = _matmul_cols(X_new)
    BX_new = _b_matmul_cols(X_new)
    theta_new = jnp.real(jnp.sum(jnp.conj(X_new) * AX_new, axis=-2))
    R_new = AX_new - BX_new * theta_new[..., None, :]
    res = jnp.linalg.norm(R_new, axis=-2)
    return X_new, P_new, res

  res0 = jnp.full(X.shape[:-2] + (k,), jnp.inf, dtype=jnp.real(X).dtype)

  def _step(carry, _):
    X, P, res, converged = carry

    def _do_update(_):
      Xn, Pn, resn = _update(X, P)
      conv = jnp.max(resn) < tol
      return Xn, Pn, resn, conv

    def _skip(_):
      return X, P, res, converged

    Xn, Pn, resn, conv = jax.lax.cond(
      converged,
      _skip,
      _do_update,
      operand=None,
    )
    return (Xn, Pn, resn, conv), resn

  def _body(i, carry):
    del i
    (Xn, Pn, resn, conv), _ = _step(carry, None)
    return (Xn, Pn, resn, conv)

  if return_history:
    (X, P, res, _), history = jax.lax.scan(
      _step,
      (X, P, res0, False),
      None,
      length=maxit,
    )
  else:
    X, P, res, _ = jax.lax.fori_loop(0, maxit, _body, (X, P, res0, False))

  AX = _matmul_cols(X)
  evals = jnp.real(jnp.sum(jnp.conj(X) * AX, axis=-2))
  order = jnp.argsort(evals, axis=-1)
  order = order[..., ::-1] if which == "largest" else order
  evals = jnp.take_along_axis(evals, order, axis=-1)[..., :k]
  order = jnp.broadcast_to(order[..., None, :], X.shape[:-1] + (k,))
  evecs = jnp.take_along_axis(X, order, axis=-1)[..., :k]

  if return_history:
    hist = jnp.moveaxis(history, 0, -2)
    return evals, evecs, hist

  return evals, evecs


lobpcg_matrix_free_batched = batch_lobpcg_matrix_free
