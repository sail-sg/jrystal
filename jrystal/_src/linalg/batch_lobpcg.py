from typing import Tuple, Union

import jax
import jax.numpy as jnp
from jaxtyping import Array


def block_mgs(W, V=None, reorth=True):
  """Orthonormalize columns with QR."""
  if W.ndim == 1:
    W = W[:, None]
  if W.shape[1] == 0:
    return jnp.zeros_like(W)
  if V is not None and V.size:
    Vh = jnp.conj(V).T
    W = W - V @ (Vh @ W)
    if reorth:
      W = W - V @ (Vh @ W)
  Q, _ = jnp.linalg.qr(W, mode="reduced")
  return Q


def block_mgs_batch(W, V=None, reorth=True):
  """Batched block MGS using vmap over the leading batch axes."""
  if W.ndim <= 2:
    return block_mgs(W, V=V, reorth=reorth)
  batch_shape = W.shape[:-2]
  W_flat = W.reshape((-1,) + W.shape[-2:])

  if V is None:
    Q_flat = jax.vmap(lambda Wi: block_mgs(Wi, V=None, reorth=reorth))(W_flat)
  else:
    V_arr = jnp.asarray(V)
    if V_arr.ndim <= 2:
      V_arr = jnp.broadcast_to(V_arr, batch_shape + V_arr.shape)
    V_flat = V_arr.reshape((-1,) + V_arr.shape[-2:])
    Q_flat = jax.vmap(lambda Wi, Vi: block_mgs(Wi, V=Vi, reorth=reorth)
                     )(W_flat, V_flat)

  return Q_flat.reshape(batch_shape + Q_flat.shape[-2:])


def batch_lobpcg_matrix_free(
  matmul,
  k,
  n=None,
  tol=1e-8,
  maxit=200,
  v0=None,
  batch_size=None,
  preconditioner=None,
  which="largest",
  reorth=True,
  seed=0,
  return_history=False,
) -> Union[Tuple[Array, Array], Tuple[Array, Array, Array]]:
  """LOBPCG for batched extremal eigenpairs (matrix-free).

  Args:
    matmul: callable. matmul(X) with X shape (..., n, p) returns A@X.
    k: int. Number of eigenpairs.
    n: int. Problem size (required if v0 is None).
    tol: float. Convergence threshold on residual norms.
    maxit: int. Maximum iterations.
    v0: ndarray or None. Initial block, shape (..., n, k).
    batch_size: int or None. Required if v0 is None and batch dim is needed.
    preconditioner: callable, ndarray, or None. If callable, M^{-1}(X) with
      X shape (..., n, p). If ndarray, elementwise scaling; vector shape (n,)
      or (..., n) is broadcast across columns.
    which: "largest" or "smallest".
    reorth: bool. Use reorthogonalization in MGS.
    seed: int. RNG seed if v0 is None.
    return_history: bool. If True, return residual history.
  """
  if v0 is None and n is None:
    raise ValueError("Provide n when v0 is None.")
  if v0 is None and batch_size is None:
    raise ValueError("Provide batch_size when v0 is None for batched mode.")
  if which not in ("largest", "smallest"):
    raise ValueError('which must be "largest" or "smallest"')

  def _matmul_cols(X):
    X2 = X if X.ndim >= 2 else X[..., None]
    AX = matmul(X2)
    return AX if AX.ndim == X2.ndim else AX[..., None]

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

  # Initialize X
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

  X = block_mgs_batch(X, V=None, reorth=reorth)
  AX0 = _matmul_cols(X)
  if AX0.dtype != X.dtype:
    X = X.astype(AX0.dtype)
  P = jnp.zeros_like(X)

  def _update(X, P):
    AX = _matmul_cols(X)
    w = jnp.sum(jnp.conj(X) * AX, axis=-2)
    R = AX - X * w[..., None, :]
    Z = _apply_precond(R)
    res = jnp.linalg.norm(R, axis=-2)

    B = jnp.concatenate([X, P, Z], axis=-1)
    B = block_mgs_batch(B, V=None, reorth=reorth)
    AB = _matmul_cols(B)
    T = jnp.einsum("...ni,...nj->...ij", jnp.conj(B), AB)
    evals, vecs = jnp.linalg.eigh(T)
    evals = jnp.real(evals)
    idx = jnp.argsort(evals, axis=-1)
    idx = idx[..., ::-1] if which == "largest" else idx
    idxk = idx[..., :k]
    idxk = jnp.broadcast_to(idxk[..., None, :], vecs.shape[:-1] + (k,))
    vecs = jnp.take_along_axis(vecs, idxk, axis=-1)
    X_new = jnp.einsum("...ni,...ik->...nk", B, vecs)
    X_new = block_mgs_batch(X_new, V=None, reorth=reorth)

    # Update P as component of new direction orthogonal to X
    XHX = jnp.einsum("...ni,...nj->...ij", jnp.conj(X), X_new)
    proj = jnp.einsum("...ni,...ij->...nj", X, XHX)
    delta = X_new - proj
    P_new = block_mgs_batch(delta, V=None, reorth=reorth)
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
      converged, _skip, _do_update, operand=None
    )
    return (Xn, Pn, resn, conv), resn

  def _body(i, carry):
    (Xn, Pn, resn, conv), _ = _step(carry, None)
    return (Xn, Pn, resn, conv)

  if return_history:
    (X, P, res, _), history = jax.lax.scan(
      _step, (X, P, res0, False), None, length=maxit
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
