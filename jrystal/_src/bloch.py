""" Internal functions for PlaneWave class
"""
import numpy as np
import jax
from jax import core
import jax.numpy as jnp
import jax.interpreters.ad as ad
from functools import partial

# import jax._src.ad_util as ad_util
from jax.interpreters import mlir
from .grid import g_vectors, r_vectors


def u(a, cg, r, *, force_fft=False):
  r"""This is the periodic part of the bloch wave function.
  Which is denoted in u on wikipedia.
  Args:
    a: the lattice vectors in the real space, which has shape `(nd, nd)`.
    cg: a tensor of shape `(batch_dims_of_cg..., n1, n2, ..., nd)`
      coefficient to linearly combine over different g.
      here `batch_dims_of_cg...` is often used for different i, k in the
      bloch wave function $\psi_{ik}$.
    r: a coordinate in the real space.
      a vector of shape (batch_dims_of_r..., d),
      here `batch_dims_of_r...` is often used for the evaluating
      on a 3d grid.
    force_fft: whether to force fft, it should only be used when `r` is
      created from `r_vectors(a, cg.shape[-ndim:])`.
  Returns:
    a complex tensor that is the wave function value at `r`.
    the shape of the tensor is `(batch_dims_of_cg..., batch_dims_of_r...)`.

    .. math::
      \sum_{g} c_{g} \exp(\mathrm{i}G_gr)
  """
  return u_p.bind(a, cg, r, force_fft=force_fft)


u_p = core.Primitive("u")


def _u(g_vec, r, cg):
  r"""This is the plain implementation
  We follow the math
  .. math::
    \sum_{g} c_{g} \exp(\mathrm{i}G_gr)
  """
  ndim = r.shape[-1]
  gr = jnp.tensordot(g_vec, r, axes=(-1, -1))
  expigr = jnp.exp(1.j * gr)
  return jnp.tensordot(
    cg, expigr, axes=(tuple(range(-ndim, 0)), tuple(range(ndim)))
  )


def _u_map(g_vec, r, cg):
  """This is the implementation using jax.lax.map
  When the size of the G points are large, the plain implementation
  could result in OOM.
  `_u_map` simply iterate over the batch dimension of `r`, and call
  `_u` on each element in `r`.
  """
  ndim = r.shape[-1]
  r_flat = jnp.reshape(r, (-1, ndim))
  out = jnp.moveaxis(jax.lax.map(lambda r: _u(g_vec, r, cg), r_flat), 0, -1)
  out = jnp.reshape(out, (*cg.shape[:-ndim], *r.shape[:-1]))
  return out


def _u_fft(g_vec, r, cg):
  """This is the fft version of `_u`.
  When the `r` happen to be the `r_vectors` of the crystal,
  we can use fft to accelerate the computation of `_u`.
  """
  ndim = r.shape[-1]
  grid_sizes = cg.shape[-ndim:]
  out = np.prod(grid_sizes) * jnp.fft.ifftn(cg, axes=tuple(range(-ndim, 0)))
  out = jnp.reshape(out, (*cg.shape[:-ndim], *r.shape[:-1]))
  return out


def _u_impl(a, cg, r, *, force_fft=False):
  """The implementation of the `u` primitive.
  One of `_u`, `_u_map` and `_u_fft` will be called, based on `r` passed.
  """
  ndim = a.shape[-1]
  assert r.shape[-1] == ndim
  assert cg.ndim >= ndim
  grid_sizes = cg.shape[-ndim:]
  g_vec = g_vectors(a, grid_sizes)
  r_vec = r_vectors(a, grid_sizes)

  if force_fft:
    if np.prod(r.shape) != np.prod(r_vec.shape):
      raise ValueError(
        "force_fft is True, but r is not the r_vectors of the crystal."
      )
    return _u_fft(g_vec, r, cg)

  if np.prod(r.shape) == np.prod(r_vec.shape):
    pred = jnp.array_equal(r.flatten(), r_vec.flatten())
    # if `r` happen to be the `r_vectors` of the crystal,
    # we use `_u_fft` to accelerate the computation.
    # otherwise, we call `_u_map`.
    # we don't use the plain `_u` here as it will cause oom.
    # Even if the `pred` is `True` XLA will allocate memory
    # for the unexecuted branch.
    return jax.lax.cond(pred, _u_fft, _u_map, g_vec, r, cg)
  else:
    # if `r` is not the `r_vectors` of the crystal,
    # then we assume the user is just passing some random `r`
    # with a small batch size, therefore we call `_u` directly
    # without the fear of oom.
    return _u(g_vec, r, cg)


u_p.def_impl(_u_impl)
mlir.register_lowering(u_p, mlir.lower_fun(_u_impl, False))


@u_p.def_abstract_eval
def _u_p_abstract_eval(a, cg, r, *, force_fft):
  ndim = a.shape[1]
  if r.dtype == jnp.float32:
    dtype = jnp.complex64
  elif r.dtype == jnp.float64:
    dtype = jnp.complex128
  return core.ShapedArray(cg.shape[:-ndim] + r.shape[:-1], dtype=dtype)


def _u_jvp_rule(primals, tangents, *, force_fft):
  r"""This defines the JVP rule for the `u` primitive.
  We linearize the `u` function at the `primals`, and evaluated
  the linearized function on `tangents`.

  JVP wrt `cg` is simple.

  JVP wrt `a` is a bit tricky. Notice `u` doesn't directly depend on `a`,
  but it depends on `g_vec`, which depends on `a`. Therefore, we need to
  linearize `g_vec` wrt `a` first, then linearize `u` wrt `g_vec`.
  While `g_vec` wrt `a` can be done by calling `jvp` on `g_vectors`,
  here we show the math for the linearization of `u` wrt `g_vec`:

  .. math::
    &u(c, r, G) = \sum_{g} c_{g} \exp(\mathrm{i}G_gr) \\
    &\frac{\partial{u}}{\partial{G_g}} \delta{G_g} =
    c_{g} \mathrm{i}\delta{G_g}r\exp(\mathrm{i}G_gr)

  This is equal to applying FFT on $c_{g}\mathrm{i}\delta{G_g}$,
  notice that $\delta{G_g}$ has shape `(n1, n2, n3, 3)`, so does
  $c_{g}\mathrm{i}\delta{G_g}$. The FFT is applied on the `(n1, n2, n3)` axes.
  After the FFT, we can then reduce the `(3,)` axes by contracting with `r`.

  JVP wrt `r` is similar.

  .. math::
    \frac{\partial{u}}{\partial{r}}\delta{r} =
    \sum_{g}c_{g}\mathrm{i}G_{g}\exp(\mathrm{i}G_gr)\delta{r}

  """
  a, cg, r = primals
  a_dot, cg_dot, r_dot = tangents
  ndim = a.shape[1]
  g_vec = g_vectors(a, cg.shape[-ndim:])

  jvps = []
  if not isinstance(a_dot, ad.Zero):
    # if we linearize `u` wrt `a`, since `u` depends on `a` via `g_vec`,
    # The first step we linearize the `g_vectors` function wrt `a`.
    gv = lambda a: g_vectors(a, cg.shape[-ndim:])
    g_vec_dot = jax.jvp(gv, (a,), (a_dot,))[1]
    # The second step, we linearize the `u` function wrt `g_vec`.
    cgigdot = jnp.expand_dims(cg,
                              -ndim - 1) * 1.j * jnp.moveaxis(g_vec_dot, -1, 0)
    jvp1 = jnp.sum(
      u(a, cgigdot, r, force_fft=force_fft) * jnp.moveaxis(r, -1, 0),
      axis=-r.ndim
    )
    jvps.append(jvp1)
  if not isinstance(cg_dot, ad.Zero):
    # `u` is linear in `cg` (FFT), therefore jvp just replaces `cg` with `cg_dot`.
    jvp2 = u(a, cg_dot, r, force_fft=force_fft)
    jvps.append(jvp2)
  if not isinstance(r_dot, ad.Zero):
    cgig = jnp.expand_dims(cg, -ndim - 1) * 1.j * jnp.moveaxis(g_vec, -1, 0)
    jvp3 = jnp.sum(
      u(a, cgig, r, force_fft=force_fft) * jnp.moveaxis(r_dot, -1, 0),
      axis=-r.ndim
    )
    jvps.append(jvp3)

  return u(a, cg, r, force_fft=force_fft), sum(jvps)


def _u_transpose_rule(cotangent, a, cg, r, *, force_fft):
  ndim = a.shape[1]
  if ad.is_undefined_primal(a) or ad.is_undefined_primal(r):
    # `u` is not linear to `a` or `r`, therefore we can't compute the transpose.
    raise ValueError(
      "Transpose of u is not defined wrt a or r, "
      "because u is not linear wrt a or r. "
      "This shouldn't happen because jax always linearize before transpose."
    )
  elif ad.is_undefined_primal(cg):

    def _transpose_u_fft(cotangent):
      return jax.linear_transpose(partial(_u_fft, g_vec, r),
                                  cg.aval)(cotangent)[0]

    def _transpose_u(cotangent):
      return jax.linear_transpose(partial(_u_map, g_vec, r),
                                  cg.aval)(cotangent)[0]

    def _transpose_u_map(cotangent):
      """lax.cond will allocate memory for both branches, which will cause oom.
      we use map here to avoid oom.
      """
      ndim = r.shape[-1]
      r_flat = jnp.reshape(r, (-1, ndim))
      ctg_flat = jnp.moveaxis(
        jnp.reshape(cotangent, (*cotangent.shape[:-(r.ndim - 1)], -1)), -1, 0
      )
      carry, _ = jax.lax.scan(
        lambda carry, r_ctg: (
          carry + jax.linear_transpose(partial(_u, g_vec, r_ctg[0]), cg.aval)
          (r_ctg[1])[0], 0.
        ),
        init=jnp.zeros_like(cg.aval),
        xs=(r_flat, ctg_flat)
      )
      return carry

    g_vec = g_vectors(a, cg.aval.shape[-ndim:])
    r_vec = r_vectors(a, cg.aval.shape[-ndim:])

    if force_fft:
      if np.prod(r.shape) != np.prod(r_vec.shape):
        raise ValueError(
          "force_fft is True, but r is not the r_vectors of the crystal."
        )
      return _transpose_u_fft(cotangent)

    if np.prod(r.shape) == np.prod(r_vec.shape):
      pred = jnp.array_equal(r.flatten(), r_vec.flatten())
      wrt_cg = jax.lax.cond(pred, _transpose_u_fft, _transpose_u_map, cotangent)
    else:
      wrt_cg = _transpose_u(cotangent)
    return (None, wrt_cg, None)


def _u_batch_rule(batched_args, batch_dims, *, force_fft):
  a, cg, r = batched_args
  ndim = a.shape[-1]
  bd_a, bd_cg, bd_r = batch_dims
  if batch_dims == (None, 0, None):
    bdim = 0
  elif batch_dims == (None, None, 0):
    bdim = cg.ndim - ndim
  else:
    raise NotImplementedError
  return u(a, cg, r, force_fft=force_fft), bdim


jax.interpreters.ad.primitive_jvps[u_p] = _u_jvp_rule
jax.interpreters.ad.primitive_transposes[u_p] = _u_transpose_rule
jax.interpreters.batching.primitive_batchers[u_p] = _u_batch_rule


def bloch_wave(a, cg, k_vec):
  r"""Return a wave function that takes a coordinate `r` as input,
  that generates the wave function value at `r` for all `k`.

  .. math::
    \psi_{ik}(r) = \exp(\mathrm{i}kr) \sum_{g} c_{ikg}
    \exp(\mathrm{i}G_gr)

  Args:
    a: the lattice vectors in the real space, which has shape (nd, nd).
    cg: a tensor of shape (batch..., nk, n1, n2, ..., nd)
      coefficient to linearly combine different g.
    k_vec: a grid of k vectors (nk, d)

  Returns:
    wave: a wave function that takes r as input.
  """
  ndim = a.shape[1]
  grid_sizes = cg.shape[-ndim:]
  g_vec = g_vectors(a, grid_sizes)  # noqa
  # r_vec is used to check whether fft should be used.
  r_vec = r_vectors(a, grid_sizes)  # noqa

  def wave(r, force_fft=False):
    """Return the wave function value at `r`.

    Args:
      r: a coordinate in the real space.
        a vector of shape `(d,)`
    Returns:
      a complex tensor that is the wave function value at `r`.
      the shape of the tensor is `(batch, ..., nk)`
    """
    sum_cg_expigr = u(a, cg, r, force_fft=force_fft)
    kr = jnp.tensordot(k_vec, r, axes=(-1, -1))
    expikr = jnp.exp(1.j * kr)
    return sum_cg_expigr * expikr

  return wave
