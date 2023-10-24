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
from ._grid import g_vectors, r_vectors, grid_1d
from jaxtyping import Float, Array, Int, Complex

def u(a, cg, r):
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
  Returns:
    a complex tensor that is the wave function value at `r`.
    the shape of the tensor is `(batch_dims_of_cg..., batch_dims_of_r...)`.
    .. math::
      \sum_{g} c_{g} \exp(\mathrm{i}G_gr)
  """
  return u_p.bind(a, cg, r)

u_p = core.Primitive("u")

def _u(g_vec, r, cg):
  ndim = r.shape[-1]
  gr = jnp.tensordot(g_vec, r, axes=(-1, -1))
  expigr = jnp.exp(1.j * gr)
  return jnp.tensordot(
    cg, expigr, axes=(tuple(range(-ndim, 0)), tuple(range(ndim)))
  )

def _u_map(g_vec, r, cg):
  """This is because, although jax will not compute the branch not
  activated in cond, it seems to be allocate memories, which will cause an oom
  we use map here to avoid oom. This won't be called at all if r_vec is passed.
  """
  ndim = r.shape[-1]
  r_flat = jnp.reshape(r, (-1, ndim))
  out = jnp.moveaxis(jax.lax.map(lambda r: _u(g_vec, r, cg), r_flat), 0, -1)
  out = jnp.reshape(out, (*cg.shape[:-ndim], *r.shape[:-1]))
  return out

def _u_fft(g_vec, r, cg):
  ndim = r.shape[-1]
  grid_sizes = cg.shape[-ndim:]
  out = np.prod(grid_sizes) * jnp.fft.ifftn(cg, axes=tuple(range(-ndim, 0)))
  out = jnp.reshape(out, (*cg.shape[:-ndim], *r.shape[:-1]))
  return out

def _u_impl(a, cg, r):
  ndim = a.shape[-1]
  assert r.shape[-1] == ndim
  assert cg.ndim >= ndim
  grid_sizes = cg.shape[-ndim:]
  g_vec = g_vectors(a, grid_sizes)
  r_vec = r_vectors(a, grid_sizes)

  if np.prod(r.shape) == np.prod(r_vec.shape):
    pred = jnp.array_equal(r.flatten(), r_vec.flatten())
    return jax.lax.cond(pred, _u_fft, _u_map, g_vec, r, cg)
  else:
    return _u(g_vec, r, cg)


u_p.def_impl(_u_impl)
mlir.register_lowering(u_p, mlir.lower_fun(_u_impl, False))


@u_p.def_abstract_eval
def _u_p_abstract_eval(a, cg, r):
  ndim = a.shape[1]
  if r.dtype == jnp.float32:
    dtype = jnp.complex64
  elif r.dtype == jnp.float64:
    dtype = jnp.complex128
  return core.ShapedArray(cg.shape[:-ndim] + r.shape[:-1], dtype=dtype)


def _u_jvp_rule(primals, tangents):
  a, cg, r = primals
  a_dot, cg_dot, r_dot = tangents
  ndim = a.shape[1]
  g_vec = g_vectors(a, cg.shape[-ndim:])

  jvps = []
  if not isinstance(a_dot, ad.Zero):
    gv = lambda a: g_vectors(a, cg.shape[-ndim:])
    g_vec_dot = jax.jvp(gv, (a,), (a_dot,))[1]
    cgigdot = jnp.expand_dims(cg,
                              -ndim - 1) * 1.j * jnp.moveaxis(g_vec_dot, -1, 0)
    jvp1 = jnp.sum(u(a, cgigdot, r) * jnp.moveaxis(r, -1, 0), axis=-r.ndim)
    jvps.append(jvp1)
  if not isinstance(cg_dot, ad.Zero):
    jvp2 = u(a, cg_dot, r)
    jvps.append(jvp2)
  if not isinstance(r_dot, ad.Zero):
    cgig = jnp.expand_dims(cg, -ndim - 1) * 1.j * jnp.moveaxis(g_vec, -1, 0)
    jvp3 = jnp.sum(u(a, cgig, r) * jnp.moveaxis(r_dot, -1, 0), axis=-r.ndim)
    jvps.append(jvp3)

  return u(a, cg, r), sum(jvps)


def _u_transpose_rule(cotangent, a, cg, r):
  ndim = a.shape[1]
  if ad.is_undefined_primal(a) or ad.is_undefined_primal(r):
    raise NotImplementedError
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
    if np.prod(r.shape) == np.prod(r_vec.shape):
      pred = jnp.array_equal(r.flatten(), r_vec.flatten())
      wrt_cg = jax.lax.cond(pred, _transpose_u_fft, _transpose_u_map, cotangent)
    else:
      wrt_cg = _transpose_u(cotangent)
    return (None, wrt_cg, None)


def _u_batch_rule(batched_args, batch_dims):
  a, cg, r = batched_args
  ndim = a.shape[-1]
  bd_a, bd_cg, bd_r = batch_dims
  if batch_dims == (None, 0, None):
    bdim = 0
  elif batch_dims == (None, None, 0):
    bdim = cg.ndim - ndim
  else:
    raise NotImplementedError
  return u(a, cg, r), bdim


jax.interpreters.ad.primitive_jvps[u_p] = _u_jvp_rule
jax.interpreters.ad.primitive_transposes[u_p] = _u_transpose_rule
jax.interpreters.batching.primitive_batchers[u_p] = _u_batch_rule


def bloch_wave(a, cg, k_vec):
  r"""Return a wave function that takes a coordinate `r` as input,
  that generates the wave function value at `r` for all `k`.

  .. math::
    \psi_{ik}(r) = \exp(\mathrm{i}kr) \sum_{g} c_{ikg} \exp(\mathrm{i}gr)

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
  g_vec = g_vectors(a, grid_sizes)
  # r_vec is used to check whether fft should be used.
  r_vec = r_vectors(a, grid_sizes)

  def wave(r):
    """
    Args:
      r: a coordinate in the real space.
        a vector of shape `(d,)`
    Returns:
      a complex tensor that is the wave function value at `r`.
      the shape of the tensor is `(batch..., nk)`
    """
    sum_cg_expigr = u(a, cg, r)
    kr = jnp.tensordot(k_vec, r, axes=(-1, -1))
    expikr = jnp.exp(1.j * kr)
    return sum_cg_expigr * expikr

  return wave


