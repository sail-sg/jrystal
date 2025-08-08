"""Customized FFT operations that better supports single program, multiple data
(SPMD) computation.

The default jax.fft.fftn function computes the discrete Fourier transform of an
(n+m)-dimensional input along the last n dimensions, and is batched along the
first m dimensions. By default, however, it will ignore the sharding of the
input and gather the input on all devices. We need to utilize the module jax.
experimental.custom_partitioning to insert a CustomCallOp into the XLA graph
with custom SPMD lowering results.

Example:
  >>> import jax.numpy as jnp
  >>> from jrystal.spmd.fft import fftn3d, ifftn3d
  >>>
  >>> # Create a 3D array
  >>> x = jnp.random.randn(8, 8, 8)
  >>>
  >>> # Apply 3D FFT
  >>> y = fftn3d(x)
  >>>
  >>> # Apply inverse 3D FFT
  >>> z = ifftn3d(y)
"""

from typing import Any, Tuple

import jax
import jax.interpreters
import jax.numpy as jnp
from jax import core
from jax._src.interpreters import batching
from jax.interpreters import mlir

from .custom_sharding import custom_sharding_by_mesh


def fftn3d(x: jnp.ndarray) -> jnp.ndarray:
  """Compute the 3D FFT with proper SPMD support.

    Args:
      x: Input array of shape (..., x, y, z) where the last 3 dimensions
      will be transformed.

    Returns:
      Array of same shape as input containing the 3D FFT.
    """
  if x.ndim < 3:
    raise ValueError(f"Input must have at least 3 dimensions, got {x.ndim}")
  return _fftn_p.bind(x)


def ifftn3d(x: jnp.ndarray) -> jnp.ndarray:
  """Compute the inverse 3D FFT with proper SPMD support.

    Args:
        x: Input array of shape (..., N, M, P) where the last 3 dimensions
           will be transformed.

    Returns:
        Array of same shape as input containing the inverse 3D FFT.
    """
  if x.ndim < 3:
    raise ValueError(f"Input must have at least 3 dimensions, got {x.ndim}")
  return _ifftn_p.bind(x)


# Core implementation functions
@custom_sharding_by_mesh
def _ifftn_impl(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.fft.ifftn(x, axes=range(-3, 0))


@custom_sharding_by_mesh
def _fftn_impl(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.fft.fftn(x, axes=range(-3, 0))


# Primitive definitions
_ifftn_p = core.Primitive("ifftn_sharding")
_ifftn_p.def_impl(_ifftn_impl)
mlir.register_lowering(_ifftn_p, mlir.lower_fun(_ifftn_impl, False))

_fftn_p = core.Primitive("fftn_sharding")
_fftn_p.def_impl(_fftn_impl)
mlir.register_lowering(_fftn_p, mlir.lower_fun(_fftn_impl, False))


def _ifftn_jvp(primals, tangents):
  return ifftn3d(*primals), ifftn3d(*tangents)


def _ifftn_tranpose_rule(cotangent, x):
  return ifftn3d(cotangent),


def _fftn_jvp(primals, tangents):
  return fftn3d(*primals), fftn3d(*tangents)


def _fftn_tranpose_rule(cotangent, x):
  return fftn3d(cotangent),


def _ifftn_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return ifftn3d(x), 0


def _fftn_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return fftn3d(x), 0


@_ifftn_p.def_abstract_eval
def _ifftn_abstract_eval(x):
  return core.ShapedArray(x.shape, dtype=x.dtype)


@_fftn_p.def_abstract_eval
def _fftn_abstract_eval(x):
  return core.ShapedArray(x.shape, dtype=x.dtype)


jax._src.interpreters.ad.primitive_jvps[_ifftn_p] = _ifftn_jvp
jax._src.interpreters.ad.deflinear2(_ifftn_p, _ifftn_tranpose_rule)
jax._src.interpreters.batching.primitive_batchers[_ifftn_p
                                                 ] = _ifftn_batching_rule
jax._src.interpreters.ad.primitive_jvps[_fftn_p] = _fftn_jvp
jax._src.interpreters.ad.deflinear2(_fftn_p, _fftn_tranpose_rule)
jax._src.interpreters.batching.primitive_batchers[_fftn_p] = _fftn_batching_rule
