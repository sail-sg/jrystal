"""Sharding-friendly FFT wrappers implemented with custom partitioning.

This module keeps the semantics of :func:`jax.numpy.fft.fftn` and
:func:`jax.numpy.fft.ifftn`, but constrains sharding so FFT axes are
replicated while non-FFT axes preserve input sharding.
"""

from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

AxisLike = Optional[Union[int, Sequence[int], Tuple[int, ...]]]
ShapeLike = Optional[Sequence[int]]


def _parse_callback_args(cb_args):
  """Parse callback args for both static and non-static callback signatures."""
  if len(cb_args) == 3:
    mesh, arg_shapes, result_shape = cb_args
    s = None
    axes = None
    norm = None
  elif len(cb_args) == 6:
    s, axes, norm, mesh, arg_shapes, result_shape = cb_args
  else:
    raise ValueError(
      "Unexpected callback arguments for custom_partitioning. "
      f"Got {len(cb_args)} arguments."
    )
  return s, axes, norm, mesh, arg_shapes, result_shape


def _normalize_axes(rank: int, axes: AxisLike, s: ShapeLike) -> Tuple[int, ...]:
  """Normalize FFT axes into sorted non-negative axis indices."""
  if axes is None:
    if s is None:
      axes = tuple(range(rank))
    else:
      if len(s) > rank:
        raise ValueError(
          "If axes is None, len(s) must be <= input rank. "
          f"Got len(s)={len(s)}, rank={rank}."
        )
      axes = tuple(range(rank - len(s), rank))
  elif isinstance(axes, int):
    axes = (axes,)
  else:
    axes = tuple(axes)

  normalized = []
  for ax in axes:
    _ax = ax + rank if ax < 0 else ax
    if _ax < 0 or _ax >= rank:
      raise ValueError(
        f"Invalid FFT axis {ax} for rank {rank}. "
        f"Normalized axis {_ax} is out of range."
      )
    normalized.append(_ax)

  if len(set(normalized)) != len(normalized):
    raise ValueError(f"FFT axes must be unique, got {axes}.")

  return tuple(sorted(normalized))


def _mask_fft_axes(sharding, rank: int, fft_axes: Tuple[int, ...]):
  """Force FFT axes to be replicated in a NamedSharding spec."""
  if not isinstance(sharding, NamedSharding):
    return sharding

  spec = list(sharding.spec)
  if len(spec) < rank:
    spec.extend([None] * (rank - len(spec)))
  else:
    spec = spec[:rank]

  for ax in fft_axes:
    spec[ax] = None
  return NamedSharding(sharding.mesh, P(*spec))


def _fft_sharding_rule(
  s: ShapeLike,
  axes: AxisLike,
  norm: Optional[str],
  mesh,
  value_types,
  result_types,
):
  """Build a dynamic shardy rule for rank/axes-dependent FFT replication."""
  del mesh, norm, result_types
  rank = len(value_types[0].shape)
  fft_axes = _normalize_axes(rank, axes, s)

  factors = tuple(f"d{i}" for i in range(rank))
  lhs = " ".join(factors)
  rhs = " ".join(factors)
  rule = f"{lhs} -> {rhs}"
  need_replication = tuple(factors[ax] for ax in fft_axes)

  return rule, {"need_replication_factors": need_replication}


def _fftn_semantic(
  x: jax.Array,
  s: ShapeLike = None,
  axes: AxisLike = None,
  norm: Optional[str] = None,
) -> jax.Array:
  """Compute an n-D FFT with custom sharding propagation."""
  return jnp.fft.fftn(x, s=s, axes=axes, norm=norm)


_fftn_partitioned = custom_partitioning(
  _fftn_semantic, static_argnums=(1, 2, 3)
)


def _fftn_infer_sharding_from_operands(*cb_args):
  s, axes, norm, mesh, arg_shapes, result_shape = _parse_callback_args(cb_args)
  del mesh, result_shape, norm
  in_sharding = arg_shapes[0].sharding
  rank = len(arg_shapes[0].shape)
  fft_axes = _normalize_axes(rank, axes, s)
  return _mask_fft_axes(in_sharding, rank, fft_axes)


def _fftn_partition(*cb_args):
  s, axes, norm, mesh, arg_shapes, result_shape = _parse_callback_args(cb_args)
  del result_shape
  in_sharding = arg_shapes[0].sharding
  rank = len(arg_shapes[0].shape)
  fft_axes = _normalize_axes(rank, axes, s)
  sharding = _mask_fft_axes(in_sharding, rank, fft_axes)

  def lower_fn(x):
    return jnp.fft.fftn(x, s=s, axes=axes, norm=norm)

  return mesh, lower_fn, sharding, (sharding,)


_fftn_partitioned.def_partition(
  partition=_fftn_partition,
  infer_sharding_from_operands=_fftn_infer_sharding_from_operands,
  sharding_rule=_fft_sharding_rule,
)


def _ifftn_semantic(
  x: jax.Array,
  s: ShapeLike = None,
  axes: AxisLike = None,
  norm: Optional[str] = None,
) -> jax.Array:
  """Compute an inverse n-D FFT with custom sharding propagation."""
  return jnp.fft.ifftn(x, s=s, axes=axes, norm=norm)


_ifftn_partitioned = custom_partitioning(
  _ifftn_semantic, static_argnums=(1, 2, 3)
)


def _ifftn_infer_sharding_from_operands(*cb_args):
  s, axes, norm, mesh, arg_shapes, result_shape = _parse_callback_args(cb_args)
  del mesh, result_shape, norm
  in_sharding = arg_shapes[0].sharding
  rank = len(arg_shapes[0].shape)
  fft_axes = _normalize_axes(rank, axes, s)
  return _mask_fft_axes(in_sharding, rank, fft_axes)


def _ifftn_partition(*cb_args):
  s, axes, norm, mesh, arg_shapes, result_shape = _parse_callback_args(cb_args)
  del result_shape
  in_sharding = arg_shapes[0].sharding
  rank = len(arg_shapes[0].shape)
  fft_axes = _normalize_axes(rank, axes, s)
  sharding = _mask_fft_axes(in_sharding, rank, fft_axes)

  def lower_fn(x):
    return jnp.fft.ifftn(x, s=s, axes=axes, norm=norm)

  return mesh, lower_fn, sharding, (sharding,)


_ifftn_partitioned.def_partition(
  partition=_ifftn_partition,
  infer_sharding_from_operands=_ifftn_infer_sharding_from_operands,
  sharding_rule=_fft_sharding_rule,
)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _fftn_with_jvp(
  a: jax.Array,
  s: ShapeLike = None,
  axes: AxisLike = None,
  norm: Optional[str] = None,
) -> jax.Array:
  return _fftn_partitioned(a, s=s, axes=axes, norm=norm)


@_fftn_with_jvp.defjvp
def _fftn_with_jvp_rule(
  s: ShapeLike, axes: AxisLike, norm: Optional[str], primals, tangents
):
  a, = primals
  ta, = tangents
  y = _fftn_partitioned(a, s=s, axes=axes, norm=norm)
  ty = jnp.fft.fftn(ta, s=s, axes=axes, norm=norm)
  return y, ty


def fftn(
  a: jax.Array,
  s: ShapeLike = None,
  axes: AxisLike = None,
  norm: Optional[str] = None,
) -> jax.Array:
  """Compute a multidimensional discrete Fourier transform along given axes.

  Signature intentionally mirrors :func:`jax.numpy.fft.fftn`.

  Args:
    a (jax.Array): The input array to transform.
    s (ShapeLike): The sharding specification for the input array.
    axes (AxisLike): The axes over which to perform the FFT.
    norm (Optional[str]): The normalization mode.

  Returns:
    jax.Array: An array containing the multidimensional discrete Fourier
      transform of ``a``.
  """
  return _fftn_with_jvp(a, s=s, axes=axes, norm=norm)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def _ifftn_with_jvp(
  a: jax.Array,
  s: ShapeLike = None,
  axes: AxisLike = None,
  norm: Optional[str] = None,
) -> jax.Array:
  return _ifftn_partitioned(a, s=s, axes=axes, norm=norm)


@_ifftn_with_jvp.defjvp
def _ifftn_with_jvp_rule(
  s: ShapeLike, axes: AxisLike, norm: Optional[str], primals, tangents
):
  a, = primals
  ta, = tangents
  y = _ifftn_partitioned(a, s=s, axes=axes, norm=norm)
  ty = jnp.fft.ifftn(ta, s=s, axes=axes, norm=norm)
  return y, ty


def ifftn(
  a: jax.Array,
  s: ShapeLike = None,
  axes: AxisLike = None,
  norm: Optional[str] = None,
) -> jax.Array:
  """Compute a multidimensional inverse discrete Fourier transform along given
  axes.

  Signature intentionally mirrors :func:`jax.numpy.fft.ifftn`.

  Args:
    a (jax.Array): The input array to transform.
    s (ShapeLike): The sharding specification for the input array.
    axes (AxisLike): The axes over which to perform the inverse FFT.
    norm (Optional[str]): The normalization mode.

  Returns:
    jax.Array: An array containing the multidimensional inverse discrete Fourier
      transform of ``a``.
  """
  return _ifftn_with_jvp(a, s=s, axes=axes, norm=norm)


__all__ = ["fftn", "ifftn"]
