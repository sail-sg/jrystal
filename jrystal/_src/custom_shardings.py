"""Customized operations that better supports single program, multiple data
(spmd) computation.

See:
https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import PartitionSpec as P
from typing import Callable


def _supported_sharding(sharding, shape):
  rank = len(shape.shape)
  max_shared_dims = min(len(sharding.spec), rank - 1)
  names = tuple(sharding.spec[:max_shared_dims]
               ) + tuple(None for _ in range(rank - max_shared_dims))
  return NamedSharding(sharding.mesh, P(*names))


def _infer_sharding_from_operands(mesh, arg_shapes, result_shape):
  arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
  return _supported_sharding(arg_shardings[0], arg_shapes[0])


def custom_sharding_by_batches(fun: Callable[..., jax.Array]) -> Callable:

  def _partition(mesh, arg_shapes, result_shape):
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    return mesh, fun, _supported_sharding(
      arg_shardings[0], arg_shapes[0]
      ), (_supported_sharding(arg_shardings[0], arg_shapes[0]),)

  sharded_fun = custom_partitioning(fun)
  sharded_fun.def_partition(
    infer_sharding_from_operands=_infer_sharding_from_operands,
    partition=_partition
  )

  return sharded_fun


@custom_sharding_by_batches
def fftn(x):
  return jnp.fft.fftn(x, axes=range(-3, 0))


@custom_sharding_by_batches
def qr(x):
  return jnp.linalg.qr(x)[0]
