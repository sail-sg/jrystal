"""Customized operations that better supports single program, multiple data
(spmd) computation.

See:
https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html
"""

from typing import Callable

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .custom_partitioning import jrystal_custom_partitioning


def _supported_sharding(sharding, shape):
  rank = len(shape.shape)
  max_shared_dims = min(len(sharding.spec), rank - 1)
  names = tuple(sharding.spec[:max_shared_dims]
               ) + tuple(None for _ in range(rank - max_shared_dims))
  return NamedSharding(sharding.mesh, P(*names))


def _infer_sharding_from_operands(mesh, arg_shapes, result_shape):
  arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
  return _supported_sharding(arg_shardings[0], arg_shapes[0])


def custom_sharding_by_mesh(fun: Callable[..., jax.Array]) -> Callable:
  """Inserts a CustomCallOp into the XLA graph with custom SPMD lowering rules
  such that the transformed function can be sharded over the first batched
  dimensions.

  Args:
      fun (Callable[..., jax.Array]): a function that takes arguments have
        batched dimensions.

  Returns:
      Callable:
  """

  def _partition(mesh, arg_shapes, result_shape):
    result_shardings = jax.tree_map(lambda x: x.sharding, result_shape)
    arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
    # For FFT operations, we want to maintain the same sharding pattern
    # for both input and output
    sharding = _supported_sharding(arg_shardings[0], arg_shapes[0])
    return mesh, fun, sharding, (sharding,)

  sharded_fun = jrystal_custom_partitioning(fun)
  sharded_fun.def_partition(
    infer_sharding_from_operands=_infer_sharding_from_operands,
    partition=_partition,
  )

  return sharded_fun
