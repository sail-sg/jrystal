"""Customized fft operation that better supports single program, multiple data
(spmd) computation.

The default jax.fft.fft function computes the discrete Fourier transform of an
N-dimensional input along the last dimension, and is batched along the first
N-1 dimensions. By default, however, it will ignore the sharding of the input
and gather the input on all devices. We need to utilize the module
jax.experimental.custom_partitioning to insert a CustomCallOp into the XLA
graph with custom SPMD lowering rults.

See:
https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html

Usage:

"""

import jax
from jax.sharding import NamedSharding
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import PartitionSpec as P
from jax.numpy.fft import fftn


# For an N-D input, keeps sharding along the first N-1 dimensions
# but replicate along the last dimension
def _supported_sharding(sharding, shape):
  rank = len(shape.shape)
  max_shared_dims = min(len(sharding.spec), rank - 1)
  names = tuple(sharding.spec[:max_shared_dims]
               ) + tuple(None for _ in range(rank - max_shared_dims))
  return NamedSharding(sharding.mesh, P(*names))


def _partition(mesh, arg_shapes, result_shape):
  arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
  return mesh, _fftn, _supported_sharding(
    arg_shardings[0], arg_shapes[0]
    ), (_supported_sharding(arg_shardings[0], arg_shapes[0]),)


def _infer_sharding_from_operands(mesh, arg_shapes, result_shape):
  arg_shardings = jax.tree_map(lambda x: x.sharding, arg_shapes)
  return _supported_sharding(arg_shardings[0], arg_shapes[0])


def _fftn(x):
  return fftn(x, axes=range(-3, 0))


@custom_partitioning
def sharded_fftn(x):
  return _fftn(x)


sharded_fftn.def_partition(
  infer_sharding_from_operands=_infer_sharding_from_operands,
  partition=_partition
)

if __name__ == '__main__':
  import numpy as np
  from jax.sharding import Mesh
  from jax.experimental.pjit import pjit

  num_gpus = len(jax.local_devices())
  mesh = Mesh(np.array(jax.devices()).reshape([1, 1, -1]), ('s', 'k', 'i'))
  spec = P('i')
  named_sharding = NamedSharding(mesh, spec)

  with mesh:
    key = jax.random.PRNGKey(123)
    grid = 16
    ns = 2
    nk = 3
    ni = 100

    shape = [ns, nk, ni, grid, grid, grid]

    x = jax.random.normal(key, shape)
    y = pjit(lambda x: x, in_shardings=None, out_shardings=named_sharding)(x)
    y = jax.device_put(y, named_sharding)
    print(y.addressable_shards[-1].data.shape)
    pjit_my_fft = pjit(
      sharded_fftn, in_shardings=named_sharding, out_shardings=named_sharding
    )
    print(pjit_my_fft(y).shape)
