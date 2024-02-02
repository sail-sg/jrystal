"""Customized fft operations that better supports single program, multiple data
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
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jrystal._src.sharding import custom_sharding_by_batches
from jax import core
from jax.interpreters import mlir


# TODO: support differentiation rule
@custom_sharding_by_batches
def _fftn(x):
  return jnp.fft.fftn(x, axes=range(-3, 0))


def ifftn(x):
  return _ifftn_p.bind(x)


@custom_sharding_by_batches
def _ifftn_impl(x):
  return jnp.fft.ifftn(x, axes=range(-3, 0))


_ifftn_p = core.Primitive("ifftn_sharding")
_ifftn_p.def_impl(_ifftn_impl)
mlir.register_lowering(_ifftn_p, mlir.lower_fun(_ifftn_impl, False))


def _ifftn_jvp(primals, tangents):
  return ifftn(*primals), ifftn(*tangents)


def _ifftn_tranpose_rule(cotangent, x):
  return ifftn(cotangent),


@_ifftn_p.def_abstract_eval
def _ifftn_abstract_eval(x):
  return core.ShapedArray(x.shape, dtype=x.dtype)


jax.interpreters.ad.primitive_jvps[_ifftn_p] = _ifftn_jvp
# jax.interpreters.ad.primitive_transposes[_ifftn_p] = _ifftn_tranpose_rule
jax._src.interpreters.ad.deflinear2(_ifftn_p, _ifftn_tranpose_rule)


if __name__ == '__main__':
  # On CPU devices please enable:
  # import os
  # os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

  import numpy as np
  from jax.sharding import Mesh
  from jax.experimental.pjit import pjit
  from functools import partial

  num_gpus = len(jax.local_devices())
  mesh = Mesh(np.array(jax.devices()).reshape([1, 1, -1]), ('s', 'k', 'i'))
  spec = P('s', 'k', 'i')
  named_sharding = NamedSharding(mesh, spec)

  with mesh:
    key = jax.random.PRNGKey(123)

    shape = [2, 1, 4, 8, 8, 8]

    x = jax.random.normal(key, shape, dtype=jnp.complex64)
    print("input shape: ", x.addressable_shards[-1].data.shape)

    jitted_ifftn = pjit(
      ifftn, in_shardings=named_sharding
    )
    print(
      "customized ifftn output shape on single device: ",
      jitted_ifftn(x).addressable_shards[-1].data.shape
    )

    jitted_jax_ifftn = pjit(
      partial(jnp.fft.ifftn, axes=range(-3, 0)), in_shardings=named_sharding
    )

    print(
      "jax numpy ifftn output shape on single device: ",
      jitted_jax_ifftn(x).addressable_shards[-1].data.shape
    )

    def loss1(x):
      return jnp.sum(jnp.abs(ifftn(x)))

    grad1 = pjit(jax.grad(loss1), in_shardings=named_sharding)(x)

    def loss2(x):
      return jnp.sum(jnp.abs(jnp.fft.ifftn(x, axes=range(-3, 0))))

    grad2 = pjit(jax.grad(loss2), in_shardings=named_sharding)(x)

    print("gradients of both method are same: ", np.array_equal(grad1, grad2))

    print(
      "customized ifftn gradinet shape on single device: ",
      grad1.addressable_shards[-1].data.shape
    )

    print(
      "jax numpy ifftn gradinet shape on single device: ",
      grad2.addressable_shards[-1].data.shape
    )
