import jax
import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental.pjit import pjit
from functools import partial
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from .fft import ifftn

num_gpus = len(jax.local_devices())
mesh = Mesh(np.array(jax.devices()).reshape([1, 1, -1]), ('s', 'k', 'i'))
spec = P('s', 'k', 'i')
named_sharding = NamedSharding(mesh, spec)

# On CPU devices please enable:
# import os
# os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=4'

with mesh:
  key = jax.random.PRNGKey(123)

  shape = [2, 1, 4, 8, 8, 8]

  x = jax.random.normal(key, shape, dtype=jnp.complex64)
  print("input shape: ", x.addressable_shards[-1].data.shape)

  jitted_ifftn = pjit(ifftn, in_shardings=named_sharding)
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
