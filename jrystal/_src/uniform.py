from typing import Optional

import jax
import numpy as np
from jax.sharding import NamedSharding
from jaxtyping import Array


def uniform(
  key: Array,
  shape: tuple,
  out_sharding: Optional[NamedSharding] = None
) -> Array:
  if out_sharding is None:
    return jax.random.uniform(key, shape)

  mesh_shape = list(out_sharding.mesh.shape.values())
  sharding_dim = len(mesh_shape)
  num_devices = np.prod(mesh_shape)

  single_device_arrays_shape = list(shape)
  for i in range(sharding_dim):
    single_device_arrays_shape[i] //= mesh_shape[i]

  output = []
  for i in range(num_devices):
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, single_device_arrays_shape)
    output.append(jax.device_put(u, jax.devices()[i]))

  return jax.make_array_from_single_device_arrays(shape, out_sharding, output)
