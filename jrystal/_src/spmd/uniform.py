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
  # assert num_devices == jax.device_count(), f"Number of devices {num_devices} does not match the number of devices in the sharding {jax.device_count()}"

  single_device_arrays_shape = list(shape)
  for i in range(sharding_dim):
    single_device_arrays_shape[i] //= mesh_shape[i]

  output = []
  for i in range(num_devices):
    key, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, single_device_arrays_shape)
    output.append(jax.device_put(u, jax.devices()[i]))

  return jax.make_array_from_single_device_arrays(shape, out_sharding, output)


if __name__ == "__main__":
  from functools import partial

  import jax
  import jax.numpy as jnp
  import numpy as np
  from jax.experimental.pjit import pjit
  from jax.sharding import Mesh, NamedSharding
  from jax.sharding import PartitionSpec as P

  num_gpus = len(jax.local_devices())
  mesh = Mesh(np.array(jax.devices()).reshape([1, 1, -1]), ('s', 'k', 'i'))
  spec = P('s', 'k', 'i')
  named_sharding = NamedSharding(mesh, spec)

  with mesh:
    uniform(jax.random.PRNGKey(123), (1, 1, 16, 123), named_sharding)
