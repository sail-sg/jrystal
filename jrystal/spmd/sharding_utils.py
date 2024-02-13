"""Utility functions for sharding."""

import jax
from typing import Any, Union
from jax.sharding import Sharding


def tree_device_put(tree: Any, sharding: Union[Any, Sharding]) -> Any:
  """Transfer a pytree to device by a pytree of sharding.

  Args:
      tree (Any): a pytree to be transfored
      sharding (Any|Sharding): a pytree of ``Jax.Sharding`` objects. The
      sharding may have the same structure as tree, or have a subtree that
      only contains a subset of all the leaves.

  Returns:
      Any:  A new pytree with the same structure as ``tree`` but sharded
      and transformed to device
  """
  if isinstance(sharding, Sharding) and isinstance(tree, dict):
    for key, value in tree.items():
      if isinstance(value, dict):
        tree[key] = tree_device_put(value, sharding)
      else:
        tree[key] = jax.device_put(value, sharding)

  elif isinstance(tree, dict) and isinstance(sharding, dict):
    for key, s in sharding.items():
      if key in tree:
        if isinstance(tree[key], dict) and isinstance(sharding[key], dict):
          tree[key] = tree_device_put(tree[key], s)
        else:
          tree[key] = jax.device_put(tree[key], s)

      else:
        raise ValueError(f"sharding key {key} does not match any key in tree.")
  else:
    raise ValueError("sharding must a pytree of a jax sharding object")

  return tree


if __name__ == "__main__":
  import numpy as np
  import jax.numpy as jnp
  from jax.sharding import Mesh
  from jax.sharding import NamedSharding
  from jax.sharding import PartitionSpec as P
  from jax.tree_util import tree_map

  mesh = Mesh(np.array(jax.devices()).reshape([1, 1, -1]), ('s', 'k', 'i'))
  spec = P('s', 'k', 'i')
  named_sharding = NamedSharding(mesh, spec)

  sharding_tree = {"a1": {"a2": {"b3": named_sharding}}}
  print("mesh: ", mesh)
  print("sharding: ", named_sharding)

  with mesh:
    tree = {
      "a1":
        {
          "a2": {
            "a3": jnp.ones([64, 64, 64]), "b3": jnp.ones([64, 64, 64])
          },
          "b2": jnp.ones([64, 64, 64])
        }
    }

    tree1 = tree_device_put(tree, named_sharding)
    print(tree_map(lambda x: x.addressable_shards[0].data.shape, tree1))

    tree = {
      "a1":
        {
          "a2": {
            "a3": jnp.ones([64, 64, 64]), "b3": jnp.ones([64, 64, 64])
          },
          "b2": jnp.ones([64, 64, 64])
        }
    }

    tree2 = tree_device_put(tree, sharding_tree)
    print(tree_map(lambda x: x.addressable_shards[0].data.shape, tree2))
