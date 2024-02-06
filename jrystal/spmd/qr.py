"""Customized qr decomposition operation that better supports single program,
multiple data (spmd) computation.

See:
https://jax.readthedocs.io/en/latest/jax.experimental.custom_partitioning.html
"""

import jax
# from jax.sharding import NamedSharding
# from jax.experimental.custom_partitioning import custom_partitioning
# from jax.sharding import PartitionSpec as P
from .sharding import custom_sharding_by_batches


@custom_sharding_by_batches
def qr(x):
  return jax.numpy.linalg.qr(x)[0]
