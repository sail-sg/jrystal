# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kinetic-energy operator utilities."""
import einops
import jax.numpy as jnp
from jaxtyping import Array, Float


def kinetic_operator(
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"] = None,
) -> Float[Array, "kpt x y z"]:
  r"""Evaluate :math:`\|\mathbf{G}+\mathbf{k}\|^2 / 2` on the reciprocal grid.

  Args:
    g_vector_grid (Float[Array, "x y z 3"]): Reciprocal-space G-vector grid.
    kpts (Float[Array, "kpt 3"], optional): :math:`k`-point coordinates. If
      ``None``, ``[0, 0, 0]`` is used.

  Returns:
    Float[Array, "kpt x y z"]: Kinetic operator values for each :math:`k` point
    and grid point.
  """
  kpts = jnp.zeros([1, 3]) if kpts is None else kpts
  kpts = einops.rearrange(kpts, "kpt d -> kpt 1 1 1 d")

  return jnp.sum((g_vector_grid + kpts)**2, axis=-1) / 2
