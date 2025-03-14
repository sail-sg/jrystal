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
"""Kinetic operator."""
import jax.numpy as jnp
from jaxtyping import Float, Array

import einops


def kinetic_operator(
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"] = None,
) -> Float[Array, "kpt x y z"]:
  r"""Compute kinetic operator matrix element in reciprocal space.
  
  This function returns the following expression:
  
  .. math::
  
    \Vert \mathbf{G} + \mathbf{k} \Vert^2 / 2,
  
  the kinetic energy matrix element evaluated at each k-point and reciprocal lattice vector which will be contracted with the coefficient matrix to obtain the kinetic energy.
    
  Args:
      g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
      kpts (Float[Array, 'num_k 3'], optional): k-points. Defaults to None.

  Returns:
      Float[Array, 'kpt x y z']: kinetic energy matrix element evaluated at
      each k-point and reciprocal lattice vector. 
  """
  kpts = jnp.zeros([1, 3]) if kpts is None else kpts
  kpts = einops.rearrange(kpts, "kpt d -> kpt 1 1 1 d")
  return jnp.sum((g_vector_grid + kpts)**2, axis=-1) / 2
