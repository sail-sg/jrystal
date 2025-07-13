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
"""Functions for dealing with beta functions. """
import numpy as np
from typing import List, Optional
from jaxtyping import Float, Array, Int
from ..sbt import batched_sbt
from .interpolate import cubic_spline


def beta_sbt_grid_single_atom(
  r_grid: Float[Array, "r"],
  nonlocal_beta_grid: Float[Array, "beta r"],
  nonlocal_angular_momentum: Int[Array, "beta"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Optional[Float[Array, "kpt 3"]] = None
) -> Float[Array, "kpt beta x y z"]:
  """
  Calculate the spherical bessel transform of the beta functions for a single
  atom.

  .. math::

    \beta_l(G) = \int_0^\infty  \beta(r) j_l(Gr) r^2 dr

  Return the beta function value of angular momentum values :math:`l` at the
  reciprocal vectors :math:`G` per atom

  Args:
      r_grid (Float[Array, "r"]): the r grid corresponding to the beta
      functions.
      nonlocal_beta_grid (Float[Array, "beta r"]): beta values.
      nonlocal_angular_momentum (List[int]): angular momentum corresponding
      to the beta functions.
      g_vector_grid (Float[Array, "x y z 3"]): reciprocal vectors to
      interpolate.
      kpts (Optional[Float[Array, "kpt 3"]]): k-points. Default is None.

  Returns:
      Float[Array, "kptbeta x y z"]: the beta functions in reciprocal space.

  .. warning::
    Cubic spline interpolation is not implemented in JAX. This function uses ``NumPy`` and is not differentiable.

  """
  assert len(nonlocal_angular_momentum) == nonlocal_beta_grid.shape[0]
  assert r_grid.shape[0] == nonlocal_beta_grid.shape[1]

  nonlocal_angular_momentum = nonlocal_angular_momentum.tolist()
  if kpts is not None:
    gk_vector_grid = np.expand_dims(
      kpts, axis=(1, 2, 3)
    ) + np.expand_dims(g_vector_grid, 0)  # [nk x y z 3]
  else:
    gk_vector_grid = np.expand_dims(g_vector_grid, 0)  # [1 x y z 3]

  radius = np.sqrt(np.sum(gk_vector_grid**2, axis=-1))
  k, beta_k = batched_sbt(
    r_grid, nonlocal_beta_grid, l=nonlocal_angular_momentum,
    kmax=np.max(radius)
  )

  beta_sbt = cubic_spline(k, beta_k, radius)
  beta_sbt = np.swapaxes(beta_sbt, 0, 1)
  return beta_sbt


def beta_sbt_grid_multi_atoms(
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[Int[Array, "beta"]],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Optional[Float[Array, "kpt 3"]] = None
) -> Float[Array, "kpt beta x y z"]:
  """
  Calculate the spherical bessel transform of the beta functions for multiple atoms.

  .. math::

    \beta_l(G) = \int_0^\infty  \beta(r) j_l(Gr) r^2 dr

  Return the beta function value of angular momentum values :math:`l` at the reciprocal vectors :math:`G` per atom

  Args:
    r_grid (List[Float[Array, "r"]]): the r grid corresponding to the beta
      functions.
    nonlocal_beta_grid (List[Float[Array, "beta r"]]): beta values.
    nonlocal_angular_momentum (List[List[int]]): angular momentum corresponding to the beta functions.
    g_vector_grid (Float[Array, "x y z 3"]): reciprocal vectors to interpolate.
    kpts (Optional[Float[Array, "kpt 3"]]): k-points. Default is None.

  Returns:
    Float[Array, "kpt beta x y z"]: An jax.array of the beta functions evaluated in reciprocal space grid.

  """

  output = []
  for r, b, l in zip(r_grid, nonlocal_beta_grid, nonlocal_angular_momentum):
    output.append(beta_sbt_grid_single_atom(r, b, l, g_vector_grid, kpts))
  output = np.concatenate(output, axis=1)
  return output
