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
from typing import List, Optional

import numpy as np
from jaxtyping import Array, Float, Int

# from interpax import CubicSpline
from scipy.interpolate import CubicSpline

from ..sbt import batch_sbt, sbt_numerical


def _beta_sbt_single_atom(
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
      Float[Array, "kpt beta x y z"]: the beta functions in reciprocal space.
  """
  assert len(nonlocal_angular_momentum) == nonlocal_beta_grid.shape[0]
  assert r_grid.shape[0] == nonlocal_beta_grid.shape[1]
  if r_grid[0] == 0:
    r_grid = r_grid[1:]
    nonlocal_beta_grid = nonlocal_beta_grid[:, 1:]

  nonlocal_angular_momentum = list(nonlocal_angular_momentum)

  if kpts is not None:
    gk_vector_grid = np.expand_dims(
      kpts, axis=(1, 2, 3)
    ) + np.expand_dims(g_vector_grid, 0)  # [nk x y z 3]
  else:
    gk_vector_grid = np.expand_dims(g_vector_grid, 0)  # [1 x y z 3]

  radius = np.sqrt(np.sum(gk_vector_grid**2, axis=-1))
  k, beta_k = sbt_numerical(
    r_grid, nonlocal_beta_grid, l=nonlocal_angular_momentum,
    kmax=np.max(radius)
  )

  beta_sbt = CubicSpline(k, beta_k, axis=1)(radius)
  beta_sbt = np.swapaxes(beta_sbt, 0, 1)
  return beta_sbt


def beta_sbt_grid(
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Optional[Float[Array, "kpt 3"]] = None
) -> List[Float[Array, "kpt beta x y z"]]:
  """
  Calculate the spherical bessel transform of the beta functions for multiple
  atoms.

  .. math::

    \beta_l(G) = \int_0^\infty  \beta(r) j_l(Gr) r^2 dr

  Return the beta function value of angular momentum values :math:`l` at the
  reciprocal vectors :math:`G` per atom

  Args:
    r_grid (List[Float[Array, "r"]]): the r grid corresponding to the beta
      functions.
    nonlocal_beta_grid (List[Float[Array, "beta r"]]): beta values.
    nonlocal_angular_momentum (List[List[int]]): angular momentum corresponding
    to the beta functions.
    g_vector_grid (Float[Array, "x y z 3"]): reciprocal vectors to interpolate.
    kpts (Optional[Float[Array, "kpt 3"]]): k-points. Default is None.

  Returns:
    List[Float[Array, "kpt beta x y z"]]: A List of jax.array of the beta
    functions evaluated in reciprocal space grid.

  """
  # TODO: parallelize the calculation.
  output = []
  for r, b, l in zip(r_grid, nonlocal_beta_grid, nonlocal_angular_momentum):
    output.append(_beta_sbt_single_atom(r, b, l, g_vector_grid, kpts))
  # output = np.concatenate(output, axis=1)
  return output
