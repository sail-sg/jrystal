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
"""Utilities for Brillouin-zone :math:`k`-path generation.

This module wraps ASE band-path utilities.
"""

import numpy as np
from ase import cell
from jaxtyping import Array, Float

from jrystal._src.const import BOHR2ANGSTROM


def get_k_path(
  cell_vectors: Float[Array, 'd d'],
  path: str,
  num: int,
  fractional: bool = False
) -> np.array:
  """Return points along a high-symmetry :math:`k` path.

  Args:
    cell_vectors (Float[Array, 'd d']): Real-space cell vectors in Bohr.
    path (str): Path string of special points (ASE notation).
    num (int): Number of sampled points along the path.
    fractional (bool): If ``True``, return fractional coordinates. If
      ``False``, return absolute coordinates in reciprocal-space units
      (:math:`1 / \mathrm{Bohr}`).

  Returns:
    np.ndarray: Array of sampled :math:`k` points.
  """
  _cell = cell.Cell(cell_vectors * BOHR2ANGSTROM)
  kpts = _cell.bandpath(path, npoints=num).cartesian_kpts() * BOHR2ANGSTROM
  kpts = np.matmul(kpts, cell_vectors.T)

  if fractional:
    return kpts

  else:
    B = np.linalg.inv(cell_vectors).T * 2 * np.pi
    return kpts @ B
