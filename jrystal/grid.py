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
"""Grid operations for crystalline systems.

This module provides functions for working with real and reciprocal space grids in crystalline systems.
It includes utilities for:

- Generating G-vectors and R-vectors
- :math:`k`-point sampling for Brillouin zone integration
- Frequency space operations and masks
- Grid transformations between real and reciprocal space
"""

from ._src.grid import (
  g_vectors,
  r_vectors,
  k_vectors,
  spherical_mask,
  cubic_mask,
  proper_grid_size,
  translation_vectors,
  estimate_max_cutoff_energy,
  grid_vector_radius,
  g2cell_vectors,
  r2cell_vectors,
  r2g_vector_grid,
  g2r_vector_grid,
)

__all__ = [
  "g_vectors",
  "r_vectors",
  "k_vectors",
  "spherical_mask",
  "cubic_mask",
  "proper_grid_size",
  "translation_vectors",
  "estimate_max_cutoff_energy",
  "grid_vector_radius",
  "g2cell_vectors",
  "r2cell_vectors",
  "r2g_vector_grid",
  "g2r_vector_grid",
]
