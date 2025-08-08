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
"""Norm Conserving Pseudopotential for Plane Waves. """
from .local import (
  hamiltonian_local, energy_local, potential_local_reciprocal
)
from .beta import beta_sbt_grid
from .nloc import (
  potential_nonlocal_psi_reciprocal,
  hamiltonian_nonlocal,
  hamiltonian_matrix,
  energy_nonlocal,
  hamiltonian_trace
)

__call__ = [
  beta_sbt_grid,
  potential_local_reciprocal,
  hamiltonian_local,
  energy_local,
  potential_nonlocal_psi_reciprocal,
  hamiltonian_nonlocal,
  hamiltonian_matrix,
  energy_nonlocal,
  hamiltonian_trace
]
