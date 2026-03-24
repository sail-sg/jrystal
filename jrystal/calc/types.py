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
"""Shared calc-layer dataclasses and protocols."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional, Protocol

from chex import dataclass as chex_dataclass
from jaxtyping import Array, Float

if TYPE_CHECKING:
  from .runtime import RuntimeContext


@chex_dataclass
class KSampling:
  """Unified k-point sampling container for mesh and path workflows."""

  mode: Literal["mesh", "path"]
  kpts: Float[Array, "kpt 3"]
  weights: Float[Array, " kpt"]
  labels: Optional[list[str]] = None
  segments: Optional[list[tuple[int, int]]] = None


@dataclass
class EnergyDecomposition:
  """Decomposed electronic-energy terms."""

  kinetic: float = 0.0
  hartree: float = 0.0
  xc: float = 0.0
  external: float = 0.0
  external_local: float = 0.0
  external_nonlocal: float = 0.0
  ewald: float = 0.0


@dataclass
class GroundStateResult:
  """Unified ground-state workflow result."""

  config: Any
  crystal: Any
  params_pw: dict
  params_occ: dict
  total_energy: float
  energy_terms: EnergyDecomposition
  converged: bool
  density: Any
  total_energy_history: list[float] = field(default_factory=list)


@dataclass
class BandStructureResult:
  """Unified band-structure workflow result."""

  config: Any
  crystal: Any
  kpath: KSampling
  eigenvalues: Any
  ground_state_energy: float = 0.0


class ElectronicBackend(Protocol):
  """Minimal backend protocol for backend-specific runtime operations."""

  def build_potentials(self, ctx: "RuntimeContext") -> "RuntimeContext":
    """Compute backend-specific potentials and attach them to the context."""

  def total_energy(self, params: Any, ctx: "RuntimeContext") -> float:
    """Compute total energy from parameters and runtime context."""


__all__ = [
  "BandStructureResult",
  "ElectronicBackend",
  "EnergyDecomposition",
  "GroundStateResult",
  "KSampling",
]
