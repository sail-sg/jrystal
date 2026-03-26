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


# ---------------------------------------------------------------------------
# Basis / grid objects
# ---------------------------------------------------------------------------

@chex_dataclass
class KSampling:
  """Unified k-point sampling container for mesh and path workflows."""

  mode: Literal["mesh", "path"]
  kpts: Float[Array, "kpt 3"]
  weights: Float[Array, " kpt"]
  labels: Optional[list[str]] = None
  segments: Optional[list[tuple[int, int]]] = None


@dataclass
class PlaneWaveBasis:
  """Plane-wave basis descriptor."""

  freq_mask: Any  # Bool[Array, "x y z"]
  grid_sizes: tuple[int, ...]
  num_g: int  # number of True entries in freq_mask


@dataclass
class ExecutionPlan:
  """Device and parallelism strategy for a calculation."""

  num_devices: int = 1
  parallel_over_k: bool = False
  # Reserved for future multi-GPU / multi-host:
  # parallel_axes: tuple[str, ...] = ("k",)
  # replicate_fft_axes: bool = True
  # multi_host: bool = False


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------

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
  eigenvalues: Optional[Any] = None
  total_energy_history: list[float] = field(default_factory=list)


@dataclass
class BandStructureResult:
  """Unified band-structure workflow result."""

  config: Any
  crystal: Any
  kpath: KSampling
  eigenvalues: Any
  ground_state_energy: float = 0.0


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

class ElectronicBackend(Protocol):
  """Backend protocol abstracting AE / NC / USPP physics differences.

  Workflow solvers call these methods without knowing which backend
  (all-electron, norm-conserving, ultrasoft) is in use.
  """

  def build_potentials(self, ctx: "RuntimeContext") -> "RuntimeContext":
    """Compute backend-specific potentials and attach them to *ctx*."""
    ...

  def total_energy(
    self, coeff: Any, occ: Any, ctx: "RuntimeContext",
  ) -> float:
    """Compute electronic total energy (excluding Ewald) from
    plane-wave coefficients and occupation numbers."""
    ...

  def hamiltonian_apply(
    self, coeff: Any, density: Any, ctx: "RuntimeContext",
  ) -> Any:
    """Apply H to wavefunctions (H|psi>). Used by SCF eigensolver."""
    ...


__all__ = [
  "BandStructureResult",
  "ElectronicBackend",
  "EnergyDecomposition",
  "ExecutionPlan",
  "GroundStateResult",
  "KSampling",
  "PlaneWaveBasis",
]
