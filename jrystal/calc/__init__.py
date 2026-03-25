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
"""Public calculation API.

Usage::

    import jrystal as jr

    config = jr.config.get_config("config.yaml")
    result = jr.calc.energy(config)
    bands  = jr.calc.band(config, result)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .backend import get_backend
from .opt_utils import set_env_params
from .runtime import build_runtime_context
from .solver_direct_opt import run_direct_opt
from .solver_nscf import run_nscf
from .solver_scf import run_scf
from .types import BandStructureResult, GroundStateResult

if TYPE_CHECKING:
  from ..config import JrystalConfigDict


def energy(config: JrystalConfigDict) -> GroundStateResult:
  """Run a ground-state energy calculation.

  Automatically selects the electronic backend (all-electron or
  norm-conserving) and solver (``direct_opt`` or ``scf``) based on
  *config*.

  Args:
    config: Jrystal configuration dictionary.

  Returns:
    Converged ground-state result.
  """
  set_env_params(config)
  backend = get_backend(config)
  ctx = build_runtime_context(config, mode="mesh", backend=backend)

  solver_type = config.solver.type
  if solver_type == "scf":
    return run_scf(config, ctx, backend)
  elif solver_type in ("direct_opt", "direct"):
    return run_direct_opt(config, ctx, backend)
  else:
    raise ValueError(
      f"Unknown solver type '{solver_type}'. "
      "Use 'direct_opt' or 'scf'."
    )


def band(
  config: JrystalConfigDict,
  ground_state_result: Optional[GroundStateResult] = None,
) -> BandStructureResult:
  """Run a band-structure calculation.

  If *ground_state_result* is not provided, a ground-state
  calculation is run first via :func:`energy`.

  Args:
    config: Jrystal configuration dictionary.
    ground_state_result: Optional pre-computed ground-state result.

  Returns:
    Band-structure result with eigenvalues along the k-path.
  """
  set_env_params(config)

  if ground_state_result is None:
    ground_state_result = energy(config)

  backend = get_backend(config)
  ctx = build_runtime_context(config, mode="path", backend=backend)
  return run_nscf(config, ctx, backend, ground_state_result)


__all__ = [
  "band",
  "energy",
  "BandStructureResult",
  "GroundStateResult",
]
