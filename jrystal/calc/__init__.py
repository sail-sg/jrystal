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

from copy import deepcopy
from typing import TYPE_CHECKING, Optional

from absl import logging

from .backend import get_backend
from .opt_utils import set_env_params
from .runtime import build_runtime_context
from .solver_direct_opt import run_direct_opt
from .solver_nscf import run_nscf
from .solver_scf import run_scf
from .types import BandStructureResult, GroundStateResult

if TYPE_CHECKING:
  from ..config import JrystalConfigDict


def _run_with_mode(
  config: JrystalConfigDict,
  backend,
  ctx,
  mode: str,
) -> GroundStateResult:
  if mode == "scf":
    return run_scf(config, ctx, backend)
  if mode == "direct_opt":
    return run_direct_opt(config, ctx, backend)
  raise ValueError(
    f"Unknown solver mode '{mode}'. Use 'auto', 'direct_opt', or 'scf'."
  )


def energy(config: JrystalConfigDict) -> GroundStateResult:
  """Run a ground-state energy calculation.

  Automatically selects the electronic backend (all-electron or
  norm-conserving) and solver (``auto``, ``direct_opt``, or ``scf``) based on
  *config*.

  Args:
    config: Jrystal configuration dictionary.

  Returns:
    Converged ground-state result.
  """
  set_env_params(config)
  backend = get_backend(config)
  ctx = build_runtime_context(config, mode="mesh", backend=backend)

  solver_mode = config.solver.mode
  if solver_mode in ("scf", "direct_opt"):
    return _run_with_mode(config, backend, ctx, solver_mode)

  if solver_mode != "auto":
    raise ValueError(
      f"Unknown solver mode '{solver_mode}'. "
      "Use 'auto', 'direct_opt', or 'scf'."
    )

  primary = config.solver.auto.primary
  fallback = config.solver.auto.fallback
  fallback_on_nonconverged = config.solver.auto.fallback_on_nonconverged
  fallback_on_error = config.solver.auto.fallback_on_error

  try:
    result = _run_with_mode(config, backend, ctx, primary)
  except Exception as exc:
    if not fallback_on_error:
      raise
    logging.warning(
      "Primary solver '%s' failed with %s: %s. Falling back to '%s'.",
      primary,
      exc.__class__.__name__,
      exc,
      fallback,
    )
    fallback_config = deepcopy(config)
    fallback_config.solver.mode = fallback
    return _run_with_mode(fallback_config, backend, ctx, fallback)

  if result.converged or not fallback_on_nonconverged:
    return result

  logging.warning(
    "Primary solver '%s' did not converge. Falling back to '%s'.",
    primary,
    fallback,
  )
  fallback_config = deepcopy(config)
  fallback_config.solver.mode = fallback
  return _run_with_mode(fallback_config, backend, ctx, fallback)


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
