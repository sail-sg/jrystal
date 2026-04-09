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
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import ase

from ..io import (
  load_checkpoint,
  save_band_structure,
  save_config_snapshot,
  save_ground_state,
  save_run_metadata,
  setup_output_dir,
)
from ..terminal_ui import (
  close_log,
  get_log_level,
  open_log,
  set_log_level,
  stage_warning,
)
from .backend import get_backend
from .opt_utils import create_crystal, set_env_params
from .runtime import build_runtime_context
from .solver_direct_opt import run_direct_opt
from .solver_nscf import run_nscf
from .solver_scf import run_scf
from .types import BandStructureResult, GroundStateResult
from .workflow_logging import log_system_info

if TYPE_CHECKING:
  from ..config import JrystalConfigDict


def _run_with_mode(
  config: JrystalConfigDict,
  backend,
  ctx,
  mode: str,
  *,
  requested_solver_mode: str,
  restart_state: Optional[dict] = None,
  output_dir: Optional[Path] = None,
) -> GroundStateResult:
  if mode == "scf":
    return run_scf(
      config,
      ctx,
      backend,
      requested_solver_mode=requested_solver_mode,
      restart_state=restart_state,
      output_dir=output_dir,
    )
  if mode == "direct_opt":
    return run_direct_opt(
      config,
      ctx,
      backend,
      requested_solver_mode=requested_solver_mode,
      restart_state=restart_state,
      output_dir=output_dir,
    )
  raise ValueError(
    f"Unknown solver mode '{mode}'. Use 'auto', 'direct_opt', or 'scf'."
  )


def _physical_state_from_result(result: GroundStateResult) -> Optional[dict]:
  if result.coefficients is None or result.occupations is None:
    return None
  if not isinstance(result.coefficients, dict):
    return None
  if "w_re" not in result.coefficients or "w_im" not in result.coefficients:
    return None
  return {
    "density": result.density,
    "coefficients":
      {
        "w_re": result.coefficients["w_re"],
        "w_im": result.coefficients["w_im"],
      },
    "occupations": result.occupations,
    "eigenvalues":
      (
        result.eigenvalues
        if result.eigenvalues is not None else result.occupations * 0.0
      ),
    "has_eigenvalues": result.eigenvalues is not None,
    "step": max(result.num_iterations - 1, 0),
    "total_energy": result.total_energy,
  }


def _run_ground_state(
  config: JrystalConfigDict,
  *,
  backend,
  ctx,
  output_dir: Path,
) -> GroundStateResult:
  requested_solver_mode = config.solver.mode
  restart_state = None
  if config.io.restart != "from_scratch":
    restart_state, restart_step = load_checkpoint(config.io.restart, config, ctx)
    stage_warning(
      "Init",
      f"Loaded restart checkpoint from {config.io.restart} at step {restart_step}.",
    )

  if requested_solver_mode in ("scf", "direct_opt"):
    return _run_with_mode(
      config,
      backend,
      ctx,
      requested_solver_mode,
      requested_solver_mode=requested_solver_mode,
      restart_state=restart_state,
      output_dir=output_dir,
    )

  if requested_solver_mode != "auto":
    raise ValueError(
      f"Unknown solver mode '{requested_solver_mode}'. "
      "Use 'auto', 'direct_opt', or 'scf'."
    )

  primary = config.solver.auto.primary
  fallback = config.solver.auto.fallback
  fallback_on_nonconverged = config.solver.auto.fallback_on_nonconverged
  fallback_on_error = config.solver.auto.fallback_on_error

  try:
    result = _run_with_mode(
      config,
      backend,
      ctx,
      primary,
      requested_solver_mode=requested_solver_mode,
      restart_state=restart_state,
      output_dir=output_dir,
    )
  except Exception as exc:
    if not fallback_on_error:
      raise
    stage_warning(
      "AUTO",
      f"Primary solver '{primary}' failed with "
      f"{exc.__class__.__name__}: {exc}. Falling back to '{fallback}'.",
    )
    fallback_config = deepcopy(config)
    fallback_config.solver.mode = fallback
    return _run_with_mode(
      fallback_config,
      backend,
      ctx,
      fallback,
      requested_solver_mode=requested_solver_mode,
      restart_state=restart_state,
      output_dir=output_dir,
    )

  if result.converged or not fallback_on_nonconverged:
    return result

  stage_warning(
    "AUTO",
    f"Primary solver '{primary}' did not converge. "
    f"Falling back to '{fallback}'.",
  )
  fallback_config = deepcopy(config)
  fallback_config.solver.mode = fallback
  fallback_state = _physical_state_from_result(result) or restart_state
  return _run_with_mode(
    fallback_config,
    backend,
    ctx,
    fallback,
    requested_solver_mode=requested_solver_mode,
    restart_state=fallback_state,
    output_dir=output_dir,
  )


def _run_metadata(
  *,
  task: str,
  config: JrystalConfigDict,
  crystal,
  started_at: str,
  actual_solver: Optional[str] = None,
  converged: Optional[bool] = None,
  finished_at: Optional[str] = None,
  error: Optional[str] = None,
) -> dict[str, object]:
  formula = (
    ase.Atoms(symbols=crystal.symbols).get_chemical_formula()
    if getattr(crystal, "symbols", None) else "system"
  )
  if not isinstance(actual_solver, str):
    actual_solver = None
  if not isinstance(converged, bool):
    converged = None
  return {
    "task":
      task,
    "formula":
      formula,
    "requested_solver_mode":
      config.solver.mode,
    "actual_solver":
      actual_solver,
    "restart":
      None if config.io.restart == "from_scratch" else config.io.restart,
    "converged":
      converged,
    "started_at":
      started_at,
    "finished_at":
      finished_at,
    "error":
      error,
  }


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
  crystal = create_crystal(config)
  run_dir = setup_output_dir(config, crystal, task="energy")
  save_config_snapshot(config, run_dir)
  open_log(run_dir / "jrystal.log")
  previous_log_level = get_log_level()
  set_log_level(config.io.log_level)
  started_at = datetime.now().isoformat()
  try:
    set_env_params(config)
    backend = get_backend(config)
    ctx = build_runtime_context(config, mode="mesh", backend=backend)
    log_system_info(
      config,
      ctx,
      backend,
      task="energy",
      started_at=started_at,
    )
    result = _run_ground_state(
      config,
      backend=backend,
      ctx=ctx,
      output_dir=run_dir,
    )
    actual_solver = getattr(result, "actual_solver", None)
    converged = getattr(result, "converged", None)
    if isinstance(result, GroundStateResult):
      result = save_ground_state(
        config,
        result,
        run_dir,
        ctx=ctx,
        backend=backend,
      )
      actual_solver = result.actual_solver
      converged = result.converged
    save_run_metadata(
      run_dir,
      _run_metadata(
        task="energy",
        config=config,
        crystal=crystal,
        started_at=started_at,
        actual_solver=actual_solver,
        converged=converged,
        finished_at=datetime.now().isoformat(),
      ),
    )
    return result
  except Exception as exc:
    save_run_metadata(
      run_dir,
      _run_metadata(
        task="energy",
        config=config,
        crystal=crystal,
        started_at=started_at,
        finished_at=datetime.now().isoformat(),
        error=f"{exc.__class__.__name__}: {exc}",
      ),
    )
    raise
  finally:
    set_log_level(previous_log_level)
    close_log()


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
  crystal = create_crystal(config)
  run_dir = setup_output_dir(config, crystal, task="band")
  save_config_snapshot(config, run_dir)
  open_log(run_dir / "jrystal.log")
  previous_log_level = get_log_level()
  set_log_level(config.io.log_level)
  started_at = datetime.now().isoformat()

  try:
    set_env_params(config)
    backend_mesh = get_backend(config)
    ctx_mesh = build_runtime_context(config, mode="mesh", backend=backend_mesh)
    log_system_info(
      config,
      ctx_mesh,
      backend_mesh,
      task="band",
      started_at=started_at,
    )

    if ground_state_result is None:
      ground_state_result = _run_ground_state(
        config,
        backend=backend_mesh,
        ctx=ctx_mesh,
        output_dir=run_dir,
      )

    actual_solver = getattr(ground_state_result, "actual_solver", None)
    converged = getattr(ground_state_result, "converged", None)
    if isinstance(ground_state_result, GroundStateResult):
      ground_state_result = save_ground_state(
        config,
        ground_state_result,
        run_dir,
        ctx=ctx_mesh,
        backend=backend_mesh,
      )
      actual_solver = ground_state_result.actual_solver
      converged = ground_state_result.converged

    backend_path = get_backend(config)
    ctx_path = build_runtime_context(config, mode="path", backend=backend_path)
    result = run_nscf(config, ctx_path, backend_path, ground_state_result)
    save_band_structure(config, result, run_dir)
    save_run_metadata(
      run_dir,
      _run_metadata(
        task="band",
        config=config,
        crystal=crystal,
        started_at=started_at,
        actual_solver=actual_solver,
        converged=converged,
        finished_at=datetime.now().isoformat(),
      ),
    )
    return result
  except Exception as exc:
    save_run_metadata(
      run_dir,
      _run_metadata(
        task="band",
        config=config,
        crystal=crystal,
        started_at=started_at,
        finished_at=datetime.now().isoformat(),
        error=f"{exc.__class__.__name__}: {exc}",
      ),
    )
    raise
  finally:
    set_log_level(previous_log_level)
    close_log()


__all__ = [
  "band",
  "energy",
  "BandStructureResult",
  "GroundStateResult",
]
