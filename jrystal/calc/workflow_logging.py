"""Shared logging helpers for workflow progress and summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from absl import logging


def _format_optional(value: Optional[float], fmt: str) -> str:
  if value is None:
    return "n/a"
  return fmt.format(value)


def log_ground_state_start(
  solver_name: str,
  *,
  max_steps: int,
  num_bands: int,
  smearing: float,
  xc: str,
  controls: str,
) -> None:
  """Log a solver-agnostic ground-state workflow header."""
  logging.info(
    "[%s] Start | max_steps=%d | num_bands=%d | smearing=%.4f | xc=%s",
    solver_name,
    max_steps,
    num_bands,
    smearing,
    xc,
  )
  logging.info("[%s] Controls | %s", solver_name, controls)


def format_ground_state_iteration(
  solver_name: str,
  *,
  step: int,
  max_steps: int,
  total_energy: float,
  delta_energy: Optional[float],
  step_time: Optional[float] = None,
  density_delta: Optional[float] = None,
  energy_std: Optional[float] = None,
) -> str:
  """Create a consistent iteration-progress string."""
  parts = [
    f"[{solver_name}] iter {step}/{max_steps}",
    f"E_total={total_energy:.6f} Ha",
    f"dE={_format_optional(delta_energy, '{:.2e} Ha')}",
  ]
  if density_delta is not None:
    parts.append(f"dRho={density_delta:.2e}")
  if energy_std is not None:
    parts.append(f"E_std={energy_std:.2e} Ha")
  if step_time is not None:
    parts.append(f"dt={step_time:.2f}s")
  return " | ".join(parts)


def log_ground_state_finish(
  solver_name: str,
  *,
  converged: bool,
  steps_completed: int,
  max_steps: int,
  total_energy: float,
  wall_time: float,
) -> None:
  """Log a consistent solver-finish summary."""
  message = (
    f"[{solver_name}] Finished | "
    f"status={'converged' if converged else 'not_converged'} | "
    f"steps={steps_completed}/{max_steps} | "
    f"E_total={total_energy:.6f} Ha | "
    f"wall_time={wall_time:.2f}s"
  )
  if converged:
    logging.info(message)
  else:
    logging.warning(message)


def log_energy_breakdown(
  solver_name: str,
  energy_terms: Mapping[str, float],
  *,
  ewald: float,
  total_energy: float,
) -> None:
  """Log a unified energy-breakdown footer."""
  logging.info("[%s] Energy breakdown:", solver_name)
  for name, value in energy_terms.items():
    logging.info("[%s]   %s=%.4f Ha", solver_name, name, float(value))
  logging.info("[%s]   ewald=%.4f Ha", solver_name, float(ewald))
  logging.info("[%s]   total=%.4f Ha", solver_name, float(total_energy))
