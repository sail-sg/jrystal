"""Shared logging helpers for workflow progress and summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Optional

from ..terminal_ui import console_line, metric, stage_prefix, stage_warning


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
  prefix = stage_prefix(solver_name)
  console_line(
    f"{prefix} Start | max_steps={max_steps} | num_bands={num_bands} | "
    f"smearing={smearing:.4f} | xc={xc}"
  )
  console_line(f"{prefix} Controls | {controls}")


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
    f"iter {step}/{max_steps}",
    metric("E", f"{total_energy:.6f} Ha", color="green"),
    metric("dE", _format_optional(delta_energy, "{:.2e} Ha"), color="yellow"),
  ]
  if density_delta is not None:
    parts.append(metric("dRho", f"{density_delta:.2e}", color="magenta"))
  if energy_std is not None:
    parts.append(metric("Estd", f"{energy_std:.2e} Ha", color="red"))
  if step_time is not None:
    parts.append(metric("dt", f"{step_time:.2f}s", color="blue"))
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
    "Finished | "
    f"status={'converged' if converged else 'not_converged'} | "
    f"steps={steps_completed}/{max_steps} | "
    f"E_total={total_energy:.6f} Ha | "
    f"wall_time={wall_time:.2f}s"
  )
  if converged:
    console_line(f"{stage_prefix(solver_name)} {message}")
  else:
    stage_warning(solver_name, message, color="red")


def log_energy_breakdown(
  solver_name: str,
  energy_terms: Mapping[str, float],
  *,
  ewald: float,
  total_energy: float,
) -> None:
  """Log a unified energy-breakdown footer."""
  prefix = stage_prefix(solver_name)
  console_line(f"{prefix} Energy breakdown:")
  for name, value in energy_terms.items():
    console_line(f"{prefix}   {name}={float(value):.4f} Ha")
  console_line(f"{prefix}   ewald={float(ewald):.4f} Ha")
  console_line(f"{prefix}   total={float(total_energy):.4f} Ha")
