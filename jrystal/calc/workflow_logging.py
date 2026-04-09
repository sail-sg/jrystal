"""Shared logging helpers for workflow progress and summaries."""

from __future__ import annotations

import os
from collections.abc import Mapping
from math import ceil
from typing import Optional

import ase
import numpy as np

from .. import __version__
from ..terminal_ui import (
  console_line,
  get_log_level,
  stage_prefix,
  stage_warning,
)

HARTREE_TO_EV = 27.211386245988


def _format_optional(value: Optional[float], fmt: str) -> str:
  if value is None:
    return "n/a"
  return fmt.format(value)


def _dtype_label(use_x64: bool) -> str:
  return "float64/complex128" if use_x64 else "float32/complex64"


def _solver_mode_summary(config) -> str:
  mode = config.solver.mode
  if mode != "auto":
    return mode
  auto = config.solver.auto
  return (
    f"auto primary={auto.primary} "
    f"fallback={auto.fallback}"
  )


def _electronic_counts(config, num_electrons: int) -> tuple[int, int, int]:
  occ_max = 2 if bool(config.system.spin_restricted) else 1
  occupied = int(ceil(num_electrons / occ_max))
  empty = int(config.occupation.empty_bands)
  return occupied, empty, occupied + empty


def _method_label(config) -> str:
  if not config.method.use_pseudopotential:
    return f"AE xc={config.method.xc}"
  pp_kind = str(config.method.pseudopotential_type).lower()
  if pp_kind in {"nc", "normcons", "normconserving"}:
    kind = "NC"
  elif pp_kind in {"us", "ultrasoft"}:
    kind = "USPP"
  else:
    kind = pp_kind
  return f"{kind} xc={config.method.xc}"


def _k_summary(config, ctx) -> str:
  if ctx.ksampling.mode == "path":
    return f"path n={int(ctx.ksampling.kpts.shape[0])}"
  grid = tuple(int(x) for x in config.ksampling.k_grid_sizes)
  return (
    f"mesh {grid[0]}x{grid[1]}x{grid[2]} "
    f"irr={int(ctx.ksampling.kpts.shape[0])}"
  )


def _started_label(started_at: str) -> str:
  return started_at.replace("T", " ")[:19]


def log_system_info(
  config,
  ctx,
  backend,
  *,
  task: str,
  started_at: str,
) -> None:
  """Log a run-level system summary once per workflow."""
  required = ("crystal", "basis", "ksampling", "execution")
  if any(not hasattr(ctx, attr) for attr in required):
    return
  if not hasattr(backend, "num_electrons"):
    return
  crystal = ctx.crystal
  num_electrons = int(backend.num_electrons(ctx))
  occupied, empty, total_bands = _electronic_counts(config, num_electrons)
  symbols = list(getattr(crystal, "symbols", []) or [])
  formula = (
    ase.Atoms(symbols=symbols).get_chemical_formula() if symbols else "system"
  )
  lattice = np.asarray(getattr(crystal, "cell_vectors", np.zeros((3, 3))))
  console_line("=" * 72, level="normal")
  console_line(
    f"Jrystal v{__version__} | task={task} | started={_started_label(started_at)}",
    level="normal",
  )
  console_line(
    f"pid={os.getpid()} | solver={_solver_mode_summary(config)}",
    level="normal",
  )
  console_line("=" * 72, level="normal")
  console_line(
    (
      f"System   formula={formula} atoms={int(crystal.num_atom)} "
      f"electrons={num_electrons}"
    ),
    level="normal",
  )
  console_line(
    (
      f"Method   {_method_label(config)} "
      f"spin={'restricted' if config.system.spin_restricted else 'unrestricted'}"
    ),
    level="normal",
  )
  console_line(
    (
      f"Basis    cutoff={float(config.basis.cutoff_energy):.1f} Ha "
      f"grid={tuple(int(x) for x in ctx.basis.grid_sizes)} "
      f"g={int(ctx.basis.num_g)}"
    ),
    level="normal",
  )
  console_line(
    f"Bands    occupied={occupied} empty={empty} total={total_bands}",
    level="normal",
  )
  console_line(f"K        {_k_summary(config, ctx)}", level="normal")
  console_line(
    (
      f"Exec     devices={ctx.execution.num_devices} "
      f"pk={ctx.execution.parallel_over_k} "
      f"dtype={_dtype_label(bool(config.execution.jax_enable_x64))}"
    ),
    level="normal",
  )
  console_line(
    (
      f"a1=({lattice[0,0]:7.3f},{lattice[0,1]:7.3f},{lattice[0,2]:7.3f}) "
      f"vol={float(crystal.vol):.3f}"
    ),
    level="normal",
  )
  console_line(
    f"a2=({lattice[1,0]:7.3f},{lattice[1,1]:7.3f},{lattice[1,2]:7.3f})",
    level="normal",
  )
  console_line(
    f"a3=({lattice[2,0]:7.3f},{lattice[2,1]:7.3f},{lattice[2,2]:7.3f})",
    level="normal",
  )


def log_ground_state_start(
  solver_name: str,
  *,
  max_steps: int,
  num_bands: int,
  smearing: float,
  xc: str,
  controls: str,
) -> None:
  """Log a compact ground-state workflow header."""
  prefix = stage_prefix(solver_name)
  console_line(
    f"{prefix} start steps={max_steps} bands={num_bands} smear={smearing:.4f}",
    level="normal",
  )
  console_line(f"{prefix} {xc} | {controls}", level="normal")


def format_ground_state_iteration(
  solver_name: str,
  *,
  step: int,
  max_steps: int,
  total_energy: float,
  delta_energy: Optional[float],
  step_time: Optional[float] = None,
  cumulative_time: Optional[float] = None,
  density_delta: Optional[float] = None,
  energy_std: Optional[float] = None,
  chemical_potential: Optional[float] = None,
) -> str:
  """Create a compact <=80-char-ish iteration-progress string."""
  del solver_name
  parts = [
    f"{step}/{max_steps}",
    f"E={total_energy:.6f}",
    f"dE={_format_optional(delta_energy, '{:.1e}')}",
  ]
  if density_delta is not None:
    parts.append(f"dR={density_delta:.1e}")
  if energy_std is not None:
    parts.append(f"sd={energy_std:.1e}")
  if step_time is not None:
    parts.append(f"dt={step_time:.2f}s")
  if get_log_level() == "verbose":
    if chemical_potential is not None:
      parts.append(f"mu={chemical_potential:.6f}")
    if cumulative_time is not None:
      parts.append(f"T={cumulative_time:.1f}s")
  return " ".join(parts)


def log_ground_state_finish(
  solver_name: str,
  *,
  converged: bool,
  steps_completed: int,
  max_steps: int,
  total_energy: float,
  wall_time: float,
) -> None:
  """Log a compact solver-finish summary."""
  prefix = stage_prefix(solver_name)
  status = "converged" if converged else "not_converged"
  line1 = (
    f"{prefix} done {status} steps={steps_completed}/{max_steps} "
    f"t={wall_time:.2f}s"
  )
  line2 = (
    f"{prefix} E={total_energy:.6f} Ha "
    f"({total_energy * HARTREE_TO_EV:.3f} eV)"
  )
  if converged:
    console_line(line1, level="quiet")
    console_line(line2, level="quiet")
  else:
    stage_warning(solver_name, line1.removeprefix(f"{prefix} "), color="red")
    stage_warning(solver_name, line2.removeprefix(f"{prefix} "), color="red")


def log_energy_breakdown(
  solver_name: str,
  energy_terms: Mapping[str, float],
  *,
  ewald: float,
  total_energy: float,
) -> None:
  """Log a compact energy-breakdown footer."""
  prefix = stage_prefix(solver_name)
  console_line(f"{prefix} energy breakdown", level="quiet")
  for name, raw_value in energy_terms.items():
    value = float(raw_value)
    console_line(
      f"{prefix} {name}={value:.4f} Ha ({value * HARTREE_TO_EV:.2f} eV)",
      level="quiet",
    )
  console_line(
    f"{prefix} ewald={float(ewald):.4f} Ha ({float(ewald) * HARTREE_TO_EV:.2f} eV)",
    level="quiet",
  )
  console_line(
    (
      f"{prefix} total={float(total_energy):.4f} Ha "
      f"({float(total_energy) * HARTREE_TO_EV:.2f} eV)"
    ),
    level="quiet",
  )


def log_timing_breakdown(
  solver_name: str,
  breakdown: Mapping[str, Mapping[str, float]],
  *,
  total_wall_time: float,
) -> None:
  """Log a compact timing summary."""
  prefix = stage_prefix(solver_name)
  console_line(f"{prefix} timing", level="normal")
  for phase, stats in breakdown.items():
    console_line(
      (
        f"{prefix} {phase}: n={int(stats['calls'])} "
        f"tot={stats['total_s']:.3f}s avg={stats['avg_s']:.3f}s "
        f"pct={stats['pct_total']:.1f}%"
      ),
      level="normal",
    )
  console_line(f"{prefix} wall={total_wall_time:.3f}s", level="normal")
