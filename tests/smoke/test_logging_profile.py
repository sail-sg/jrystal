from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import jrystal as jr
from jrystal import terminal_ui
from jrystal.calc.timer import PhaseTimer
from jrystal.calc.workflow_logging import (
  format_ground_state_iteration,
  log_system_info,
)
from tests.smoke.helpers import make_config


def test_terminal_log_level_filters_normal_lines_but_keeps_warnings(tmp_path):
  log_path = tmp_path / "jrystal.log"
  old_level = terminal_ui.get_log_level()
  terminal_ui.set_log_level("quiet")
  terminal_ui.open_log(log_path)
  try:
    terminal_ui.stage_line("SCF", "hidden-normal-line")
    terminal_ui.stage_warning("SCF", "visible-warning", color="red")
  finally:
    terminal_ui.close_log()
    terminal_ui.set_log_level(old_level)
  text = log_path.read_text(encoding="utf-8")
  assert "hidden-normal-line" not in text
  assert "visible-warning" in text


def test_iteration_format_stays_compact():
  old_level = terminal_ui.get_log_level()
  try:
    terminal_ui.set_log_level("normal")
    normal = format_ground_state_iteration(
      "SCF",
      step=12,
      max_steps=100,
      total_energy=-7.84321,
      delta_energy=2.1e-8,
      step_time=0.31,
      density_delta=8.7e-5,
      cumulative_time=4.2,
      chemical_potential=0.2341,
    )
    terminal_ui.set_log_level("verbose")
    verbose = format_ground_state_iteration(
      "SCF",
      step=12,
      max_steps=100,
      total_energy=-7.84321,
      delta_energy=2.1e-8,
      step_time=0.31,
      density_delta=8.7e-5,
      cumulative_time=4.2,
      chemical_potential=0.2341,
    )
  finally:
    terminal_ui.set_log_level(old_level)

  assert len(normal) <= 80
  assert len(verbose) <= 80
  assert "mu=" not in normal
  assert "mu=" in verbose
  assert "T=" in verbose


def test_log_system_info_includes_band_counts(tmp_path):
  config = make_config()
  ctx = SimpleNamespace(
    crystal=SimpleNamespace(
      num_electron=np.asarray(8),
      num_atom=2,
      symbols=["Si", "Si"],
      cell_vectors=np.eye(3, dtype=float),
      vol=1.0,
    ),
    basis=SimpleNamespace(
      grid_sizes=(4, 4, 4),
      num_g=3,
    ),
    ksampling=SimpleNamespace(
      mode="mesh",
      kpts=jnp.zeros((1, 3), dtype=jnp.float32),
    ),
    execution=SimpleNamespace(
      num_devices=1,
      parallel_over_k=False,
    ),
  )

  class _Backend:

    def num_electrons(self, _ctx):
      return 8

  log_path = tmp_path / "jrystal.log"
  old_level = terminal_ui.get_log_level()
  terminal_ui.set_log_level("normal")
  terminal_ui.open_log(log_path)
  try:
    log_system_info(
      config,
      ctx,
      _Backend(),
      task="energy",
      started_at="2026-04-09T12:34:56",
    )
  finally:
    terminal_ui.close_log()
    terminal_ui.set_log_level(old_level)

  text = log_path.read_text(encoding="utf-8")
  assert "Bands    occupied=4 empty=2 total=6" in text
  assert "solver=auto" in text


def test_phase_timer_records_summary():
  timer = PhaseTimer(enabled=True)
  timer.record("update_step", 0.5)
  timer.record("update_step", 0.25)
  timer.record("checkpoint", 0.1)

  summary = timer.summary(total_wall_time=1.0)

  assert summary["update_step"]["calls"] == 2
  assert summary["update_step"]["total_s"] == 0.75
  assert summary["checkpoint"]["pct_total"] == 10.0


def test_execution_profile_default_is_false():
  config = jr.config.get_config(None)
  assert config.execution.profile is False
