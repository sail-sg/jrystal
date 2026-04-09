from __future__ import annotations

import json

from jrystal.io import save_ground_state
from tests.smoke.helpers import (
  make_config,
  make_dummy_ctx,
  make_ground_state_result,
)


def test_convergence_json_keeps_new_iteration_fields(tmp_path):
  config = make_config()
  run_dir = tmp_path / "run"
  run_dir.mkdir()
  ctx = make_dummy_ctx(config)
  result = make_ground_state_result(config)
  result.convergence_history = [
    {
      "step": 1,
      "total_energy": -7.0,
      "delta_energy": None,
      "delta_density": 0.5,
      "wall_time": 0.4,
      "cumulative_time_s": 0.4,
      "chemical_potential_ha": -0.1,
    },
  ]

  save_ground_state(config, result, run_dir, ctx=ctx, backend=None)

  convergence = json.loads(
    (run_dir / "ground_state" / "convergence.json").read_text(
      encoding="utf-8"
    )
  )
  energy = json.loads(
    (run_dir / "ground_state" / "energy.json").read_text(encoding="utf-8")
  )

  assert "cumulative_time_s" in convergence["columns"]
  assert "chemical_potential_ha" in convergence["columns"]
  assert convergence["units"][
    convergence["columns"].index("cumulative_time_s")
  ] == "s"
  assert convergence["units"][
    convergence["columns"].index("chemical_potential_ha")
  ] == "Ha"
  assert energy["chemical_potential_ha"] == -0.1
