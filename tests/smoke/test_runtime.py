from __future__ import annotations

import json

from jrystal.io import save_config_snapshot, save_run_metadata, setup_output_dir
from tests.smoke.helpers import make_config, make_dummy_crystal


def test_setup_output_dir_and_run_metadata(tmp_path):
  config = make_config()
  config.io.output_dir = str(tmp_path)
  run_dir = setup_output_dir(config, make_dummy_crystal(), task="energy")
  save_config_snapshot(config, run_dir)
  save_run_metadata(
    run_dir,
    {
      "task": "energy",
      "requested_solver_mode": "auto",
      "actual_solver": "scf",
    },
  )
  assert run_dir.name.startswith("Si2_energy_")
  assert (run_dir / "config.yaml").exists()
  payload = json.loads((run_dir / "run.json").read_text())
  assert payload["actual_solver"] == "scf"
