from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np
import pytest

from jrystal.io import (
  _infer_fermi_energy,
  load_checkpoint,
  make_checkpoint_manager,
  save_band_structure,
  save_checkpoint,
  save_config_snapshot,
  save_ground_state,
)
from tests.smoke.helpers import (
  make_band_result,
  make_config,
  make_dummy_ctx,
  make_ground_state_result,
)


def test_save_ground_state_and_band_outputs(tmp_path):
  config = make_config()
  run_dir = tmp_path / "run"
  run_dir.mkdir()
  ctx = make_dummy_ctx(config)

  gs_result = save_ground_state(
    config,
    make_ground_state_result(config),
    run_dir,
    ctx=ctx,
    backend=None,
  )
  assert gs_result.fermi_energy == -0.1
  assert (run_dir / "ground_state" / "energy.json").exists()
  assert (run_dir / "ground_state" / "convergence.json").exists()
  assert (run_dir / "ground_state" / "density.npy").exists()
  assert (run_dir / "ground_state" / "eigenvalues.npy").exists()
  assert (run_dir / "ground_state" / "occupations.npy").exists()
  assert (run_dir / "ground_state" / "coefficients.npz").exists()

  payload = json.loads((run_dir / "ground_state" / "energy.json").read_text())
  assert payload["actual_solver"] == "scf"
  assert payload["fermi_energy_available"] is True

  save_band_structure(config, make_band_result(config), run_dir)
  assert (run_dir / "band" / "eigenvalues.npy").exists()
  assert (run_dir / "band" / "kpath.json").exists()


def test_checkpoint_roundtrip(tmp_path):
  config = make_config()
  run_dir = tmp_path / "run"
  run_dir.mkdir()
  save_config_snapshot(config, run_dir)
  ctx = make_dummy_ctx(config)

  state = {
    "density": jnp.zeros((1, 4, 4, 4), dtype=jnp.float32),
    "coefficients":
      {
        "w_re": jnp.ones((1, 1, 3, 6), dtype=jnp.float32),
        "w_im": jnp.zeros((1, 1, 3, 6), dtype=jnp.float32),
      },
    "occupations": jnp.ones((1, 1, 6), dtype=jnp.float32),
    "eigenvalues": jnp.zeros((1, 1, 6), dtype=jnp.float32),
    "has_eigenvalues": True,
    "step": 3,
    "total_energy": -7.5,
  }
  manager = make_checkpoint_manager(run_dir)
  save_checkpoint(manager, state, 3)
  manager.wait_until_finished()

  restored, step = load_checkpoint(str(run_dir), config, ctx)
  assert step == 3
  np.testing.assert_allclose(
    np.asarray(restored["density"]), np.asarray(state["density"])
  )
  np.testing.assert_allclose(
    np.asarray(restored["coefficients"]["w_re"]),
    np.asarray(state["coefficients"]["w_re"]),
  )
  assert float(restored["total_energy"]) == pytest.approx(-7.5)


def test_restart_shape_mismatch_raises(tmp_path):
  config = make_config()
  run_dir = tmp_path / "run"
  run_dir.mkdir()
  save_config_snapshot(config, run_dir)
  ctx = make_dummy_ctx(config)
  manager = make_checkpoint_manager(run_dir)
  save_checkpoint(
    manager,
    {
      "density": jnp.zeros((1, 4, 4, 4), dtype=jnp.float32),
      "coefficients":
        {
          "w_re": jnp.zeros((1, 1, 3, 6), dtype=jnp.float32),
          "w_im": jnp.zeros((1, 1, 3, 6), dtype=jnp.float32),
        },
      "occupations": jnp.zeros((1, 1, 6), dtype=jnp.float32),
      "eigenvalues": jnp.zeros((1, 1, 6), dtype=jnp.float32),
      "has_eigenvalues": False,
      "step": 0,
      "total_energy": 0.0,
    },
    0,
  )
  manager.wait_until_finished()

  incompatible = make_config()
  incompatible.basis.cutoff_energy = config.basis.cutoff_energy + 1
  with pytest.raises(ValueError, match="basis.cutoff_energy"):
    load_checkpoint(str(run_dir), incompatible, ctx)


def test_infer_fermi_energy_uses_highest_occupied_state_for_insulator():
  config = make_config()
  config.occupation.smearing = 1e-3
  result = make_ground_state_result(config)
  result.eigenvalues = jnp.asarray(
    [[[-3.0, -1.0, 1.0, 4.0, 6.0, 8.0]]],
    dtype=jnp.float32,
  )
  result.occupations = jnp.asarray(
    [[[2.0, 2.0, 0.0, 0.0, 0.0, 0.0]]],
    dtype=jnp.float32,
  )

  fermi = _infer_fermi_energy(
    result, k_weights=jnp.ones((1,), dtype=jnp.float32)
  )

  assert fermi == pytest.approx(-1.0)
