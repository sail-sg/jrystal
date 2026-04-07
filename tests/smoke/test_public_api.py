from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp
import numpy as np
import pytest

from jrystal import calc
from jrystal._src import occupation as _occupation
from jrystal._src.entropy import von_neumann
from jrystal.calc import band, energy
from jrystal.calc.solver_direct_opt import (
  _build_occupation_setup,
  _free_energy_from_total_energy,
  _freeze_occupation_gradient,
)
from jrystal.calc.solver_scf import _occupation_max
from jrystal.calc.types import BandStructureResult, GroundStateResult
from tests.smoke.helpers import (
  make_band_result,
  make_config,
  make_dummy_crystal,
  make_ground_state_result,
)


def _patch_common_workflow_io(monkeypatch, run_dir: Path):
  monkeypatch.setattr(
    calc, "create_crystal", lambda _config: make_dummy_crystal()
  )
  monkeypatch.setattr(
    calc, "setup_output_dir", lambda *_args, **_kwargs: run_dir
  )
  monkeypatch.setattr(
    calc, "save_config_snapshot", lambda *_args, **_kwargs: None
  )
  monkeypatch.setattr(calc, "save_run_metadata", lambda *_args, **_kwargs: None)
  monkeypatch.setattr(calc, "open_log", lambda *_args, **_kwargs: None)
  monkeypatch.setattr(calc, "close_log", lambda *_args, **_kwargs: None)
  monkeypatch.setattr(calc, "set_env_params", lambda *_args, **_kwargs: None)


def test_direct_opt_free_energy_includes_entropy_when_smearing_is_finite():
  occupation = jnp.asarray([[[1.6, 0.4]]], dtype=jnp.float32)
  free_energy, entropy = _free_energy_from_total_energy(
    jnp.asarray(10.0, dtype=jnp.float32),
    occupation,
    0.2,
  )

  assert float(entropy) == pytest.approx(float(von_neumann(occupation)))
  assert float(free_energy) == pytest.approx(10.0 - 0.2 * float(entropy))


def test_direct_opt_single_k_zero_smearing_uses_fixed_occupation():
  with mock.patch(
    "jrystal.calc.solver_direct_opt._occupation.params_init",
  ) as params_init_mock:
    setup = _build_occupation_setup(
      num_electrons=8,
      spin=0,
      spin_restricted=True,
      num_bands=6,
      num_kpts=1,
      smearing=0.0,
    )

  assert not setup.trainable
  assert setup.params == {}
  params_init_mock.assert_not_called()
  np.testing.assert_allclose(
    np.asarray(setup.fn({})),
    np.asarray(
      _occupation._get_fixed_occupation(
        num_k=1,
        num_electrons=8,
        spin=0,
        num_bands=6,
        spin_restricted=True,
      )
    ),
  )


def test_direct_opt_warmup_freezes_occupation_gradient():
  grad = {
    "pw": {
      "w_re": jnp.asarray([1.0, -2.0]),
    },
    "occ": {
      "param_up": jnp.asarray([0.3, -0.4]),
    },
  }

  frozen = _freeze_occupation_gradient(grad)

  np.testing.assert_allclose(
    np.asarray(frozen["pw"]["w_re"]),
    np.asarray(grad["pw"]["w_re"]),
  )
  np.testing.assert_allclose(
    np.asarray(frozen["occ"]["param_up"]),
    np.zeros_like(np.asarray(grad["occ"]["param_up"])),
  )


def test_scf_occupation_max_depends_on_spin_restriction():
  assert _occupation_max(True) == 2.0
  assert _occupation_max(False) == 1.0


def test_energy_returns_ground_state_result_via_public_api(
  monkeypatch, tmp_path
):
  config = make_config()
  config.solver.mode = "direct_opt"
  run_dir = tmp_path / "energy"
  run_dir.mkdir()
  expected = make_ground_state_result(config)
  build_modes = []

  _patch_common_workflow_io(monkeypatch, run_dir)
  monkeypatch.setattr(calc, "get_backend", lambda *_args, **_kwargs: "backend")
  monkeypatch.setattr(
    calc,
    "build_runtime_context",
    lambda _config, *, mode, backend: (
      build_modes.append((mode, backend)) or
      SimpleNamespace(ksampling=SimpleNamespace(weights=jnp.ones((1,))))
    ),
  )
  monkeypatch.setattr(
    calc, "_run_ground_state", lambda *_args, **_kwargs: expected
  )
  monkeypatch.setattr(
    calc, "save_ground_state", lambda *_args, **_kwargs: expected
  )

  result = energy(config)

  assert result is expected
  assert isinstance(result, GroundStateResult)
  assert build_modes == [("mesh", "backend")]


def test_run_ground_state_auto_falls_back_to_direct_opt(monkeypatch, tmp_path):
  config = make_config()
  config.solver.mode = "auto"
  config.solver.auto.primary = "scf"
  config.solver.auto.fallback = "direct_opt"
  scf_result = make_ground_state_result(config, converged=False)
  direct_opt_result = make_ground_state_result(config, converged=True)
  direct_opt_result.actual_solver = "direct_opt"
  calls = []

  def _fake_run_with_mode(
    _config,
    _backend,
    _ctx,
    mode,
    *,
    requested_solver_mode,
    restart_state=None,
    output_dir=None,
  ):
    calls.append((mode, requested_solver_mode, restart_state, output_dir))
    if mode == "scf":
      return scf_result
    return direct_opt_result

  monkeypatch.setattr(calc, "_run_with_mode", _fake_run_with_mode)

  result = calc._run_ground_state(
    config,
    backend="backend",
    ctx="ctx",
    output_dir=tmp_path,
  )

  assert result is direct_opt_result
  assert len(calls) == 2
  assert calls[0][0] == "scf"
  assert calls[1][0] == "direct_opt"
  assert calls[1][1] == "auto"


def test_band_uses_provided_ground_state_result(monkeypatch, tmp_path):
  config = make_config()
  run_dir = tmp_path / "band"
  run_dir.mkdir()
  ground_state_result = make_ground_state_result(config)
  expected = make_band_result(config)
  build_modes = []
  save_calls = []

  _patch_common_workflow_io(monkeypatch, run_dir)
  monkeypatch.setattr(calc, "get_backend", lambda *_args, **_kwargs: "backend")
  monkeypatch.setattr(
    calc,
    "build_runtime_context",
    lambda _config, *, mode, backend:
    (build_modes.append((mode, backend)) or SimpleNamespace(mode=mode)),
  )
  monkeypatch.setattr(
    calc,
    "_run_ground_state",
    lambda *_args, **_kwargs:
    (_ for _ in
     ()).throw(AssertionError("_run_ground_state should not be called")),
  )
  monkeypatch.setattr(
    calc,
    "save_ground_state",
    lambda *_args, **_kwargs: ground_state_result,
  )
  monkeypatch.setattr(
    calc,
    "run_nscf",
    lambda _config, ctx, backend, gs_result:
    (build_modes.append(("nscf", ctx.mode, backend, gs_result)) or expected),
  )
  monkeypatch.setattr(
    calc,
    "save_band_structure",
    lambda _config, result, output_dir: save_calls.append((result, output_dir)),
  )

  result = band(config, ground_state_result=ground_state_result)

  assert result is expected
  assert isinstance(result, BandStructureResult)
  assert build_modes[0] == ("mesh", "backend")
  assert build_modes[1] == ("path", "backend")
  assert build_modes[2][0] == "nscf"
  assert save_calls == [(expected, run_dir)]
