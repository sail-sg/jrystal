"""Smoke tests for the public jr.calc.energy() / jr.calc.band() API."""
from pathlib import Path
import tempfile

import jax
import numpy as np
from absl.testing import absltest
from unittest import mock

import jrystal.calc as calc
from jrystal.calc import band, energy
from jrystal.calc.types import BandStructureResult, GroundStateResult
from jrystal.config import JrystalConfigDict, _normalize_config

jax.config.update("jax_enable_x64", True)


def _tiny_ae_config():
  """Minimal AE config for fast smoke testing."""
  raw = {
    "system": {"crystal": "diamond"},
    "basis": {"grid_sizes": 16, "cutoff_energy": 20},
    "ksampling": {"k_grid_sizes": [1, 1, 1]},
    "solver": {
      "mode": "direct_opt",
      "direct_opt": {"max_steps": 3},
    },
    "occupation": {"empty_bands": 2},
  }
  return JrystalConfigDict(_normalize_config(raw))


class EnergyDispatchTest(absltest.TestCase):

  def test_energy_returns_ground_state_result(self):
    config = _tiny_ae_config()
    result = energy(config)
    self.assertIsInstance(result, GroundStateResult)
    self.assertIsNotNone(result.total_energy)
    self.assertIsNotNone(result.density)
    self.assertIsNotNone(result.energy_terms)

  def test_energy_direct_opt_solver(self):
    config = _tiny_ae_config()
    config.solver.mode = "direct_opt"
    result = energy(config)
    self.assertIsInstance(result, GroundStateResult)

  def test_energy_unknown_solver_raises(self):
    config = _tiny_ae_config()
    config.solver.mode = "bogus"
    with self.assertRaises(ValueError):
      energy(config)

  def test_energy_scf_rejects_unrestricted_spin(self):
    config = _tiny_ae_config()
    config.solver.mode = "scf"
    config.system.spin_restricted = False
    with self.assertRaises(NotImplementedError):
      energy(config)

  def test_energy_scf_solver_returns_ground_state_result(self):
    config = _tiny_ae_config()
    config.solver.mode = "scf"
    config.solver.scf.max_iter = 1
    config.execution.verbose = False

    result = energy(config)

    self.assertIsInstance(result, GroundStateResult)
    self.assertIsNotNone(result.total_energy)
    self.assertIsNotNone(result.density)

  def test_energy_scf_uses_scf_max_iter(self):
    config = _tiny_ae_config()
    config.solver.mode = "scf"
    config.solver.direct_opt.max_steps = 0
    config.solver.scf.max_iter = 1
    config.execution.verbose = False

    result = energy(config)

    self.assertIsInstance(result, GroundStateResult)
    self.assertIsNotNone(result.total_energy)
    self.assertLessEqual(len(result.total_energy_history), 1)

  def test_energy_auto_falls_back_to_direct_opt(self):
    config = _tiny_ae_config()
    config.solver.mode = "auto"
    config.solver.auto.primary = "scf"
    config.solver.auto.fallback = "direct_opt"
    scf_result = mock.Mock(converged=False)
    direct_opt_result = mock.Mock(converged=True)

    with mock.patch.object(calc, "set_env_params"), mock.patch.object(
      calc,
      "get_backend",
      return_value=mock.sentinel.backend,
    ), mock.patch.object(
      calc,
      "build_runtime_context",
      return_value=mock.sentinel.ctx,
    ), mock.patch.object(
      calc,
      "run_scf",
      return_value=scf_result,
    ) as run_scf_mock, mock.patch.object(
      calc,
      "run_direct_opt",
      return_value=direct_opt_result,
    ) as run_direct_opt_mock:
      result = energy(config)

    self.assertIs(result, direct_opt_result)
    run_scf_mock.assert_called_once()
    run_direct_opt_mock.assert_called_once()

  def test_band_returns_band_structure_result(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      temp_path = Path(tmp_dir)
      k_path_file = temp_path / "k_path.npy"
      np.save(
        k_path_file,
        np.array([
          [0.0, 0.0, 0.0],
          [0.5, 0.0, 0.0],
          [0.5, 0.5, 0.0],
        ]),
      )

      raw = {
        "system": {"crystal": "diamond"},
        "basis": {"grid_sizes": 16, "cutoff_energy": 20},
        "ksampling": {"k_grid_sizes": [1, 1, 1]},
        "solver": {
          "mode": "direct_opt",
          "direct_opt": {"max_steps": 2},
        },
        "occupation": {"empty_bands": 2},
        "band": {
          "k_path_file": str(k_path_file),
          "num_kpoints": 99,
          "epoch": 1,
          "fine_tuning_epoch": 1,
        },
        "execution": {
          "verbose": False,
          "parallel_over_k_path": False,
        },
        "io": {"save_dir": tmp_dir},
      }
      config = JrystalConfigDict(_normalize_config(raw))

      ground_state_result = energy(config)
      result = band(config, ground_state_result=ground_state_result)

      self.assertIsInstance(result, BandStructureResult)
      self.assertEqual(result.kpath.kpts.shape[0], 3)
      self.assertEqual(result.eigenvalues.shape[1], 3)
      self.assertTrue((temp_path / "CC_band_structure.npy").exists())


if __name__ == "__main__":
  absltest.main()
