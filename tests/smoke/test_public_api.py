"""Smoke tests for the public jr.calc.energy() / jr.calc.band() API."""
import jax
from absl.testing import absltest

from jrystal.calc import energy, band
from jrystal.calc.types import GroundStateResult
from jrystal.config import JrystalConfigDict, _normalize_config

jax.config.update("jax_enable_x64", True)


def _tiny_ae_config():
  """Minimal AE config for fast smoke testing."""
  raw = {
    "system": {"crystal": "diamond"},
    "basis": {"grid_sizes": 16, "cutoff_energy": 20},
    "ksampling": {"k_grid_sizes": [1, 1, 1]},
    "solver": {"epoch": 3, "type": "direct_opt"},
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
    config.solver.type = "direct_opt"
    result = energy(config)
    self.assertIsInstance(result, GroundStateResult)

  def test_energy_unknown_solver_raises(self):
    config = _tiny_ae_config()
    config.solver.type = "bogus"
    with self.assertRaises(ValueError):
      energy(config)


if __name__ == "__main__":
  absltest.main()
