"""Smoke tests for package imports."""
from absl.testing import absltest


class ImportTest(absltest.TestCase):

  def test_import_jrystal(self):
    import jrystal  # noqa: F401

  def test_import_calc(self):
    import jrystal.calc  # noqa: F401

  def test_calc_has_energy(self):
    import jrystal.calc as calc
    self.assertTrue(callable(calc.energy))

  def test_calc_has_band(self):
    import jrystal.calc as calc
    self.assertTrue(callable(calc.band))

  def test_import_config(self):
    from jrystal.config import get_config
    config = get_config()
    self.assertIsNotNone(config)

  def test_import_crystal(self):
    from jrystal import Crystal
    self.assertTrue(callable(Crystal))

  def test_import_grid(self):
    import jrystal.grid  # noqa: F401

  def test_import_pseudopotential(self):
    import jrystal.pseudopotential  # noqa: F401

  def test_import_backend(self):
    from jrystal.calc.backend import (  # noqa: F401
      AllElectronBackend,
      NormConservingBackend,
      get_backend,
    )

  def test_import_solvers(self):
    from jrystal.calc.solver_direct_opt import run_direct_opt  # noqa: F401
    from jrystal.calc.solver_scf import run_scf  # noqa: F401
    from jrystal.calc.solver_nscf import run_nscf  # noqa: F401


if __name__ == '__main__':
  absltest.main()
