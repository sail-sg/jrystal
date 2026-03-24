"""Smoke tests for package imports."""
from absl.testing import absltest


class ImportTest(absltest.TestCase):

  def test_import_jrystal(self):
    import jrystal  # noqa: F401

  def test_import_calc(self):
    import jrystal.calc  # noqa: F401

  def test_calc_has_energy_normcons(self):
    import jrystal.calc as calc
    self.assertTrue(hasattr(calc, 'energy_normcons'))

  def test_calc_has_energy_all_electrons(self):
    import jrystal.calc as calc
    self.assertTrue(hasattr(calc, 'energy_all_electrons'))

  def test_calc_has_band_normcons(self):
    import jrystal.calc as calc
    self.assertTrue(hasattr(calc, 'band_normcons'))

  def test_calc_has_band_all_electrons(self):
    import jrystal.calc as calc
    self.assertTrue(hasattr(calc, 'band_all_electrons'))

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


if __name__ == '__main__':
  absltest.main()
