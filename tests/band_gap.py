import jrystal
from absl.testing import absltest
import jax.numpy as jnp
from jrystal._src.const import HARTREE2EV


class Test_band_gap(absltest.TestCase):

  def test_diamond(self):
    config = jrystal.config.get_config()
    config.crystal = "diamond"
    config.grid_sizes = 24
    config.epoch = 3000
    config.band_structure_epoch = config.epoch
    config.verbose = False
    eigenvalues = jrystal.band_structure.train(config)
    lumo = jnp.min(eigenvalues[:, 6])
    homo = jnp.max(eigenvalues[:, 5])
    band_gap = (lumo - homo) * HARTREE2EV
    self.assertTrue(band_gap > 3.5 and band_gap < 4)


if __name__ == "__main__":
  absltest.main()
