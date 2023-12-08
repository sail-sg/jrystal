import jrystal
from absl.testing import absltest


class Test_total_energy_diamond(absltest.TestCase):

  def test_diamond(self):
    config = jrystal.config.get_config()
    config.crystal = "diamond"
    config.grid_sizes = 24
    config.epoch = 2000
    total_energy = jrystal.total_energy.train(config, return_fn="total_energy")
    ewald = jrystal.training_utils.get_ewald_coulomb_repulsion(config)
    crystal = jrystal.training_utils.create_crystal(config)
    energy = total_energy(crystal)[0] + ewald
    self.assertTrue(
      energy > -65. and energy < -55.
    )


if __name__ == "__main__":
  absltest.main()
