import jrystal
from absl.testing import absltest


class Test_energy_gap_diamond(absltest.TestCase):

  def test_diamond(self):
    config = jrystal.config.get_config()
    config.crystal = "diamond1"
    config.grid_sizes = 32
    config.epoch = 5000
    config.verbose = False

    total_energy = jrystal.total_energy.train(config, return_fn="total_energy")
    ewald = jrystal.training_utils.get_ewald_coulomb_repulsion(config)
    crystal = jrystal.training_utils.create_crystal(config)
    energy1 = total_energy(crystal)[0] + ewald

    config.crystal = "diamond2"
    total_energy = jrystal.total_energy.train(config, return_fn="total_energy")
    crystal = jrystal.training_utils.create_crystal(config)
    ewald = jrystal.training_utils.get_ewald_coulomb_repulsion(config)
    energy2 = total_energy(crystal)[0] + ewald

    energy_gap = energy1 - energy2
    self.assertTrue(energy_gap < 0.2 and energy_gap > 0.18)


if __name__ == "__main__":
  absltest.main()
