"""Smoke tests for config loading and migration."""
from pathlib import Path

from absl.testing import absltest

from jrystal.config import _migrate_flat_config, get_config


class ConfigTest(absltest.TestCase):

  def test_default_config_loads(self):
    config = get_config()
    self.assertEqual(config.schema_version, 1)
    self.assertEqual(config.system.crystal, "diamond")
    self.assertEqual(config.method.pseudopotential_type, "nc")

  def test_legacy_config_migrates(self):
    legacy_config = {
      "crystal": "si",
      "crystal_file_path_path": "/tmp/si.xyz",
      "spin": 2,
      "xc": "lda_x",
      "use_pseudopotential": True,
      "pseudopotential_type": "nc",
      "grid_sizes": 32,
      "k_grid_sizes": [2, 2, 2],
      "empty_bands": 12,
      "band_structure_empty_bands": None,
      "ewald_args": {
        "ewald_eta": 0.2,
        "ewald_cutoff": 1000.0,
      },
      "parallel_over_k_mesh": True,
      "seed": 7,
    }

    migrated = _migrate_flat_config(legacy_config)

    self.assertEqual(migrated["schema_version"], 1)
    self.assertEqual(migrated["system"]["crystal"], "si")
    self.assertEqual(migrated["system"]["crystal_file_path"], "/tmp/si.xyz")
    self.assertEqual(migrated["method"]["pseudopotential_type"], "nc")
    self.assertEqual(migrated["basis"]["grid_sizes"], 32)
    self.assertEqual(migrated["ksampling"]["k_grid_sizes"], [2, 2, 2])
    self.assertEqual(migrated["occupation"]["empty_bands"], 12)
    self.assertEqual(migrated["band"]["empty_bands"], 12)
    self.assertEqual(migrated["ewald"]["eta"], 0.2)
    self.assertEqual(migrated["ewald"]["cutoff"], 1000.0)
    self.assertTrue(migrated["execution"]["parallel_over_k_mesh"])
    self.assertEqual(migrated["execution"]["seed"], 7)

  def test_yaml_config_loads(self):
    repo_root = Path(__file__).resolve().parents[2]
    config = get_config(str(repo_root / "config.yaml"))
    self.assertEqual(config.schema_version, 1)
    self.assertEqual(config.system.crystal, "diamond")
    self.assertEqual(config.band.empty_bands, config.occupation.empty_bands)


if __name__ == "__main__":
  absltest.main()
