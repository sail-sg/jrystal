"""Smoke tests for config loading and migration."""
from pathlib import Path
import tempfile

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
      "type": "scf",
      "epoch": 200,
      "optimizer": "adam",
      "optimizer_args": {"learning_rate": 0.05},
      "convergence_condition": 1.0e-5,
      "scf_max_iter": 12,
      "lobpcg_max_iter": 9,
      "mixing_beta": 0.7,
      "diis_max_hist": 6,
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
    self.assertEqual(migrated["solver"]["mode"], "scf")
    self.assertEqual(migrated["solver"]["direct_opt"]["max_steps"], 200)
    self.assertEqual(
      migrated["solver"]["direct_opt"]["optimizer"]["name"],
      "adam",
    )
    self.assertEqual(
      migrated["solver"]["direct_opt"]["optimizer"]["learning_rate"],
      0.05,
    )
    self.assertEqual(
      migrated["solver"]["direct_opt"]["convergence"]["energy_std_tol"],
      1.0e-5,
    )
    self.assertEqual(migrated["solver"]["scf"]["max_iter"], 12)
    self.assertEqual(
      migrated["solver"]["scf"]["eigensolver"]["max_iter"],
      9,
    )
    self.assertEqual(migrated["solver"]["scf"]["mixing"]["beta"], 0.7)
    self.assertEqual(
      migrated["solver"]["scf"]["mixing"]["history_size"],
      6,
    )
    self.assertEqual(
      migrated["solver"]["scf"]["convergence"]["energy_tol"],
      1.0e-5,
    )
    self.assertTrue(migrated["execution"]["parallel_over_k_mesh"])
    self.assertEqual(migrated["execution"]["seed"], 7)

  def test_yaml_config_loads(self):
    repo_root = Path(__file__).resolve().parents[2]
    config = get_config(str(repo_root / "config.yaml"))
    self.assertEqual(config.schema_version, 1)
    self.assertEqual(config.system.crystal, "diamond")
    self.assertEqual(config.band.empty_bands, config.occupation.empty_bands)
    self.assertEqual(config.solver.mode, "auto")
    self.assertEqual(config.solver.auto.primary, "scf")
    self.assertEqual(config.solver.scf.max_iter, 100)
    self.assertEqual(config.solver.scf.eigensolver.max_iter, 6)
    self.assertEqual(config.solver.scf.mixing.history_size, 8)
    self.assertEqual(config.solver.scf.mixing.beta, 0.8)
    self.assertEqual(config.solver.direct_opt.max_steps, 10000)

  def test_solver_scf_max_iteration_alias_loads(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      config_path = Path(tmp_dir) / "config.yaml"
      config_path.write_text(
        "\n".join([
          "schema_version: 1",
          "solver:",
          "  type: scf",
          "  scf_max_iteration: 7",
          "  convergence_condition: 1.0e-5",
          "",
        ]),
        encoding="utf-8",
      )

      config = get_config(str(config_path))

    self.assertEqual(config.solver.mode, "scf")
    self.assertEqual(config.solver.scf.max_iter, 7)
    self.assertEqual(config.solver.scf.convergence.energy_tol, 1.0e-5)
    self.assertEqual(
      config.solver.direct_opt.convergence.energy_std_tol,
      1.0e-5,
    )

  def test_solver_diis_fields_load(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
      config_path = Path(tmp_dir) / "config.yaml"
      config_path.write_text(
        "\n".join([
          "schema_version: 1",
          "solver:",
          "  scf:",
          "    mixing:",
          "      history_size: 12",
          "      beta: 0.65",
          "    eigensolver:",
          "      max_iter: 9",
          "",
        ]),
        encoding="utf-8",
      )

      config = get_config(str(config_path))

    self.assertEqual(config.solver.scf.mixing.history_size, 12)
    self.assertEqual(config.solver.scf.mixing.beta, 0.65)
    self.assertEqual(config.solver.scf.eigensolver.max_iter, 9)


if __name__ == "__main__":
  absltest.main()
