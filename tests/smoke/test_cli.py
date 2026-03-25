"""Smoke tests for the subcommand CLI."""

from pathlib import Path
import subprocess
import sys
from unittest import mock

from absl.testing import absltest

import main
from jrystal.config import get_config


class CliHelpersTest(absltest.TestCase):

  def test_parser_uses_subcommands_and_default_config(self):
    parser = main._build_parser()
    args, unknown = parser.parse_known_args(["energy"])

    self.assertEqual(args.command, "energy")
    self.assertEqual(args.config, "config.yaml")
    self.assertEqual(unknown, [])

  def test_parse_overrides_supports_equals_and_space_syntax(self):
    overrides = main._parse_overrides([
      "--basis.cutoff_energy=200",
      "--solver.direct_opt.max_steps",
      "500",
    ])

    self.assertEqual(
      overrides,
      {
        "basis.cutoff_energy": "200",
        "solver.direct_opt.max_steps": "500",
      },
    )

  def test_apply_overrides_updates_nested_config_and_legacy_aliases(self):
    config = get_config()
    main._apply_overrides(
      config,
      {
        "basis.cutoff_energy": "200",
        "solver.epoch": "500",
        "solver.mode": "scf",
        "execution.verbose": "false",
      },
    )

    self.assertEqual(config.basis.cutoff_energy, 200)
    self.assertEqual(config.solver.direct_opt.max_steps, 500)
    self.assertEqual(config.solver.mode, "scf")
    self.assertFalse(config.execution.verbose)

  def test_energy_command_delegates_to_public_api(self):
    config = get_config()
    direct_opt_result = mock.Mock(converged=True)

    with mock.patch.object(
      main.jr.calc,
      "energy",
      return_value=direct_opt_result,
    ) as energy_mock:
      result = main._run_energy_command(config)

    self.assertIs(result, direct_opt_result)
    energy_mock.assert_called_once()


class CliSubcommandHelpTest(absltest.TestCase):

  def test_energy_subcommand_help(self):
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
      [sys.executable, str(repo_root / "main.py"), "energy", "--help"],
      capture_output=True,
      text=True,
      check=False,
    )

    self.assertEqual(result.returncode, 0, msg=result.stderr)
    self.assertIn("usage:", result.stdout)
    self.assertIn("config", result.stdout)


if __name__ == "__main__":
  absltest.main()
