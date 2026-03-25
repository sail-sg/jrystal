import argparse
from typing import Sequence

import yaml

import jrystal as jr

_LEGACY_OVERRIDE_PATHS = {
  "solver.type": "solver.mode",
  "solver.epoch": "solver.direct_opt.max_steps",
  "solver.optimizer": "solver.direct_opt.optimizer.name",
  "solver.scheduler": "solver.direct_opt.scheduler",
  "solver.scf_max_iter": "solver.scf.max_iter",
  "solver.scf_max_iteration": "solver.scf.max_iter",
  "solver.lobpcg_max_iter": "solver.scf.eigensolver.max_iter",
  "solver.mixing_beta": "solver.scf.mixing.beta",
  "solver.diis_max_hist": "solver.scf.mixing.history_size",
  "solver.convergence_window_size": "solver.direct_opt.convergence.window_size",
}


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="jrystal",
    description="JAX-based Differentiable DFT Framework",
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  commands = {
    "energy": "Ground-state energy using configured solver mode",
    "scf": "Ground-state energy via SCF",
    "direct-opt": "Ground-state energy via direct optimisation",
    "band": "Band structure calculation",
  }
  for name, help_text in commands.items():
    subparser = subparsers.add_parser(name, help=help_text)
    subparser.add_argument(
      "config",
      nargs="?",
      default="config.yaml",
      help="Path to YAML config file (default: config.yaml)",
    )

  return parser


def _parse_overrides(args: Sequence[str]) -> dict[str, str]:
  """Parse ``--key=value`` or ``--key value`` overrides."""
  overrides = {}
  i = 0
  while i < len(args):
    arg = args[i]
    if not arg.startswith("--"):
      raise ValueError(f"Unexpected argument: {arg}")

    token = arg[2:]
    if not token:
      raise ValueError("Empty override flag is not allowed.")

    if "=" in token:
      key, value = token.split("=", 1)
    else:
      if i + 1 >= len(args) or args[i + 1].startswith("--"):
        raise ValueError(f"Override '{arg}' requires a value.")
      key, value = token, args[i + 1]
      i += 1

    if not key:
      raise ValueError(f"Override '{arg}' is missing a key.")
    overrides[key] = value
    i += 1

  return overrides


def _apply_overrides(config, overrides: dict[str, str]) -> None:
  """Apply CLI overrides onto a nested config."""
  for key, raw_value in overrides.items():
    value = yaml.safe_load(raw_value)
    resolved_key = _LEGACY_OVERRIDE_PATHS.get(key, key)

    if key == "solver.convergence_condition":
      config.solver.direct_opt.convergence.energy_std_tol = value
      config.solver.scf.convergence.energy_tol = value
      continue

    if key == "solver.optimizer_args":
      if not isinstance(value, dict):
        raise TypeError(
          "Override 'solver.optimizer_args' must be a mapping."
        )
      for sub_key, sub_value in value.items():
        config.solver.direct_opt.optimizer[sub_key] = sub_value
      continue

    parts = resolved_key.split(".")
    target = config

    for part in parts[:-1]:
      if part not in target:
        raise KeyError(f"Unknown config override path: {key}")
      target = target[part]

    leaf = parts[-1]
    if leaf not in target:
      raise KeyError(f"Unknown config override field: {key}")
    target[leaf] = value


def _run_energy_command(config):
  """Run ground-state energy with the configured solver mode."""
  return jr.calc.energy(config)


def main(argv: Sequence[str] | None = None):
  path = jr.get_pkg_path()
  logo = open(path + "/jrystal_utf8.txt", "r").read()
  print(logo)

  parser = _build_parser()
  args, unknown = parser.parse_known_args(argv)

  try:
    overrides = _parse_overrides(unknown)
  except ValueError as exc:
    parser.error(str(exc))

  config = jr.config.get_config(args.config)

  try:
    _apply_overrides(config, overrides)
    jr.config.validate_config(config.to_dict())
  except (KeyError, TypeError, ValueError) as exc:
    parser.error(str(exc))

  if args.command == "scf":
    config.solver.mode = "scf"
    jr.calc.energy(config)
  elif args.command == "direct-opt":
    config.solver.mode = "direct_opt"
    jr.calc.energy(config)
  elif args.command == "energy":
    _run_energy_command(config)
  elif args.command == "band":
    ground_state_result = _run_energy_command(config)
    jr.calc.band(config, ground_state_result=ground_state_result)
