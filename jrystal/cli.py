import argparse
from pathlib import Path
from typing import Sequence

import yaml

import jrystal as jr
from jrystal.plot.band import _AUTO_REFERENCE as _AUTO_FERMI_REFERENCE
from jrystal.terminal_ui import console_line, render_logo

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

  plot_parser = subparsers.add_parser(
    "plot",
    help="Plot saved jrystal outputs",
  )
  plot_subparsers = plot_parser.add_subparsers(
    dest="plot_command",
    required=True,
  )

  plot_band_parser = plot_subparsers.add_parser(
    "band",
    help="Plot a band structure from a saved output directory",
  )
  plot_band_parser.add_argument(
    "source",
    help="Path to the jrystal output directory containing band/ data",
  )
  plot_band_parser.add_argument(
    "--output",
    "-o",
    help="Path to save the figure (default: <source>/band/band_structure.pdf)",
  )
  plot_band_parser.add_argument(
    "--unit",
    choices=("eV", "Ha", "Ry"),
    default="eV",
    help="Energy unit for the y-axis (default: eV)",
  )
  plot_band_parser.add_argument(
    "--ymin",
    type=float,
    default=None,
    help="Lower y-axis limit in the selected unit",
  )
  plot_band_parser.add_argument(
    "--ymax",
    type=float,
    default=None,
    help="Upper y-axis limit in the selected unit",
  )
  plot_band_parser.add_argument(
    "--fermi",
    nargs="?",
    const="auto",
    default="auto",
    help=(
      "Align to the saved Fermi/reference energy, or provide an explicit "
      "reference energy in Hartree."
    ),
  )
  plot_band_parser.add_argument(
    "--absolute-energy",
    action="store_true",
    help="Do not shift eigenvalues by the saved or supplied Fermi level",
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
        raise TypeError("Override 'solver.optimizer_args' must be a mapping.")
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


def _default_band_plot_path(source: str) -> Path:
  return Path(source) / "band" / "band_structure.pdf"


def _run_plot_band_command(args) -> None:
  if args.absolute_energy and args.fermi != "auto":
    raise ValueError("Use either `--fermi` or `--absolute-energy`, not both.")

  output_path = (
    Path(args.output)
    if args.output is not None else _default_band_plot_path(args.source)
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)

  if args.absolute_energy:
    reference_energy = None
  elif args.fermi == "auto":
    reference_energy = _AUTO_FERMI_REFERENCE
  else:
    reference_energy = float(args.fermi)

  fig = jr.plot.band_structure(
    args.source,
    reference_energy=reference_energy,
    unit=args.unit,
    y_min=args.ymin,
    y_max=args.ymax,
    save_path=output_path,
  )
  try:
    fig.clf()
  except Exception:
    pass
  console_line(f"Saved band plot to {output_path}")


def main(argv: Sequence[str] | None = None):
  render_logo(variant="utf8")

  parser = _build_parser()
  args, unknown = parser.parse_known_args(argv)

  if args.command == "plot":
    if unknown:
      parser.error(f"Unexpected arguments: {' '.join(unknown)}")
    try:
      if args.plot_command == "band":
        _run_plot_band_command(args)
        return
    except (ValueError, FileNotFoundError, ImportError) as exc:
      parser.error(str(exc))

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
    jr.calc.band(config)
