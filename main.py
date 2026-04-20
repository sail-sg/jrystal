from jrystal import cli as _cli
from jrystal.cli import main

jr = _cli.jr
_build_parser = _cli._build_parser
_parse_overrides = _cli._parse_overrides
_apply_overrides = _cli._apply_overrides
_run_energy_command = _cli._run_energy_command

if __name__ == "__main__":
  main()
