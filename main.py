import argparse
import yaml
import jrystal as jr
from jrystal.config import JrystalConfigDict

parser = argparse.ArgumentParser(
  prog='Jrystal', description='Command for Jrystal package.'
)

parser.add_argument(
  "-m",
  "--mode",
  choices=["energy", "band"],
  default='energy',
  help="Set the computation mode. For total enrgy minimization, please use "
  "\'energy\'. For band structure calculation, please use \'band\'. "
)

parser.add_argument(
  "-c",
  "--config",
  default='config.yaml',
  help="Set the configuration file path."
)

args = parser.parse_args()

config = jr.config.get_config("config.yaml")

if args.mode == "energy":
  if config.use_pseudopotential:
    jr.calc.energy_ncpp(config)
  else:
    jr.calc.energy(config)
elif args.mode == "band":
  if config.use_pseudopotential:
    jr.calc.band_ncpp(config)
  else:
    jr.calc.band(config)

