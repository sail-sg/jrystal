import jrystal
from ml_collections import ConfigDict
import argparse
import yaml

parser = argparse.ArgumentParser(
  prog='Jrystal', description='Interface for jrystal package.'
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
  "--config",
  type=str,
  help="Path of configuration file.",
  default="config.yaml"
)

parser.add_argument("-g", "--grid", type=int, help="Grid sizes", default=None)
parser.add_argument(
  "-e", "--epoch", type=int, help="Epoch number", default=None
)
parser.add_argument(
  "--grid_mask",
  type=str,
  choices=['spherical', 'cubic'],
  help="method for generating G grid mask",
  default='cubic'
)

args = parser.parse_args()
config_file = args.config

with open(config_file, 'r') as file:
  config = yaml.safe_load(file)

config = ConfigDict(config)

if args.grid is not None:
  config.grid_sizes = args.grid

if args.epoch is not None:
  config.epoch = args.epoch

config.g_grid_mask_method = args.grid_mask

if args.mode == "energy":
  jrystal.total_energy.train(config)

if args.mode == "band":
  jrystal.band_structure.train(config)
