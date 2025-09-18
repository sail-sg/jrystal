import argparse

import jrystal as jr


def main():
  logo = open('jrystal_logo_ascii.txt', 'r').read()
  print(logo)

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

  parser.add_argument(
    "-l",
    "--load",
    help="Load pickled output from energy calculation for band structure calculation."
  )

  args = parser.parse_args()

  config = jr.config.get_config(args.config)

  if args.mode == "energy":
    if config.use_pseudopotential:
      jr.calc.energy_normcons(config)
    else:
      jr.calc.energy_all_electrons(config)

  elif args.mode == "band":
    if config.use_pseudopotential:
      jr.calc.band_normcons(config)
    else:
      jr.calc.band_all_electrons(config)


if __name__ == "__main__":
  main()
