import argparse

import jrystal as jr


def main():
  path = jr.get_pkg_path()
  logo = open(path + '/jrystal_utf8.txt', 'r').read()
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

  args = parser.parse_args()

  config = jr.config.get_config(args.config)

  if args.mode == "energy":
    jr.calc.energy(config)
  elif args.mode == "band":
    jr.calc.band(config)


if __name__ == "__main__":
  main()
