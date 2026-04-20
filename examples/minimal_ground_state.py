import argparse

import jrystal as jr


def main():
  parser = argparse.ArgumentParser(
    description="Run a minimal Jrystal ground-state example."
  )
  parser.add_argument(
    "--config",
    default="config.yaml",
    help="Path to a Jrystal config file.",
  )
  args = parser.parse_args()

  config = jr.config.get_config(args.config)
  result = jr.calc.energy(config)

  print(f"Total energy: {result.total_energy:.6f} Ha")
  print(f"Converged: {result.converged}")
  print(f"Kinetic energy: {result.energy_terms.kinetic:.6f} Ha")


if __name__ == "__main__":
  main()
