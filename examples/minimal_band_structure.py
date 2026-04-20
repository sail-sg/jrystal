import argparse

import jrystal as jr


def main():
  parser = argparse.ArgumentParser(
    description="Run a minimal Jrystal band-structure example."
  )
  parser.add_argument(
    "--config",
    default="config.yaml",
    help="Path to a Jrystal config file.",
  )
  args = parser.parse_args()

  config = jr.config.get_config(args.config)
  ground_state = jr.calc.energy(config)
  result = jr.calc.band(config, ground_state_result=ground_state)

  print(f"Ground-state energy: {result.ground_state_energy:.6f} Ha")
  print(f"Eigenvalue array shape: {result.eigenvalues.shape}")


if __name__ == "__main__":
  main()
