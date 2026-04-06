"""Shared plotting constants."""

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_RY = 2.0


def energy_scale(unit: str) -> float:
  if unit == "Ha":
    return 1.0
  if unit == "eV":
    return HARTREE_TO_EV
  if unit == "Ry":
    return HARTREE_TO_RY
  raise ValueError(f"Unsupported energy unit: {unit}")
