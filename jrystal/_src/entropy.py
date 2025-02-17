""" Entropy functions. """
import jax.numpy as jnp
from .typing import OccupationArray


def fermi_dirac(occupation: OccupationArray, eps: float = 1e-8) -> float:
  """the entropy corresponding to fermi-dirac distribution.

  Args:
      occupation (OccupationArray): The occupation numbers.
          Shape: [num_spin, num_k, num_bands]
      eps (float, optional): machine epsilon. Defaults to 1e-8.

  Returns:
      float: _description_
  """
  num_spin, num_k, _ = occupation.shape

  entropy = -jnp.sum(
    occupation * jnp.log(eps + occupation) +
    ((3 - num_spin) / num_k - occupation) *
    jnp.log(eps + (3 - num_spin) / num_k - occupation)
  )

  return entropy
