""" Entropy functions. """
import jax.numpy as jnp
from ._typing import OccupationArray


def fermi_dirac(occupation: OccupationArray, eps: float = 1e-8) -> float:
  r"""Compute the entropy corresponding to fermi-dirac distribution.
  
  The entropy is defined as:
    
  $$
    -\sum_{\sigma, k, i} o_{\sigma, k, i} \log(o_{\sigma, k, i}) + 
      (1-o_{\sigma, k, i}) \log(1-o_{\sigma, k, i})
  $$

  where
    $o_{\sigma, k, i}$ is the occupation number, and index $\sigma$ is the spin 
    index, $k$ is the K point index, and  $i$ is the band index.
      
  Args:
      occupation (OccupationArray): The occupation numbers.
        Shape: (num_spin, num_k, num_bands)
      eps (float, optional): machine epsilon. Defaults to 1e-8.

  Returns:
      float: the entropy that corresponds to the fermi-dirac distribution.
  """
  num_spin, num_k, _ = occupation.shape

  entropy = -jnp.sum(
    occupation * jnp.log(eps + occupation) +
    ((3 - num_spin) / num_k - occupation) *
    jnp.log(eps + (3 - num_spin) / num_k - occupation)
  )

  return entropy
