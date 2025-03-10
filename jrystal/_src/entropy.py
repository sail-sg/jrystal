""" Entropy functions. """
import jax.numpy as jnp
from jaxtyping import Float, Array


def fermi_dirac(occupation: Float[Array, 'spin kpt band'], eps: float = 1e-8) -> Float:
  r"""Compute the entropy corresponding to fermi-dirac distribution.

  The entropy is defined as:

  .. math::

      -\sum_{\sigma, k, i} [o_{\sigma, k, i} \log(o_{\sigma, k, i} + \epsilon) + 
      (1-o_{\sigma, k, i}) \log(1-o_{\sigma, k, i} + \epsilon)]

  where 

  - :math:`o_{\sigma, k, i}` is the occupation number
  - :math:`\sigma` is the spin index
  - :math:`k` is the :math:`k`-point index
  - :math:`i` is the band index
  - :math:`\epsilon` is the machine epsilon to prevent numerical instabilities.

  Args:
      occupation(Float[Array, 'spin kpt band']): The occupation numbers with shape (spin, kpt, band).
      eps(float): Machine epsilon to prevent numerical instabilities. Default: 1e-8

  Returns:
      Float: The entropy value corresponding to the fermi-dirac distribution.
  """
  num_spin, num_k, _ = occupation.shape

  entropy = -jnp.sum(
    occupation * jnp.log(eps + occupation) +
    ((3 - num_spin) / num_k - occupation) *
    jnp.log(eps + (3 - num_spin) / num_k - occupation)
  )

  return entropy
