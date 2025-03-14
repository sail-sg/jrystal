# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Entropy functions. """
import jax.numpy as jnp
from jaxtyping import Float, Array


def fermi_dirac(
  occupation: Float[Array, 'spin kpt band'], eps: float = 1e-8
) -> Float:
  r"""Compute the entropy corresponding to Fermi-Dirac distribution.

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
      Float: The entropy value corresponding to the Fermi-Dirac distribution.
  """
  num_spin, num_k, _ = occupation.shape

  entropy = -jnp.sum(
    occupation * jnp.log(eps + occupation) +
    ((3 - num_spin) / num_k - occupation) *
    jnp.log(eps + (3 - num_spin) / num_k - occupation)
  )

  return entropy
