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
"""Entropy utilities."""
import jax.numpy as jnp
from jaxtyping import Array, Float


def von_neumann(
  occupation: Float[Array, 'spin kpt band'], eps: float = 1e-8
) -> Float:
  r"""Compute the entropy term for Fermi-Dirac occupations.

  The entropy term is given by:
  .. math::
    S = - \sum_{i,j,\boldsymbol{k}} f_{i,j}(\boldsymbol{k})
      \log(f_{i,j}(\boldsymbol{k})) +
      + (1 - f_{i,j}(\boldsymbol{k})) \log((1 - f_{i,j}(\boldsymbol{k})))

  Args:
    occupation (Float[Array, 'spin kpt band']): Occupation numbers.
    eps (float): Numerical stability constant used in logarithms.

  Returns:
    Float: Entropy contribution for the provided occupations.
  """
  num_spin = occupation.shape[0]

  entropy = -jnp.sum(
    occupation * jnp.log(eps + occupation) +
    ((3 - num_spin) - occupation) * jnp.log(eps + (3 - num_spin) - occupation)
  )

  return entropy


def renyi(
  occupation: Float[Array, 'spin kpt band'],
  alpha: float,
) -> Float:
  r"""Compute the entropy term for renyi occupations.

  Args:
    occupation (Float[Array, 'spin kpt band']): Occupation numbers.
    alpha (float): Renyi index.

  Returns:
    Float: Entropy contribution for the provided occupations.
  """
  num_spin = occupation.shape[0]

  assert alpha >= 0 and alpha <= 1, "alpha must be between 0 and 1"
  entropy = -jnp.sum(
    occupation *
    jnp.log(occupation**alpha +
            ((3 - num_spin) - occupation)**alpha) / (1 - alpha)
  )

  return entropy
