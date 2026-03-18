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
"""Inner products and expectation values in real and reciprocal space."""

from typing import Optional, Union

import einops
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float


def reciprocal_braket(
  bra: Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']],
  ket: Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']],
  vol: Float,
) -> Float:
  r"""Compute :math:`\langle \mathrm{bra} | \mathrm{ket} \rangle` in
    reciprocal space.

  Args:
    bra (Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']]):
      Left operand in reciprocal space.
    ket (Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']]):
      Right operand in reciprocal space.
    vol (Float): Unit-cell volume.

  Returns:
    Float: Real-valued reciprocal-space inner product.

  Raises:
    ValueError: If ``bra`` and ``ket`` grid shapes do not match.
  """
  if bra.shape[-3:] != ket.shape[-3:]:
    raise ValueError(
      f"bra and ket shape are not aligned. Got "
      f"{bra.shape} and {ket.shape}."
    )

  num_grids = np.prod(np.array(bra.shape))
  # Parseval's theorem
  parseval_factor = 1 / num_grids
  # numerical integration weights
  numerical_integral_weight = vol / num_grids
  product = jnp.sum(
    jnp.conj(bra) * ket
  ) * parseval_factor * numerical_integral_weight
  return product.real


def real_braket(
  bra: Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']],
  ket: Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']],
  vol: Float,
) -> Float:
  r"""Compute :math:`\langle \mathrm{bra} | \mathrm{ket} \rangle` in real space.

  Args:
    bra (Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']]):
      Left operand in real space.
    ket (Union[Complex[Array, '*n x y z'], Float[Array, '*n x y z']]):
      Right operand in real space.
    vol (Float): Unit-cell volume.

  Returns:
    Float: Real-space inner product.

  Raises:
    ValueError: If ``bra`` and ``ket`` shapes do not match.
  """
  if bra.shape != ket.shape:
    raise ValueError(
      f"bra and ket shape are not aligned. Got "
      f"{bra.shape} and {ket.shape}."
    )

  num_grids = np.prod(np.array(bra.shape))
  numerical_integral_weight = vol / num_grids
  product = jnp.sum(bra * ket) * numerical_integral_weight
  return product


def expectation(
  bra: Union[Complex[Array, 'spin kpt band x y z'],
             Float[Array, 'spin kpt band x y z']],
  hamiltonian: Union[Complex[Array, 'spin kpt band x y z'],
                     Float[Array, 'spin kpt band x y z']],
  vol: Float,
  ket: Optional[Union[Complex[Array, 'spin kpt band x y z'],
                      Float[Array, 'spin kpt band x y z']]] = None,
  diagonal: bool = False,
  mode: str = 'real'
) -> Array:
  r"""Compute expectation values
    :math:`\langle \psi_i | \hat{H} | \psi_j \rangle`.

  Args:
    bra (Union[Complex[Array, 'spin kpt band x y z'],
      Float[Array, 'spin kpt band x y z']]): Left wavefunctions.
    hamiltonian (Union[Complex[Array, 'spin kpt band x y z'],
      Float[Array, 'spin kpt band x y z']]): Hamiltonian values on the same
      grid. A :math:`k`-dependent tensor with shape ``(kpt, x, y, z)`` is also
      supported.
    vol (Float): Unit-cell volume.
    ket (Optional[Union[Complex[Array, 'spin kpt band x y z'],
      Float[Array, 'spin kpt band x y z']]]): Right wavefunctions.
      If ``None``, ``bra`` is used.
    diagonal (bool): If ``True``, compute only diagonal elements.
    mode (str): Normalization mode. Must be ``'real'``, ``'reciprocal'``, or
      ``'kinetic'``.

  Returns:
    Array: Expectation values. Shape is ``(spin, kpt, band)`` when
    ``diagonal=True`` and ``(spin, kpt, band, band)`` otherwise.
  """
  ket = bra if ket is None else ket
  assert bra.ndim == 6
  assert hamiltonian.ndim in [3, 4]
  num_grids = np.prod(bra.shape[-3:])

  if mode == 'reciprocal':
    # Parseval's theorem
    parseval_factor = 1 / num_grids
    integral_factor = vol / num_grids * parseval_factor

  elif mode == 'real':
    integral_factor = vol / num_grids

  elif mode == 'kinetic':
    integral_factor = 1.

  else:
    raise ValueError(
      'Argument \'mode\' must be one of \'real\', \'reciprocal\', or',
      f'\'kinetic\'. Got {mode}'
    )

  if hamiltonian.ndim == 3:
    if diagonal is False:
      p = "a nk ni1 x y z, x y z, a nk ni2 x y z -> a nk ni1 ni2"
    else:
      p = "a nk ni x y z, x y z, a nk ni x y z -> a nk ni"

  elif hamiltonian.ndim == 4:
    if diagonal is False:
      p = "a nk ni1 x y z, nk x y z, a nk ni2 x y z -> a nk ni1 ni2"
    else:
      p = "a nk ni x y z, nk x y z, a nk ni x y z -> a nk ni"

  else:
    raise ValueError(
      "Hamitonian array must have 3 or 4 dimensions",
      f"(with k-point channel). Given {hamiltonian.ndim} dimensions."
    )
  output = einops.einsum(jnp.conj(bra), hamiltonian, ket, p) * integral_factor
  return output
