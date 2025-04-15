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
"""Integration operations for quantum mechanical calculations in real and reciprocal space.

This module provides functions for calculating inner products (brakets) and expectation values in both real and reciprocal space, which are fundamental operations in quantum mechanics and density functional theory (DFT) calculations.
"""

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
  r"""Calculate the inner product of two functions in reciprocal space.

  Computes the inner product between two wavefunctions in reciprocal space using:

  .. math::

    \langle f|g \rangle \approx \sum_{\mathbf{G}} f^*(\mathbf{G})g(\mathbf{G}) \frac{vol}{N^2}

  where :math:`f^*(\mathbf{G})` is the complex conjugate of :math:`f(\mathbf{G})`, :math:`N` is the number of grid points, and :math:`vol` is the real-space unit cell volume.

  .. NOTE::
    This formulation is particularly useful for calculating hartree and external energy integrals in reciprocal space, as they often have simpler forms than in real space.

  Args:
      bra (Complex[Array, '*n x y z'] | Float[Array, '*n x y z']): Wavefunction in reciprocal space (left side of braket).
           Shape must match ket's shape.
      ket (Complex[Array, '*n x y z'] | Float[Array, '*n x y z']): Wavefunction in reciprocal space (right side of braket).
           Shape must match bra's shape.
      vol (Float): Volume of the real-space unit cell.

  Returns:
      Float: The real-valued inner product result.

  Raises:
      ValueError: If bra and ket shapes do not match.
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
  r"""Calculate the inner product of two functions in real space.

  Computes the inner product between two wavefunctions in real space using:

  .. math::

    \langle f|g \rangle \approx \sum_{\mathbf{r}} f^*(\mathbf{r})g(\mathbf{r}) \frac{vol}{N}

  where :math:`f^*(\mathbf{r})` is the complex conjugate of :math:`f(\mathbf{r})`, :math:`N` is the number of grid points, and :math:`vol` is the unit cell volume.

  .. NOTE::

    This formulation is commonly used in planewave DFT for calculating exchange-correlation energy integrals, which are typically evaluated in real space for efficiency.

  Args:
      bra (Complex[Array, '*n x y z'] | Float[Array, '*n x y z']): Wavefunction in real space (left side of braket). Shape must match ket's shape.
      ket (Complex[Array, '*n x y z'] | Float[Array, '*n x y z']): Wavefunction in real space (right side of braket). Shape must match bra's shape.
      vol (Float): Volume of the unit cell.

  Returns:
      Float: The real-valued inner product result.

  Raises:
      ValueError: If bra and ket shapes do not match.
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
  r"""Calculate the expectation value of a Hamiltonian operator.

  Computes matrix elements of the form:

  .. math::

    E_{ij} = \langle \psi_i | \hat{H} | \psi_j \rangle \approx \sum_{\mathbf{q}} \psi_i^*(\mathbf{q}) \hat{H}(\mathbf{q}) \psi_j(\mathbf{q}) \frac{vol}{N^p}

  where :math:`\psi_i` and :math:`\psi_j` are wavefunctions, :math:`\hat{H}` is the Hamiltonian operator, :math:`\mathbf{q}` represents either real (:math:`\mathbf{r}`) or reciprocal (:math:`\mathbf{G}`) space coordinates, and :math:`p` depends on the mode:

  - For real space: :math:`p = 1`
  - For reciprocal space: :math:`p = 2` (includes Parseval factor)
  - For kinetic terms: :math:`p = 0`

  For more details on expectation values in quantum mechanics, see:
  https://en.wikipedia.org/wiki/Expectation_value_(quantum_mechanics)

  Args:
      bra (Complex[Array, 'spin kpt band x y z'] | Float[Array, 'spin kpt band x y z']): Left wavefunction in the expectation value calculation.
           Must have shape (spin, kpt, band, x, y, z).
      hamiltonian (Complex[Array, 'spin kpt band x y z'] | Float[Array, 'spin kpt band x y z']): Hamiltonian operator matrix.
           Must have shape (spin, kpt, band, x, y, z).
      vol (Float): Volume of the unit cell.
      ket (Complex[Array, 'spin kpt band x y z'] | Float[Array, 'spin kpt band x y z']): Right wavefunction. If None, uses the bra wavefunction (for diagonal elements).
           Must have same shape as bra if provided.
      diagonal (bool): If True, only compute diagonal elements :math:`E_{ii}`.
           If False, compute full matrix :math:`E_{ij}`.
      mode (str): Integration mode determining the normalization factors:
          - 'real': Real space integration (:math:`vol/N`)
          - 'reciprocal': Reciprocal space (includes :math:`1/N` Parseval factor)
          - 'kinetic': Special case with unit factor

  Returns:
      Array: Array of expectation values. Shape depends on diagonal parameter:

      - If diagonal=True: shape (spin, kpt, band)
      - If diagonal=False: shape (spin, kpt, band, band)

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
