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
"""General utility functions."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float

from . import const


def safe_real(array: Array, tol: float = 1e-8) -> Array:
  """Return the real part of a nearly real-valued array.

  Args:
    array (Array): Input array, real or complex.
    tol (float): Absolute tolerance for the imaginary component.

  Returns:
    Array: Real-valued array when imaginary parts are negligible.

  Raises:
    ValueError: If the imaginary part exceeds ``tol``.
  """
  if jnp.iscomplexobj(array):
    if jnp.allclose(array.imag, 0, atol=tol):  # Adjust tolerance as needed
      return array.real
    else:
      raise ValueError("Array has non-zero imaginary part")
  return array


def vmapstack(times: int, args: List[Dict] = None) -> Callable:
  """Apply :func:`jax.vmap` repeatedly to a function.

  Args:
    times (int): Number of nested ``vmap`` applications.
    args (Optional[List[Dict]]): Optional keyword arguments per ``vmap`` call.

  Returns:
    Callable: Decorator that wraps a function with nested ``vmap``.

  Raises:
    ValueError: If ``len(args)`` does not equal ``times``.
  """

  def decorator(f):
    if args:
      if len(args) != times:
        raise ValueError(
          f'the length of args ({len(args)}) is not the same '
          f'of times ({times}).'
        )

    for i in range(times):
      if args:
        f = jax.vmap(f, **args[i])
      else:
        f = jax.vmap(f)
    return f

  return decorator


def absolute_square(array: Complex[Array, '...']) -> Float[Array, '...']:
  """Compute element-wise squared magnitude of a complex array.

  Args:
    array (Complex[Array, '...']): Complex-valued input.

  Returns:
    Float[Array, '...']: Real-valued :math:`|z|^2` for each element.
  """
  return jnp.real(jnp.conj(array) * array)


def volume(cell_vectors: Float[Array, '3 3']) -> Float:
  """Compute unit-cell volume from cell vectors.

  Args:
    cell_vectors (Float[Array, '3 3']): Cell vectors as a ``(3, 3)`` matrix.

  Returns:
    Float: Absolute determinant of ``cell_vectors``.
  """
  return jnp.abs(jnp.linalg.det(cell_vectors))


def wave_to_density(
  wave_grid: Complex[Array, 'spin kpt band x y z'],
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
  """Compute real-space density from wavefunctions.

  Args:
    wave_grid (Complex[Array, 'spin kpt band x y z']): Real-space wavefunctions.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
    Per-state density if ``occupation`` is ``None``; otherwise occupied
    spin-resolved density.

  Raises:
    ValueError: If ``wave_grid`` and ``occupation`` cannot be aligned.
  """
  dens = absolute_square(wave_grid)

  if occupation is not None:
    try:
      occupation = jnp.expand_dims(occupation, range(-3, 0))
      dens = jnp.sum(dens * occupation, axis=(1, 2))
    except:
      raise ValueError(
        f"wave_grid's shape ({wave_grid.shape}) and occupation's shape "
        f"({occupation.shape}) cannot align."
      )
  return dens


def wave_to_density_reciprocal(
  wave_grid: Complex[Array, 'spin kpt band x y z'],
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
  """Compute reciprocal-space density from real-space wavefunctions.

  Args:
    wave_grid (Complex[Array, 'spin kpt band x y z']): Real-space wavefunctions.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
    Reciprocal-space density tensor.
  """
  dens = wave_to_density(wave_grid, occupation)
  return jnp.fft.fftn(dens, axes=range(-3, 0))


def fft_factor(n: int) -> int:
  """Return the nearest supported FFT size greater than or equal to ``n``.

  Args:
    n (int): Minimum desired FFT size.

  Returns:
    int: Closest supported FFT size not smaller than ``n``.

  Raises:
    ValueError: If ``n`` is larger than the supported lookup range.
  """

  fftw_factors = np.array(const.CUFFT_FACTORS)
  if n > 2048:
    raise ValueError(f"The grid number {n} is too large!")
  delta_n = (fftw_factors - n) >= 0
  output = fftw_factors[delta_n][0]
  return output


def expand_coefficient(
  coeff_compact: Complex[Array, "spin kpt gpt band"],
  mask: Bool[Array, 'x y z'],
) -> Complex[Array, "spin kpt band x y z"]:
  """Expand masked coefficients to the full reciprocal grid.

  Args:
    coeff_compact (Complex[Array, "spin kpt gpt band"]): Compact coefficients.
    mask (Bool[Array, 'x y z']): Boolean mask of active grid points.

  Returns:
    Complex[Array, "spin kpt band x y z"]: Expanded coefficient tensor.
  """
  coeff_compact = jnp.swapaxes(coeff_compact, -1, -2)
  coeff_shape = coeff_compact.shape[:-1] + mask.shape
  return jnp.zeros(
    coeff_shape, dtype=coeff_compact.dtype
  ).at[..., mask].set(coeff_compact)


def squeeze_coefficient(
  coeff: Complex[Array, "spin kpt band x y z"],
  mask: Bool[Array, "spin kpt band x y z"],
) -> Complex[Array, "spin kpt gpt band"]:
  """Extract compact coefficients from a full reciprocal grid.

  Args:
    coeff (Complex[Array, "spin kpt band x y z"]): Full coefficient tensor.
    mask (Bool[Array, "spin kpt band x y z"]): Mask indicating selected entries.

  Returns:
    Complex[Array, "spin kpt gpt band"]: Compact coefficient tensor.
  """
  coeff_compact = coeff[..., mask].get()
  return jnp.swapaxes(coeff_compact, -1, -2)


def check_spin_number(num_electrons: int, spin: int) -> None:
  """Validate parity compatibility between electron count and spin.

  Args:
    num_electrons (int): Total number of electrons.
    spin (int): Number of unpaired electrons.

  Raises:
    ValueError: If ``num_electrons`` and ``spin`` have different parity.
  """
  if num_electrons % 2 != spin % 2:
    raise ValueError("spin number is not valid for the system. ")
