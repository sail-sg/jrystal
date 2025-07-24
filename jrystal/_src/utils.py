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
"""Utility functions."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float

from . import const


def safe_real(array: Array, tol: float = 1e-8) -> Array:
  """Safely converts a complex array to real by checking imaginary components.

    Attempts to convert a complex array to real by verifying that all imaginary
    components are effectively zero (within specified tolerance). This is useful
    for numerical computations where results should be real but may have tiny
    imaginary components due to floating point errors.

    Args:
        array (Array): Input array that may be real or complex.
        tol (float): Tolerance threshold for considering imaginary components as zero. Defaults to 1e-8.

    Returns:
        Array: The real component of the input if imaginary parts are within
            tolerance, otherwise the original array.

    Raises:
        ValueError: If the array has imaginary components larger than the
            specified tolerance.

    Example:

    .. code-block:: python

      x = 1.0 + 1e-10j
      safe_real(x)  # Returns 1.0
      y = 1.0 + 1.0j
      safe_real(y)  # Raises ValueError
    """
  if jnp.iscomplexobj(array):
    if jnp.allclose(array.imag, 0, atol=tol):  # Adjust tolerance as needed
      return array.real
    else:
      raise ValueError("Array has non-zero imaginary part")
  return array


def vmapstack(times: int, args: List[Dict] = None) -> Callable:
  """Recursively applies JAX's vmap function to vectorize operations over multiple dimensions.

    Creates a decorator that applies JAX's vmap transformation multiple times to a function, enabling vectorized operations over multiple batch dimensions. This is particularly useful for handling multi-dimensional batch processing in neural network operations.

    Args:
        times (int): Number of vmap applications. Should match the number of batch dimensions to be processed.
        args (List[Dict]): Optional list of dictionaries containing vmap configuration for each application. Each dictionary can contain standard vmap arguments like in_axes, out_axes, axis_size, etc. Defaults to None.

    Returns:
        Callable: A decorator function that transforms the input function by applying vmap the specified number of times.

    Raises:
        ValueError: If the length of args does not match the specified number of
            vmap applications (times).

    Example:

    .. code-block:: python

      @vmapstack(2)
      def f(x):
          return x * 2
      # f can now handle 2 batch dimensions
      x = jnp.ones((3, 4, 5))  # 2 batch dims (3,4) with input dim 5
      result = f(x)  # Shape will be (3, 4, 5)
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
  """Computes the squared magnitude of complex numbers in an array.

    Calculates :math:`|z|^2` for each complex number :math:`z` in the input array by multiplying each element with its complex conjugate. This operation preserves the array shape while converting complex values to their real squared magnitudes.

    .. note::
        This is equivalent to :math:`(Re(z))^2 + (Im(z))^2` for each complex number :math:`z`, but is computed using complex conjugate multiplication for better numerical stability.

    Example:

    .. code-block:: python

      x = 3 + 4j
      absolute_square(x)  # Returns 25.0 (|3 + 4j|² = 3² + 4² = 25)

    Args:
        array (Complex[Array, '...'] ): Complex-valued array of any shape. The '...' notation indicates arbitrary dimensions are supported.

    Returns:
        Real-valued array of the same shape as input, containing the squared
        magnitudes of the complex values.
    """
  return jnp.real(jnp.conj(array) * array)


def volume(cell_vectors: Float[Array, '3 3']) -> Float:
  """Calculates the volume of a parallelepiped defined by three cell vectors.

    Computes the volume of a unit cell in a crystal structure by calculating the
    determinant of the matrix formed by the three cell vectors. The absolute value of the determinant gives the volume of the parallelepiped.

    .. note::
        The volume is calculated as :math:`|det(A)|` where :math:`A` is the matrix of cell vectors. This gives the volume of the parallelepiped formed by the three vectors regardless of their orientation.

    Example:

    .. code-block:: python

      # For a cubic cell of side length 2
      vectors = jnp.array([[2., 0., 0.],
                           [0., 2., 0.],
                           [0., 0., 2.]])
      volume(vectors)  # Returns 8.0

    Args:
        cell_vectors (Float[Array, '3 3']): A 3x3 matrix where each row represents a cell vector of the crystal structure. The vectors should be given in consistent units (e.g., Bohr radii or Angstroms).

    Returns:
        Float: The volume of the unit cell (in cubic units of the input vectors).
    """
  return jnp.abs(jnp.linalg.det(cell_vectors))


def wave_to_density(
  wave_grid: Complex[Array, 'spin kpt band x y z'],
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
  """Computes electron density from wave functions in real space.

    Calculates the electron density by taking the absolute square of wave functions and optionally applying occupation numbers. The density can be computed for the full grid or reduced along specified dimensions.

    Args:
      wave_grid (Complex[Array, 'spin kpt band x y z']): Complex wave function
        values on a real-space grid. The array has dimensions for spin,
        k-points, bands, and spatial coordinates (x,y,z).
      occupation (Optional[Float[Array, 'spin kpt band']]): Optional occupation
        numbers for each state (spin, k-point, band). If provided, the density
        will be weighted by these values. Defaults to None.

    Returns:
      Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
        The electron density grid. If occupation is None, the density grid has
        the same shape as the input wave_grid. If occupation is provided, the
        density grid is reduced over the k-points and bands dimensions.

    Raises:
        ValueError: If the shapes of wave_grid and occupation are incompatible for broadcasting.
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
  """Computes electron density from wave functions in reciprocal space.

    Calculates the electron density by first computing the real-space density
    and then performing a Fourier transform to obtain the reciprocal space
    representation. This is useful for operations that are more efficient in
    reciprocal space, such as computing the Hartree potential.

    Args:
        wave_grid (Complex[Array, 'spin kpt band x y z']): Complex wave
          function values on a real-space grid. The array has dimensions for
          spin, k-points, bands, and spatial coordinates (x,y,z).
        occupation (Optional[Float[Array, 'spin kpt band']]): Optional
          occupation numbers for each state (spin, k-point, band). If provided,
          the density will be weighted by these values. Defaults to None.

    Returns:
        Union[Float[Array, 'spin kpt band x y z'], Float[Array, 'spin kpt band']]: The electron density grid in reciprocal space. If occupation is None, the density grid has the same shape as the input wave_grid. If occupation is provided, the density grid is reduced over the k-points and bands dimensions.

    """
  dens = wave_to_density(wave_grid, occupation)
  return jnp.fft.fftn(dens, axes=range(-3, 0))


def fft_factor(n: int) -> int:
  """Finds the smallest valid FFT size that is >= n.

    Determines the smallest number greater than or equal to n that can be
    factored as

    .. math::

        \\text{FFT size} = 2^a \\times 3^b \\times 5^c \\times 7^d \\times 11^e \\times 13^f

    where :math:`e` and :math:`f` are either 0 or 1.

    Args:
        n (int): The minimum size needed for the FFT grid.

    Returns:
        int: The smallest valid FFT size >= n that satisfies the prime factorization requirements.

    Raises:
        ValueError: If n > 2048, as the implementation is limited to sizes below this threshold.
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
  """Expands compact coefficients into a full grid using a boolean mask.

    Transforms coefficients from a compact representation (where only significant points are stored) to a full grid representation by placing the coefficients at positions specified by a boolean mask. This is useful for converting between storage-efficient and computation-friendly representations.

    Args:
        coeff_compact (Complex[Array, "spin kpt gpt band"]): Compact coefficient array with dimensions for spin, k-points, grid points, and bands.
        mask (Bool[Array, 'x y z']): Boolean mask indicating valid grid points in the expanded representation. The number of True values must match the last dimension of coeff_compact.

    Returns:
        Complex[Array, "spin kpt band x y z"]: The expanded coefficient array with dimensions matching the batch dimensions of coeff_compact (spin, kpt, band) followed by the spatial dimensions of the mask (x, y, z).

    .. note::

        The function first swaps the last two axes of the input coefficients to align with the expected output format, then creates a zero-filled array of the target shape and places the coefficients at the masked positions.

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
  """Compresses coefficients by extracting values at masked positions.

    Performs the inverse operation of expand_coefficient by extracting values
    from positions specified by a boolean mask and arranging them in a compact
    format. The output is transposed to have grid points before bands for
    efficient computation.

    .. note::

        The function extracts values at masked positions and then swaps the last two axes to arrange the output as (spin, kpt, gpt, band) rather than (spin, kpt, band, gpt).

    Args:
        coeff (Complex[Array, "spin kpt band x y z"]): Full coefficient array with dimensions for spin, k-points, bands, and spatial coordinates (x, y, z).
        mask (Bool[Array, "spin kpt band x y z"]): Boolean mask of the same shape as coeff indicating which positions should be included in the compact representation.

    Returns:
        Complex[Array, "spin kpt gpt band"]: Compact coefficient array with dimensions (spin, kpt, gpt, band), where gpt represents the number of True values in the mask.

    """
  coeff_compact = coeff[..., mask].get()
  return jnp.swapaxes(coeff_compact, -1, -2)


def check_spin_number(num_electrons: int, spin: int) -> None:
  """Validates that the spin number is compatible with electron count.

    Checks if the specified spin number (number of unpaired electrons) is
    physically possible given the total number of electrons. The spin number
    and total electron count must have the same parity (both odd or both even).

    Args:
        num_electrons (int): Total number of electrons in the system.
        spin (int): Number of unpaired electrons (spin number).

    Raises:
        ValueError: If the spin number is not valid for the given number
            of electrons (i.e., if they have different parity).

    """
  if num_electrons % 2 != spin % 2:
    raise ValueError("spin number is not valid for the system. ")
