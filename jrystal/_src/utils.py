"""Utility functions."""
from typing import Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Complex, Float

from . import const
from ._typing import CellVector, OccupationArray, ScalarGrid


def safe_real(array: Array, tol: float = 1e-8) -> Array:
  """
  Convert a complex array to a real array if its imaginary part is effectively
  zero.

  Args:
    array (Array): Input array, can be real or complex.
    tol (float): Tolerance for the imaginary part to be considered zero.

  Returns:
    Array: A real array if the imaginary part is effectively zero.

  Raises:
    ValueError: If the array has a non-zero imaginary part.
  """
  if jnp.iscomplexobj(array):
    if jnp.allclose(array.imag, 0, atol=tol):  # Adjust tolerance as needed
      return array.real
    else:
      raise ValueError("Array has non-zero imaginary part")
  return array


def vmapstack(times: int, args: List[Dict] = None) -> Callable:
  """
  Apply JAX's vmap function multiple times over a function.

  Example:
    If f maps (3) -> (2), then vmapstack(f) maps (*batches, 3) -> (*batches, 2).

  Args:
    times (int): Number of times to apply vmap. Must match the dimension of the
    batches.
    args (List[Dict], optional): Arguments for each vmap application. Defaults
    to None.

  Returns:
    Callable: A function that maps from (*batches, _) to (*batches, _).

  Raises:
    ValueError: If the length of args does not match the number of times.
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
  """
  Compute the element-wise absolute square of a complex array.

  Args:
    array(Array): Input complex array.

  Returns:
    Array: An array containing the absolute square of each element.
  """
  return jnp.real(jnp.conj(array) * array)


def volume(cell_vectors: CellVector) -> Float:
  """
  Calculate the volume of a 3D grid given its shape and spacing.

  Args:
    cell_vectors (CellVector): The cell vectors of the crystal.

  Returns:
    float: The volume of the cell_vectors.
  """
  return jnp.abs(jnp.linalg.det(cell_vectors))


def wave_to_density(
  wave_grid: ScalarGrid[Complex, 3],
  occupation: Optional[OccupationArray] = None,
  axis: Optional[Union[int, Tuple, List]] = None
) -> ScalarGrid[Float, 3]:
  """Compute the density grid from the wave_grid and occupation mask,
  returning real space density values at grid points.

  Args:
    wave_grid (ScalarGrid[Complex, 3]): Wave function values at grid points.
    occupation (OccupationArray, optional): Occupation mask. If provided, the
    density will be reduced according to the occupation mask. If not provided,
    the output density will have the same shape as `wave_grid`. Defaults to
    None.
    axis (int | Tuple | List | None, optional): Axis or axes along which a
    contraction with the occupation mask is performed. If not provided, the
    function will reduce over all the spin, kpoint, and bands axes.

  Returns:
    ScalarGrid[Float, 3]: the density evaluated at grid in real space.
  """
  dens = absolute_square(wave_grid)
  axis = range(0, wave_grid.ndim - 3) if axis is None else axis

  if occupation is not None:
    try:
      occupation = jnp.expand_dims(occupation, range(-3, 0))
      dens = jnp.sum(dens * occupation, axis=axis)
    except:
      raise ValueError(
        f"wave_grid's shape ({wave_grid.shape}) and occupation's shape "
        f"({occupation.shape}) cannot align."
      )
  return dens


def wave_to_density_reciprocal(
  wave_grid: ScalarGrid[Complex, 3],
  occupation: OccupationArray = None,
  axis: Union[int, Tuple, List, None] = None
) -> ScalarGrid[Float, 3]:
  """Compute the density grid from the wave_grid and occupation mask,
  returning reciprocal space density values at grid points.

  Args:
    wave_grid (ScalarGrid[Complex, 3]): Wave function values at grid points.
    occupation (OccupationArray, optional): Occupation mask. If provided, the
    density will be reduced according to the occupation mask. If not provided,
    the output density will have the same shape as `wave_grid`. Defaults to
    None.
    axis (int | Tuple | List | None, optional): Axis or axes along which a
    contraction with the occupation mask is performed. If not provided, the
    function will reduce over all the spin, kpoint, and bands axes.

  Returns:
    ScalarGrid[Float, 3]: the density evaluated at grid in reciprocal space.
  """
  dens = wave_to_density(wave_grid, occupation, axis)
  return jnp.fft.fftn(dens, axes=range(-3, 0))


def fft_factor(n: int):
  """Get fftw factor n = 2^a * 3^b * 5^c *7 ^d * 11^e * 13^f   and  e/f = 0/1
  prime_factor_list = [2, 3, 5, 7, 11, 13] smaller than 2049.
  """

  fftw_factors = np.array(const.CUFFT_FACTORS)
  if n > 2048:
    raise ValueError(f"The grid number {n} is too large!")
  delta_n = (fftw_factors - n) >= 0
  output = fftw_factors[delta_n][0]
  return output


def expand_coefficient(
  coeff_compact: Complex[Array, "spin kpt band gpt"],
  mask: Bool[Array, 'x y z'],
) -> Complex[Array, "spin kpt band x y z"]:
  """
  Expand coefficients based on the provided mask.
  The sum of the mask should equal the last dimension of coeff_dense.

  Example:

  >>> shape = (5, 6, 7)
  >>> mask = np.random.randn(*shape) > 0
  >>> ng = jnp.sum(mask)
  >>> cg = jnp.ones([2, 3, ng])

  >>> print(_coeff_expand(cg, mask).shape)
  >>> (2, 3, 5, 6, 7)

  Args:
    coeff_dense (Complex[Array, "*batch ng"]): Dense coefficient array.
    mask (Bool[Array, '*ndim']): Boolean mask array.

  Returns:
    Complex[Array, "*batch ng"]: Expanded coefficients with shape
    (*batch, *ndim).
  """
  coeff_compact = jnp.swapaxes(coeff_compact, -1, -2)
  coeff_shape = coeff_compact.shape[:-1] + mask.shape
  return jnp.zeros(
    coeff_shape, dtype=coeff_compact.dtype
  ).at[..., mask].set(coeff_compact)


def squeeze_coefficient(
  coeff: Complex[Array, "spin kpts band x y z"],
  mask: Bool[Array, "spin kpts band x y z"],
) -> Complex[Array, "spin kpts num_g band"]:
  """
  Reshape the parameter to a compact shape with the mask, and the swap the
  last two axes such that the for each spin and kpt, the coefficients are
  a tall matrix where number of rows are greater than the number of columns.

  Args:
    coeff (Complex[Array, "spin kpts band x y z"]): The coefficient array.
    mask (Bool[Array, "spin kpts band x y z"]): The mask array.

  Returns:
    Complex[Array, "spin kpts num_g band"]: The squeezed coefficient array.
  """
  coeff_compact = coeff[..., mask].get()
  return jnp.swapaxes(coeff_compact, -1, -2)


def check_spin_number(num_electrons, spin):
  if num_electrons % 2 != spin % 2:
    raise ValueError("spin number is not valid for the system. ")
