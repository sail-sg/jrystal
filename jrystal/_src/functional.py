"""Pure functional API of wave function that will be used for constructing 
  flax.linen.modules. """
import jax
import jax.numpy as jnp
from jaxtyping import Int, Float, Array, Complex
from jrystal._src.grid import g_vectors
from jrystal._src.utils import vmapstack
from jrystal._src import errors

from typing import Union, List
from jrystal._src.jrystal_typing import ComplexGrid, MaskGrid, CellVector


def coeff_expand(
  coeff_dense: Complex[Array, "*batch ng"],
  mask: MaskGrid,
) -> ComplexGrid:
  """
  Expand coeffcients.
  The sum of masks should equals to the last dimension of cg.

  Example:

  >>> shape = (5, 6, 7)
  >>> mask = np.random.randn(*shape) > 0
  >>> ng = jnp.sum(mask)
  >>> cg = jnp.ones([2, 3, ng])

  >>> print(_coeff_expand(cg, mask).shape)
  >>> (2, 3, 5, 6, 7)

  Returns:
      expanded coeff: shape (*batch, *nd)
  """
  # n_sum = jnp.sum(mask)
  # if not jnp.array_equal(cg.shape[-1], n_sum):
  #   raise errors.ApplyExpCoeffShapeError(cg.shape, n_sum)

  @vmapstack(times=coeff_dense.ndim - 1)
  def set_mask(c):
    o = jnp.zeros_like(mask, dtype=c.dtype)
    return o.at[mask].set(c)

  return set_mask(coeff_dense)


def coeff_compress(
  coeff_grid: ComplexGrid,
  mask: MaskGrid,
) -> Complex[Array, "*batch ng"]:
  """The inverse operation of ``coeff_expand`` """

  @vmapstack(times=coeff_grid.ndim - mask.ndim)
  def _get_value(c):
    return c.at[mask].get()

  return _get_value(coeff_grid)


def get_mask_radius(
  cell_vectors: CellVector, grid_sizes: Union[List, jax.Array], e_cut: float
) -> MaskGrid:
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  g_norm = jnp.linalg.norm(g_vector_grid, axis=-1, keepdims=False)
  return g_norm**2 <= e_cut * 2


def get_mask_cubic(a, grid_sizes, e_cut):
  pass


def get_grid_sizes_radius(a: Float[Array, 'd d'], e_cut: Float):
  """Return the smallest grid sizes given the e_cut satisfy that

  .. math::
    n1 = n2 = n3 = n
    (n//2)^2 (b1 + b2 + b3)^2 / 2 <= e_cut

  Args:
      a (grid_lattice): _description_
      e_cut (_type_): _description_

  Returns:
      _type_: _description_
  """
  pass


def batched_fft(
  x: Union[Complex[Array, '...'], Float[Array, '...']], fft_dim: Int
) -> Complex[Array, '...']:
  """batched fast Fourier transform. FFT will perform over the last ``fft_dim``
    axes, and other axes are mapped. 

  Args:
      x (array): an array. 
      fft_dim (int): fft dimension.

  Returns:
      array: has the same shape as input.

  """
  if x.ndim < fft_dim:
    raise errors.ApplyFFTShapeError(fft_dim, x.shape)

  fft = vmapstack(x.ndim - fft_dim)(jnp.fft.fftn)
  return fft(x)


def batched_ifft(
  x: Union[Complex[Array, '...'], Float[Array, '...']], ifft_dim: Int
) -> Complex[Array, '...']:
  """batched invser fast Fourier transform. IFFT will perform over the last 
    ``ifft_dim`` axes, and other axes are mapped. 

  Args:
      x (array): an array. 
      ifft_dim (int): ifft dimension.

  Returns:
      array: has the same shape as input.

  """
  if x.ndim < ifft_dim:
    raise errors.ApplyFFTShapeError(ifft_dim, x.shape)

  ifft = vmapstack(x.ndim - ifft_dim)(jnp.fft.ifftn)
  return ifft(x)
