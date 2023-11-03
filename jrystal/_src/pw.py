import numpy as np
import jax.numpy as jnp

from typing import List
from jaxtyping import Int, Float, Array, Complex

from jrystal import errors
from jrystal import Crystal
from .grid import r_vectors
from .grid import g_vectors, _grid_sizes
from .utils import vmapstack

from ase.dft.kpoints import monkhorst_pack


def get_plane_wave_params(
  crystal: Crystal,
  Ecut: Float,
  g_grid_sizes: Int | List | Float[Array, 'nd'],  # noqa: F821
  k_grid_sizes,
  occ: str = 'simple',
  polarize: bool = True
):

  crystal = crystal
  Ecut = Ecut
  polarize = polarize
  g_grid_sizes = _grid_sizes(g_grid_sizes)
  k_grid_sizes = _grid_sizes(k_grid_sizes)
  occ = occ
  nspin = 2 if polarize else 1
  ni = crystal.nelec
  nk = np.prod(k_grid_sizes)
  spin = ni // 2
  g_mask = _get_mask_radius(crystal.A, g_grid_sizes, Ecut)

  ng = jnp.sum(g_mask)
  k_grid = monkhorst_pack(k_grid_sizes)
  g_vec = g_vectors(crystal.A, g_grid_sizes)
  r_vec = r_vectors(crystal.A, g_grid_sizes)
  cg_shape = [nspin, nk, ng, ni]

  return (cg_shape, g_mask, crystal.A, k_grid, spin), (r_vec, g_vec)


def _coeff_expand(
  cg: Complex[Array, "*batch ng"],
  mask: Int[Array, "*nd"],
) -> Complex[Array, "*batch_nd"]:
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
  n_sum = jnp.sum(mask)
  if not jnp.array_equal(cg.shape[-1], n_sum):
    raise errors.ApplyExpCoeffShapeError(cg.shape, n_sum)

  @vmapstack(times=cg.ndim - 1)
  def set_mask(c):
    o = jnp.zeros_like(mask, dtype=c.dtype)
    return o.at[mask == 1].set(c)

  return set_mask(cg)


def _coeff_compress(
  cg: Complex[Array, "*batch_nd"],
  mask: Int[Array, "*nd"],
) -> Complex[Array, "*batch ng"]:
  """The inverse operation of ` _coeff_expand` """

  @vmapstack(times=cg.ndim - mask.ndim)
  def _get_value(c):
    return c.at[mask == 1].get()

  return _get_value(cg)


def _get_mask_radius(a, grid_sizes, e_cut):
  g_vec = g_vectors(a, grid_sizes)
  g_norm = jnp.linalg.norm(g_vec, axis=-1, keepdims=False)
  return g_norm**2 <= e_cut * 2


def _get_mask_cubic(a, grid_sizes, e_cut):
  pass


def _get_grid_sizes_radius(a: Float[Array, 'd d'], e_cut: Float):
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


def _complex_norm_square(x: Complex[Array, '...']) -> Float[Array, '...']:
  """Compute the Square of the norm of a complex number
  """
  return jnp.abs(jnp.conj(x) * x)


if __name__ == "__main__":
  shape = (5, 6, 7)
  mask = np.random.randn(*shape) > 0
  ng = jnp.sum(mask)
  cg = jnp.ones([2, 3, ng])
  print(_coeff_expand(cg, mask).shape)
