import jax.numpy as jnp
from jaxtyping import Int, Float, Array, Complex
from jrystal._src.grid import g_vectors
from jrystal._src.utils import vmapstack
from jrystal import errors


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


if __name__ == "__main__":
  import numpy as np
  shape = (5, 6, 7)
  mask = np.random.randn(*shape) > 0
  ng = jnp.sum(mask)
  cg = jnp.ones([2, 3, ng])
  print(_coeff_expand(cg, mask).shape)
