"""Functions and modules about grids"""

import jax
import jax.numpy as jnp
from jaxtyping import Int, Array, Float
import numpy as np
from .utils import get_fftw_factor


def grid_1d(n: Int, normalize=False) -> Int[Array, 'n']:
  """Return a list of integers from 0 to n/2-1 and -n/2 to -1, which represent
  a canonical period. Used for computing fourier series.

  Args:
    n: grid size of the period
    normalize: divided by n if True.
  Returns:
    If n is even, return [0, 1, 2, ..., n/2-1, -n/2, -n/2+1, ..., -1]
                    else [0, 1, ..., n//2, -n//2, -n//2+1, ..., -1]
  """
  ub = n // 2 + 1
  lb = ub - n
  grid = jnp.roll(jnp.arange(lb, ub, 1), lb)
  return (grid / n if normalize else grid)


def _vector_grid(basis: jax.Array, grid_sizes, normalize=False):
  """_summary_

  Args:
    basis (_type_): _description_
    grid_sizes (_type_): _description_
    normalize (bool, optional): _description_. Defaults to False.
  """

  dim = len(grid_sizes)
  assert basis.shape[0] == basis.shape[1] == dim
  components = []
  for i in range(dim):
    shape = (*((grid_sizes[i] if _ == i else 1) for _ in range(dim)), dim)
    components.append(
      jnp.reshape(
        jnp.outer(grid_1d(grid_sizes[i], normalize), basis[i]), shape
      )
    )
  return sum(components)


def g_vectors(a, grid_sizes):
  r"""Given the lattice vector of the unit cell,
  and grid size on each axis, return the G vectors
  in the recirpocal space.

  In 3D,

  .. math::
    &G_{ijk} = (i\boldsymbol{b}_1 + j\boldsymbol{b}_2 + k\boldsymbol{b}_3) \\
    &i \in [0, n_i-1], j \in [0, n_j-1], k \in [0, n_k-1]

  Args:
    a: real space lattice vectors for the the unit cell.
      A `(d, d)` matrix if the spatial dimension is `d`.
    grid_sizes: number of grid points along each axis.
  Returns:
    jnp.ndarray: a tensor with shape `[*grid_sizes, d]`.
  """
  b = 2 * jnp.pi * jnp.linalg.inv(a).T
  return _vector_grid(b, grid_sizes)


def r_vectors(a, grid_sizes):
  r"""Given the lattice vector of the unit cell,
  and grid size on each axis, return the R vectors
  in the real space.

  In 3D,

  .. math::
    &R_{ijk} = \frac{i}{n_i}\boldsymbol{a}_1
    + \frac{j}{n_j}\boldsymbol{a}_2
    + \frac{k}{n_k}\boldsymbol{a}_3 \\
    &i \in [0, n_i-1], j \in [0, n_j-1], k \in [0, n_k-1]
  Args:
    a: real space lattice vectors for the the unit cell.
      A `(d, d)` matrix if the spatial dimension is `d`.
    grid_sizes: number of grid points along each axis.
  Returns:
    jnp.ndarray: a tensor with shape `[*grid_sizes, d]`.
  """
  return _vector_grid(a, grid_sizes, normalize=True)


def _grid_sizes(grid_sizes: Int | Int[Array, 'd']):
  if hasattr(grid_sizes, "__len__"):
    grid_sizes = np.array(grid_sizes)
  else:
    try:
      grid_sizes = np.ones(3, dtype=int) * int(grid_sizes)
    except:
      raise TypeError('mesh should be a scalar, tuple, list or np.array.')
  grid_sizes = np.array([get_fftw_factor(i) for i in grid_sizes])
  return grid_sizes


def _get_ewald_lattice(b: Float[Array, 'd d'],
                       ew_cut: Float[Array, 'd'] = 1e4) -> Float[Array, 'n d']:
  """get translation lattice for ewald sum

  Args:
    b (ndarray): the reciprocal vectors
    ew_cut (_type_): the real space cutoff. 1/ew_cut -> 0

  Returns:
      the translation lattice; shape: [nt, 3]
  """
  n = jnp.ceil(ew_cut / jnp.linalg.norm(jnp.sum(b, axis=0))**b.shape[0])
  grid = _vector_grid(b, [int(n) for i in range(b.shape[0])])

  return jnp.reshape(grid, [-1, b.shape[0]])
