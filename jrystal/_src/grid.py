"""Functions and modules about grids"""

import jax.numpy as jnp
from jax import lax
from jaxtyping import Int, Array, Float
import numpy as np
from jrystal._src.utils import get_fftw_factor
from jrystal._src.jrystal_typing import RealVecterGrid
from ase.dft.kpoints import monkhorst_pack
import itertools


# TODO(linmin): retire this function
# this is the same as jnp.fft.fftfreq(n)
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
  ub = (n + 1) // 2
  lb = ub - n
  grid = jnp.roll(jnp.arange(lb, ub, 1), lb)
  return (grid / n if normalize else grid)


def half_frequency_mask(grid_sizes):
  """Return a mask to select the frequency components that are less than half
  the maximum frequency, along each direction.
  """
  dim = len(grid_sizes)
  masks = []
  for size in grid_sizes:
    pos_max_freq = float((size - 1) // 2)
    neg_max_freq = float(-(size - 1) // 2)
    pos_size = int(np.floor(pos_max_freq / 2)) + 1
    neg_start = int(np.ceil(neg_max_freq / 2)) + size
    m = np.arange(size, dtype=int)
    m = np.logical_or(m < pos_size, m >= neg_start)
    masks.append(m)
  return jnp.einsum('i,j,k->ijk', *masks)


def _half_frequency_ranges(grid_sizes):
  # instead of masks, we return ranges that are non-zero
  sizes = []
  starts = []
  for size in grid_sizes:
    pos_max_freq = float((size - 1) // 2)
    neg_max_freq = float(-(size - 1) // 2)
    pos_start = 0
    pos_size = int(np.floor(pos_max_freq / 2)) + 1
    neg_start = int(np.ceil(neg_max_freq / 2)) + size
    neg_size = -int(np.ceil(neg_max_freq / 2))
    sizes.append((pos_size, neg_size))
    starts.append((pos_start, neg_start))
  return starts, sizes


def half_frequency_shape(grid_sizes):
  _, sizes = _half_frequency_ranges(grid_sizes)
  return tuple(map(sum, sizes))


def half_frequency_pad_to(tensor, grid_sizes):
  grid_sizes = tuple(grid_sizes)
  batch_dims = tensor.shape[:-len(grid_sizes)]
  starts, sizes = _half_frequency_ranges(grid_sizes)
  assert tensor.shape[-len(grid_sizes):] == tuple(map(sum, sizes))
  updates = [tensor]
  for i, sz in enumerate(sizes):
    split_updates = []
    for u in updates:
      split_updates.extend(jnp.split(u, [sz[0]], axis=i + len(batch_dims)))
    updates = split_updates
  start_indices = list(itertools.product(*starts))
  ret = jnp.zeros(batch_dims + grid_sizes, dtype=tensor.dtype)
  for s, u in zip(start_indices, updates):
    start_idx = (0,) * len(batch_dims) + s
    ret = lax.dynamic_update_slice(ret, u, start_idx)
  return ret


def _vector_grid(basis: Float[Array, "d"], grid_sizes, normalize=False):
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
    fftfreq = jnp.fft.fftfreq(
      grid_sizes[i], 1 if normalize else 1 / grid_sizes[i]
    )
    components.append(jnp.reshape(jnp.outer(fftfreq, basis[i]), shape))
  return sum(components)


def g_vectors(a, grid_sizes) -> RealVecterGrid:
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


def r_vectors(a, grid_sizes) -> RealVecterGrid:
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


def get_grid_sizes(grid_sizes: Int | Int[Array, 'd']):
  if hasattr(grid_sizes, "__len__"):
    grid_sizes = np.array(grid_sizes)
  else:
    try:
      grid_sizes = np.ones(3, dtype=int) * int(grid_sizes)
    except:
      raise TypeError('mesh should be a scalar, tuple, list or np.array.')
  grid_sizes = np.array([get_fftw_factor(i) for i in grid_sizes])
  return grid_sizes


def translation_vectors(
  cell_vectors: Float[Array, 'd d'],
  cutoff: Float[Array, 'd'] = 1e4,
) -> RealVecterGrid:
  """Construct the translation grid lattice for ewald sum.

  Args:
    cell_vectors (ndarray): the reciprocal vectors
    cutoff (ndarray): the real space cutoff such that 1/cutoff ~ 0.
          The larger the more precise of ewald sum.

  Returns:
      RealVecterGrid: the translation grid lattice; shape: [nt, 3]
  """

  dim = cell_vectors.shape[0]
  n = int(np.ceil(cutoff / np.linalg.norm(np.sum(cell_vectors, axis=0))**2))
  grid = _vector_grid(cell_vectors, [n for i in range(dim)])

  return np.reshape(grid, [-1, cell_vectors.shape[0]])


def k_vectors(cell_vectors: Float[Array, 'd d'],
              grid_sizes: Int[Array, '...']) -> Float[Array, 'num_k d']:
  b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
  return monkhorst_pack(grid_sizes) @ b
