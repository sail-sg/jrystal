"""Functions about grids"""
import itertools
from typing import List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from ._typing import CellVector, ScalarGrid, VectorGrid
from .utils import fft_factor


def _half_frequency_ranges(
  grid_sizes: Union[Tuple, List, Int[Array, 'd']]
) -> Tuple:
  # instead of masks, we return ranges that are non-zero
  sizes = []
  starts = []
  for size in grid_sizes:
    # G_min = -(size // 2)
    G_max = (size - 1) // 2
    lower_bound = -G_max // 2
    upper_bound = G_max // 2
    pos_start = 0
    pos_size = upper_bound + 1
    neg_start = lower_bound + size
    neg_size = size - neg_start
    sizes.append((pos_size, neg_size))
    starts.append((pos_start, neg_start))
  return starts, sizes


def half_frequency_shape(
  grid_sizes: Union[Tuple, List, Int[Array, 'd']]
) -> Tuple:
  _, sizes = _half_frequency_ranges(grid_sizes)
  return tuple(map(sum, sizes))


def _half_frequency_pad_to(tensor, grid_sizes):
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
  """This is a shared function that is used by :code:`g_vectors`
  and :code:`r_vectors`.

  Args:
    basis: The cell vectors or reciprocal vectors.
    grid_sizes: number of grid points along each axis.
    normalize: :code:`False` for :code:`r_vectors` and :code:`True` for
      :code:`g_vectors`.

  Returns:
    jnp.ndarray: a tensor with shape `[*grid_sizes, d]`.
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


def g_vectors(cell_vectors, grid_sizes) -> VectorGrid[Float, 3]:
  r"""Given the lattice vector of the unit cell,
  and grid size on each axis, return the G vectors in the recirpocal space.

  We denote :code:`cell_vectors` as :math:`\boldsymbol{a}`,
  reciprocal vector as :math:`\boldsymbol{b}`.
  In 3D,

  .. math::

    &G_{ijk} = (i\boldsymbol{b}_1 + j\boldsymbol{b}_2 + k\boldsymbol{b}_3) \\
    &i \in [0, n_i-1], j \in [0, n_j-1], k \in [0, n_k-1]

  Args:
    cell_vectors: real space lattice vectors for the the unit cell.
      A `(d, d)` matrix if the spatial dimension is `d`. `d` is usually `3`
    grid_sizes: number of grid points along each axis.
  Returns:
    jnp.ndarray: a tensor with shape `(*grid_sizes, d)`.
  """
  b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
  return _vector_grid(b, grid_sizes)


def r_vectors(cell_vectors, grid_sizes) -> VectorGrid[Float, 3]:
  r"""Given the lattice vector of the unit cell,
  and grid size on each axis, return the R vectors
  in the real space.

  We denote :code:`cell_vectors` as :math:`\boldsymbol{a}`,
  reciprocal vector as :math:`\boldsymbol{b}`.
  In 3D,

  .. math::

    &R_{ijk} = \frac{i}{n_i}\boldsymbol{a}_1 + \frac{j}{n_j}\boldsymbol{a}_2
      + \frac{k}{n_k}\boldsymbol{a}_3 \\
    &i \in [0, n_i-1], j \in [0, n_j-1], k \in [0, n_k-1]

  Args:
    cell_vectors: real space lattice vectors for the the unit cell.
      A `(d, d)` matrix if the spatial dimension is `d`. `d` is usually `3`.
    grid_sizes: number of grid points along each axis.
  Returns:
    jnp.ndarray: a tensor with shape `(*grid_sizes, d)`.
  """
  return _vector_grid(cell_vectors, grid_sizes, normalize=True)


def proper_grid_size(
  grid_sizes: Union[Int, Int[Array, 'd'], Tuple, List]
) -> Array:
  """Get proper grid_sizes which is an numpy array, the grid size are good
  fftw factors.

  Args:
    grid_sizes (Int | Array | Tuple | List): grid size. can be either list,
    tuple, numpy.array or scalar. A scalar indicates that all the
    dimensions have the same grid size.

  Returns:
    Array: a numpy array indicates the grid size on each dimension.
  """
  if hasattr(grid_sizes, "__len__"):
    grid_sizes = np.array(grid_sizes)
  else:
    try:
      grid_sizes = np.ones(3, dtype=int) * int(grid_sizes)
    except:
      raise TypeError('mesh should be a scalar, tuple, list or np.array.')
  grid_sizes = np.array([fft_factor(i) for i in grid_sizes])
  return grid_sizes


def translation_vectors(
  cell_vectors: CellVector,
  cutoff: Union[Float[Array, '3'], float] = 1e4,
) -> Float[Array, 'num 3']:
  """Construct the translation grid lattice for ewald sum.

  Args:
    cell_vectors (ndarray): the reciprocal vectors
    cutoff (ndarray): the real space cutoff such that 1/cutoff ~ 0.
          The larger the more precise of ewald sum.

  Returns:
    The translation grid lattice, shape: (nt, 3)
  """

  dim = cell_vectors.shape[0]
  n = int(np.ceil(cutoff / np.linalg.norm(np.sum(cell_vectors, axis=0))**2))
  grid = _vector_grid(cell_vectors, [n for i in range(dim)])

  return np.reshape(grid, [-1, cell_vectors.shape[0]])


def k_vectors(cell_vectors: CellVector,
              grid_sizes: Int[Array, '...']) -> Float[Array, 'nkpts 3']:
  """Construct a uniform sampling of k-space of given size with
  Monkhorst-Pack scheme.

  Args:
    cell_vectors: cell vectors of shape `(3, 3)`.
    grid_sizes: `(x y z)`.

  Returns:
    Float[Array, 'nkpts 3']: k-vectors.
  """
  b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
  return monkhorst_pack(grid_sizes) @ b


def spherical_mask(
  cell_vectors: CellVector,
  grid_sizes: Union[List, jax.Array],
  cutoff_energy: float
) -> ScalarGrid[Bool, 3]:
  r"""Get a spherical fft frequency mask such that

  .. math::

    \frac{\|G\|^2}{2} \leq \text{cutoff}

  holds for all :math:`G`.

  Args:
    cell_vectors: cell vectors
    grid_sizes: grid sizes `(x y z)` of the 3D reciprocal frequency.
    cutoff_energy: cutoff energy of the frequencies.

  Returns:
    A 3D frequency mask.
  """
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  g_norm = np.linalg.norm(g_vector_grid, axis=-1, keepdims=False)
  mask = g_norm**2 <= cutoff_energy * 2
  return mask


def cubic_mask(grid_sizes: Union[List, jax.Array]) -> ScalarGrid[Bool, 3]:
  r"""Get a cubic fft frequency mask such that it enables only part of the
  frequencies. Since the density :math:`\rho` is the absolute square of
  the wave function :math:`\psi`, therefore :math:`\rho` contains more
  frequency components, as the components in :math:`\psi` are combined.

  Read more on
  1. Why / How to mask the frequency components.

  Args:
    grid_sizes: grid sizes `(x y z)` of the 3D reciprocal frequency.

  Returns:
    A 3D frequency mask.
  """
  masks = []
  for size in grid_sizes:
    # G_min = -(size // 2)
    G_max = (size - 1) // 2
    lower_bound = -G_max // 2
    upper_bound = G_max // 2
    m = np.ones((size,), dtype=bool)
    m[upper_bound + 1:lower_bound] = False
    masks.append(m)

  mask = jnp.einsum('i,j,k->ijk', *masks)
  return mask


def estimate_max_cutoff_energy(
  cell_vectors: CellVector,
  mask: ScalarGrid[Bool, 3],
) -> float:
  """Estimate the maximum of cutoff energy of a cubic mask.

  Args:
    cell_vectors (CellVector): cell vectors.
    mask (ScalarGrid[Bool, 3]): a cubic mask. can also be any frequency mask.

  Returns:
    the maximum cutoff energy corresponding to the input mask.
  """
  grid_sizes = mask.shape
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  kinetic = jnp.linalg.norm(g_vector_grid, axis=-1)**2 / 2
  return jnp.max(kinetic * mask).item()


def grid_vector_radius(grid_vector: Float[Array, "*n d"]):
  """get the radius of grid vectors.

  Args:
    grid_vector (VectorField): a input vector field.
      the first seveval dimensions can be any batch dimension.
  """

  def radius(r):
    return jnp.sqrt(jnp.sum(r**2))

  ndim = grid_vector.ndim - 1
  for _ in range(ndim):
    radius = jax.vmap(radius)

  return radius(grid_vector)


def g2r_vector_grid(
  g_vector_grid: Float[Array, "*nd d"],
  cell_vectors: CellVector,
) -> Float[Array, "*nd d"]:
  """Transform the G vector grid (reciprocal space) to R vector grid.

  Args:
    g_vector_grid (Float[Array, "*nd d"]): the G vector grid.
    cell_vectors (CellVector): the cell vectors.

  Returns:
    Float[Array, "*nd d"]: the R vector grid.
  """
  grid_sizes = g_vector_grid.shape[:-1]
  r_vector_grid = r_vectors(cell_vectors, grid_sizes)
  return r_vector_grid


def r2g_vector_grid(
  r_vector_grid: Float[Array, "*nd d"],
  cell_vectors: CellVector,
) -> Float[Array, "*nd d"]:
  """Transform the R vector grid to G vector grid.

  Args:
    r_vector_grid (Float[Array, "*nd d"]): the R vector grid.
    cell_vectors (CellVector): the cell vectors.

  Returns:
    Float[Array, "*nd d"]: the G vector grid.
  """

  grid_sizes = r_vector_grid.shape[:-1]
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  return g_vector_grid


def g2cell_vectors(g_vector_grid: Float[Array, "*nd d"]) -> CellVector:
  """Transform the G vector grid to cell vectors.

  Args:
    g_vector_grid (Float[Array, "*nd d"]): the G vector grid.

  Returns:
    CellVector: the cell vectors.
  """
  
  grid_sizes = g_vector_grid.shape[:-1]
  cardinality = g_vectors(jnp.eye(3), grid_sizes)
  a = g_vector_grid.reshape([-1, 3])
  b = cardinality.reshape([-1, 3])
  
  return jnp.linalg.inv(jnp.linalg.inv(a.T@a)@a.T@b)
