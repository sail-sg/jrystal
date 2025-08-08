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
'''Grid operations for crystalline systems.

This module provides functions for working with real and reciprocal space grids in crystalline systems.
It includes utilities for:

- Generating G-vectors and R-vectors
- :math:`k`-point sampling for Brillouin zone integration
- Frequency space operations and masks
- Grid transformations between real and reciprocal space
'''
import itertools
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from ase.dft.kpoints import monkhorst_pack
from jax import lax
from jaxtyping import Array, Bool, Float, Int

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
  '''Calculate the shape of arrays in half-frequency representation.

  Args:
    grid_sizes (Union[Tuple, List, Int[Array, 'd']]): Grid dimensions along each axis.
    
  Returns:
    Tuple[int, ...]: Tuple of dimensions for the half-frequency array.
  '''
  _, sizes = _half_frequency_ranges(grid_sizes)
  return tuple(map(sum, sizes))


def _half_frequency_pad_to(
  tensor: Array, grid_sizes: Union[Tuple, List, Int[Array, 'd']]
):
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


def _vector_grid(
  basis: Float[Array, 'd'],
  grid_sizes: Union[Tuple, List, Int[Array, 'd']],
  normalize: bool = False
) -> Float[Array, 'x y z d']:
  '''This is a shared function that is used by :code:`g_vectors`
  and :code:`r_vectors`.

  Args:
    basis: The cell vectors or reciprocal vectors.
    grid_sizes: number of grid points along each axis.
    normalize: :code:`False` for :code:`r_vectors` and :code:`True` for
      :code:`g_vectors`.

  Returns:
    Float[Array, 'x y z d']: A tensor with shape [*grid_sizes, d] containing the vector grid.
  '''
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


def g_vectors(
  cell_vectors: Float[Array, '3 3'],
  grid_sizes: Union[Tuple, List, Int[Array, 'd']]
) -> Float[Array, 'x y z 3']:
  r'''Generate G-vectors (reciprocal space vectors) for a given crystal cell.
  
  Given the real space lattice vectors of a unit cell, computes the G-vectors
  in reciprocal space on a discrete grid. The G-vectors are fundamental for
  plane-wave calculations and Fourier transforms in periodic systems.
  
  The G-vectors are defined as:
  
  .. math::
    G_{ijk} = i\mathbf{b}_1 + j\mathbf{b}_2 + k\mathbf{b}_3
  
  where :math:`\mathbf{b}_i` are the reciprocal lattice vectors and
  :math:`i,j,k \in [0, n_i-1]` for grid sizes :math:`n_i`.
  
  Args:
    cell_vectors (Float[Array, '3 3']): Real space lattice vectors of the unit cell.
      A (3,3) matrix where each row is a lattice vector.
    grid_sizes (Union[Tuple, List, Int[Array, 'd']]): Number of grid points along each axis.
  
  Returns:
    Float[Array, 'x y z 3']: A tensor with shape (\*grid_sizes, 3) containing the G-vectors.
  '''
  b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
  return _vector_grid(b, grid_sizes)


def r_vectors(
  cell_vectors: Float[Array, '3 3'],
  grid_sizes: Union[Tuple, List, Int[Array, '3']]
) -> Float[Array, 'x y z 3']:
  r'''Generate R-vectors (real space position vectors) for a given crystal cell.
  
  Given the real space lattice vectors of a unit cell, computes the position
  vectors R on a discrete grid within the unit cell. These vectors define the
  sampling points where real-space quantities (like electron density) are evaluated.
  
  The R-vectors are defined as:
  
  .. math::
    R_{ijk} = \frac{i}{n_x}\mathbf{a}_1 + \frac{j}{n_y}\mathbf{a}_2 
              + \frac{k}{n_z}\mathbf{a}_3
  
  where :math:`\mathbf{a}_i` are the real space lattice vectors and
  :math:`i,j,k \in [0, n_i-1]` for grid sizes :math:`n_i`.
  
  Args:
    cell_vectors (Float[Array, '3 3']): Real space lattice vectors of the unit cell. A (3,3) matrix where each row is a lattice vector.
    grid_sizes (Union[Tuple, List, Int[Array, '3']]): Number of grid points along each axis.
  
  Returns:
    Float[Array, 'x y z 3']: A tensor with shape (*grid_sizes, 3) containing the R-vectors.
  '''
  return _vector_grid(cell_vectors, grid_sizes, normalize=True)


def proper_grid_size(
  grid_sizes: Union[Int, Int[Array, 'd'], Tuple, List]
) -> Array:
  '''Optimize grid sizes for efficient FFT operations.
  
  Converts input grid dimensions to values that are well-suited for FFT
  computations by factoring them into products of small primes (2, 3, 5, 7).
  This optimization can significantly improve FFT performance.
  
  Args:
    grid_sizes (Union[Int, Int[Array, 'd'], Tuple, List]): Input grid dimensions. Can be:
      - A single integer (same size for all dimensions)
      - A sequence of integers (size for each dimension)
      - A numpy array of integers
  
  Returns:
    Array: A numpy array containing the optimized grid sizes.
  
  Raises:
    TypeError: If grid_sizes is not a valid numeric type.
  '''
  if hasattr(grid_sizes, '__len__'):
    grid_sizes = np.array(grid_sizes)
  else:
    try:
      grid_sizes = np.ones(3, dtype=int) * int(grid_sizes)
    except:
      raise TypeError('mesh should be a scalar, tuple, list or np.array.')
  grid_sizes = np.array([fft_factor(i) for i in grid_sizes])
  return grid_sizes


def translation_vectors(
  cell_vectors: Float[Array, '3 3'],
  cutoff: Union[Float[Array, '3'], float] = 1e4,
) -> Float[Array, 'num 3']:
  '''Generate translation vectors for Ewald summation.
  
  Creates a grid of translation vectors used in Ewald summation methods for
  computing long-range interactions in periodic systems. The grid extends
  to a distance determined by the cutoff parameter.
  
  Args:
    cell_vectors (Float[Array, '3 3']): Real space lattice vectors of the unit cell.
      A (3,3) matrix where each row is a lattice vector.
    cutoff (Union[Float[Array, '3'], float]): Real space cutoff distance. Larger values give more precise
      Ewald summation results but increase computational cost.
      Default is 1e4.
  
  Returns:
    Float[Array, 'num 3']: An array of shape (:math:`n`, 3) containing translation vectors, where :math:`n` is determined by the cutoff distance.
  '''
  dim = cell_vectors.shape[0]
  n = int(np.ceil(cutoff / np.linalg.norm(np.sum(cell_vectors, axis=0))**2))
  grid = _vector_grid(cell_vectors, [n for i in range(dim)])
  return np.reshape(grid, [-1, cell_vectors.shape[0]])


def k_vectors(
  cell_vectors: Float[Array, '3 3'],
  grid_sizes: Union[Tuple, List, Int[Array, '3']]
) -> Float[Array, 'kpt 3']:
  '''Generate k-vectors for Brillouin zone sampling.
  
  Creates a uniform grid of :math:`k`-points in reciprocal space using the
  Monkhorst-Pack scheme. This sampling is essential for integrating
  periodic functions over the Brillouin zone in electronic structure
  calculations.
  
  .. Warning::

    This function is not differentiable as it uses :code:`monkhorst_pack` from :code:`ase`. In future, we will implement a custom differentiable version.
  
  Args:
    cell_vectors (Float[Array, '3 3']): Real space lattice vectors of the unit cell. A (3,3) matrix where each row is a lattice vector.
    grid_sizes (Union[Tuple, List, Int[Array, '3']]): Number of :math:`k`-points along each reciprocal lattice vector direction.
  
  Returns:
    Float[Array, 'kpt 3']: An array of shape (:math:`n`, 3) containing k-vectors, where :math:`n` is the total number of :math:`k`-points (product of :code:`grid_sizes`).
  '''
  b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
  return monkhorst_pack(grid_sizes) @ b


def spherical_mask(
  cell_vectors: Float[Array, '3 3'],
  grid_sizes: Union[List, jax.Array],
  cutoff_energy: float
) -> Bool[Array, 'x y z']:
  r'''Create a spherical mask for frequency cutoff in reciprocal space.
  
  Generates a boolean mask that selects G-vectors satisfying the energy
  cutoff condition:
  
  .. math::
    \frac{\|G\|^2}{2} \leq E_\text{cutoff}
  
  This mask is commonly used in plane-wave calculations to limit the basis
  set size while maintaining accuracy. G-vectors with kinetic energy above
  the cutoff are excluded.
  
  Args:
    cell_vectors (Float[Array, '3 3']): Real space lattice vectors of the unit cell. A (3,3) matrix where each row is a lattice vector.
    grid_sizes (Union[List, jax.Array]): Grid dimensions in reciprocal space.
    cutoff_energy: Energy cutoff for the G-vectors.
  
  Returns:
    Bool[Array, 'x y z']: A boolean array of shape (*grid_sizes) where True indicates G-vectors within the energy cutoff sphere.
  '''
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  g_norm = np.linalg.norm(g_vector_grid, axis=-1, keepdims=False)
  mask = g_norm**2 <= cutoff_energy * 2
  return mask


def cubic_mask(grid_sizes: Union[List, jax.Array]) -> Bool[Array, 'x y z']:
  r'''Create a cubic mask for frequency components in reciprocal space.
  
  Generates a mask that enables only certain frequency components in a
  cubic region of reciprocal space. This is particularly useful when
  dealing with electron density (:math:`\rho`), which contains more
  frequency components than the wavefunction (:math:`\psi`) since
  :math:`\rho = |\psi|^2`.
  
  The mask is constructed by considering the frequency mixing that occurs
  when squaring the wavefunction, ensuring all relevant frequency
  components are included.
  
  Args:
    grid_sizes (Union[List, jax.Array]): Grid dimensions in reciprocal space.
  
  Returns:
    Bool[Array, 'x y z']: A boolean array of shape (*grid_sizes) where True indicates allowed frequency components in the cubic mask.
  '''
  masks = []
  for size in grid_sizes:
    G_max = (size - 1) // 2
    lower_bound = -G_max // 2
    upper_bound = G_max // 2
    m = np.ones((size,), dtype=bool)
    m[upper_bound + 1:lower_bound] = False
    masks.append(m)

  mask = jnp.einsum('i,j,k->ijk', *masks)
  return mask


def estimate_max_cutoff_energy(
  cell_vectors: Float[Array, '3 3'],
  mask: Bool[Array, 'x y z'],
) -> float:
  '''Estimate the maximum cutoff energy corresponding to a frequency mask.
  
  Given a boolean mask in reciprocal space, calculates the maximum kinetic
  energy of the G-vectors that are included in the mask. This is useful
  for determining the effective energy cutoff of a given frequency mask,
  particularly when using non-spherical masks.
  
  Args:
    cell_vectors (Float[Array, '3 3']): Real space lattice vectors of the unit cell. A (3,3) matrix where each row is a lattice vector.
    mask (Bool[Array, 'x y z']): Boolean mask indicating which G-vectors are included.
  
  Returns:
    float: The maximum kinetic energy (in the same units as the reciprocal lattice vectors) of any G-vector included in the mask.
  '''
  grid_sizes = mask.shape
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  kinetic = jnp.linalg.norm(g_vector_grid, axis=-1)**2 / 2
  return jnp.max(kinetic * mask).item()


def grid_vector_radius(grid_vector: Float[Array, 'x y z 3']):
  '''Calculate the magnitude (radius) of vectors at each grid point.
  
  Computes the Euclidean norm of vectors at each point in a grid.
  The function is vectorized to efficiently handle arbitrary grid shapes.
  
  Args:
    grid_vector (Float[Array, 'x y z 3']): Array of vectors where the last 
      dimension contains the vector components, and earlier dimensions are 
      grid dimensions or batch dimensions.
  
  Returns:
    Float[Array, 'x y z']: Array of vector magnitudes with shape matching all but the last dimension of the input.
  '''

  def radius(r):
    return jnp.sqrt(jnp.sum(r**2))

  ndim = grid_vector.ndim - 1
  for _ in range(ndim):
    radius = jax.vmap(radius)

  return radius(grid_vector)


def g2r_vector_grid(
  g_vector_grid: Float[Array, 'x y z 3'],
  cell_vectors: Optional[Float[Array, '3 3']] = None,
) -> Float[Array, 'x y z 3']:
  '''Transform a G-vector grid to the corresponding R-vector grid.
  
  Converts a grid of vectors in reciprocal space (G-vectors) to the
  corresponding grid in real space (R-vectors). If cell vectors are
  not provided, they are computed from the G-vector grid.
  
  This transformation is useful when switching between reciprocal and
  real space representations of crystal quantities.
  
  Args:
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    cell_vectors (Optional[Float[Array, '3 3']], optional): Real space lattice vectors. If None, they will be computed from the G-vector grid. Defaults to None.

  Returns:
    Float[Array, 'x y z 3']: Grid of R-vectors in real space with the same  shape as the input.
  '''
  if cell_vectors is None:
    cell_vectors = g2cell_vectors(g_vector_grid)
  grid_sizes = g_vector_grid.shape[:-1]
  r_vector_grid = r_vectors(cell_vectors, grid_sizes)
  return r_vector_grid


def r2g_vector_grid(
  r_vector_grid: Float[Array, 'x y z 3'],
  cell_vectors: Optional[Float[Array, '3 3']] = None,
) -> Float[Array, 'x y z 3']:
  '''Transform an R-vector grid to the corresponding G-vector grid.

  Converts a grid of vectors in real space (R-vectors) to the
  corresponding grid in reciprocal space (G-vectors). This is the
  inverse operation of g2r_vector_grid.

  Args:
    r_vector_grid (Float[Array, 'x y z 3']): Grid of R-vectors in real space.
    cell_vectors (Float[Array, '3 3']): Real space lattice vectors.

  Returns:
    Float[Array, 'x y z 3']: Grid of G-vectors in reciprocal space with the same shape as the input.
  '''
  if cell_vectors is None:
    cell_vectors = r2cell_vectors(r_vector_grid)
  grid_sizes = r_vector_grid.shape[:-1]
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  return g_vector_grid


def g2cell_vectors(
  g_vector_grid: Float[Array, 'x y z 3']
) -> Float[Array, '3 3']:
  r'''Compute real space cell vectors from a G-vector grid.
  
  Determines the real space lattice vectors by solving a linear system
  that relates the G-vector grid to the standard reciprocal space basis.
  This is useful when only the G-vectors are known and the corresponding
  real space cell vectors are needed.
    
  Args:
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
  
  Returns:
    Float[Array, '3 3']: Real space lattice vectors as a (3,3) matrix where each row is a lattice vector.
  '''
  grid_sizes = g_vector_grid.shape[:-1]
  cardinality = g_vectors(jnp.eye(3), grid_sizes)
  a = cardinality.reshape([-1, 3])
  b = g_vector_grid.reshape([-1, 3])
  return jnp.linalg.inv(jnp.linalg.inv(a.T @ a) @ a.T @ b)


def r2cell_vectors(
  r_vector_grid: Float[Array, 'x y z 3']
) -> Float[Array, '3 3']:
  r'''Compute real space cell vectors from an R-vector grid.
  
  Determines the real space lattice vectors by solving a linear system
  that relates the R-vector grid to the standard real space basis.
  This is useful when only the R-vectors are known and the corresponding
  cell vectors are needed.

  Args:
    r_vector_grid (Float[Array, 'x y z 3']): Grid of R-vectors in real space.
  
  Returns:
    Float[Array, '3 3']: Real space lattice vectors as a (3,3) matrix where each row is a lattice vector.
  '''
  grid_sizes = r_vector_grid.shape[:-1]
  r = r_vector_grid.reshape((-1, 3))
  d = r_vectors(jnp.eye(3), grid_sizes).reshape((-1, 3))
  return jnp.linalg.inv(d.T @ d) @ d.T @ r
