# Copyright 2026 Garena Online Private Limited
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
'''Grid utilities for crystalline systems.

This module provides helpers for real-space and reciprocal-space grids in
periodic crystals, including:

- generation of G- and R-vector grids,
- :math:`k`-point sampling for Brillouin-zone integration,
- frequency-space masks, and
- transformations between real and reciprocal space.
'''
import itertools
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import spglib
from ase.dft.kpoints import monkhorst_pack
from jax import lax
from jaxtyping import Array, Bool, Float, Int

from .crystal import Crystal
from .utils import fft_factor


def _half_frequency_ranges(
  grid_sizes: Union[Tuple, List, Int[Array, ' d']]
) -> Tuple:
  # instead of masks, we return ranges that are non-zero
  sizes = []
  starts = []
  for size in grid_sizes:
    g_max = (size - 1) // 2
    lower_bound = -g_max // 2
    upper_bound = g_max // 2
    pos_start = 0
    pos_size = upper_bound + 1
    neg_start = lower_bound + size
    neg_size = size - neg_start
    sizes.append((pos_size, neg_size))
    starts.append((pos_start, neg_start))
  return starts, sizes


def half_frequency_shape(
  grid_sizes: Union[Tuple, List, Int[Array, ' d']]
) -> Tuple:
  '''Return the tensor shape for the half-frequency representation.

  Args:
    grid_sizes (Union[Tuple, List, Int[Array, 'd']]): Number of grid points
      along each axis.

  Returns:
    Tuple[int, ...]: Shape of the half-frequency tensor.
  '''
  _, sizes = _half_frequency_ranges(grid_sizes)
  return tuple(map(sum, sizes))


def _half_frequency_pad_to(
  tensor: Array, grid_sizes: Union[Tuple, List, Int[Array, ' d']]
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
  basis: Float[Array, ' d'],
  grid_sizes: Union[Tuple, List, Int[Array, ' d']],
  normalize: bool = False
) -> Float[Array, 'x y z d']:
  '''Construct a vector grid from a basis and grid dimensions.

  This internal helper is used by :func:`g_vectors` and :func:`r_vectors`.

  Args:
    basis: Cell vectors or reciprocal lattice vectors.
    grid_sizes: Number of grid points along each axis.
    normalize: If ``True``, use normalized frequencies (for
      :func:`r_vectors`). If ``False``, use FFT frequencies (for
      :func:`g_vectors`).

  Returns:
    Float[Array, 'x y z d']: Vector grid with shape ``(*grid_sizes, d)``.
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
  grid_sizes: Union[Tuple, List, Int[Array, ' d']]
) -> Float[Array, 'x y z 3']:
  r'''Generate reciprocal-space G vectors on a discrete grid.

  Given real-space lattice vectors, this function computes reciprocal-space
  vectors on the FFT grid. These vectors are fundamental to plane-wave methods
  and Fourier-space operators in periodic systems.

  The vectors are defined as:

  .. math::
    G_{ijk} = i\mathbf{b}_1 + j\mathbf{b}_2 + k\mathbf{b}_3

  where :math:`\mathbf{b}_i` are reciprocal lattice vectors.

  Args:
    cell_vectors (Float[Array, '3 3']): Real-space lattice vectors of the unit
      cell as a ``(3, 3)`` matrix. Each row is one lattice vector.
    grid_sizes (Union[Tuple, List, Int[Array, 'd']]): Number of grid points
      along each axis.

  Returns:
    Float[Array, 'x y z 3']: Array of shape ``(*grid_sizes, 3)`` containing
      G vectors.
  '''
  b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
  return _vector_grid(b, grid_sizes)


def r_vectors(
  cell_vectors: Float[Array, '3 3'],
  grid_sizes: Union[Tuple, List, Int[Array, '3']]
) -> Float[Array, 'x y z 3']:
  r'''Generate real-space R vectors on a discrete grid.

  Given real-space lattice vectors, this function computes position vectors on
  a discrete grid inside the unit cell. These vectors define where real-space
  fields (for example, electron density) are sampled.

  The vectors are defined as:

  .. math::
    R_{ijk} = \frac{i}{n_x}\mathbf{a}_1 + \frac{j}{n_y}\mathbf{a}_2
              + \frac{k}{n_z}\mathbf{a}_3

  where :math:`\mathbf{a}_i` are real-space lattice vectors.

  Args:
    cell_vectors (Float[Array, '3 3']): Real-space lattice vectors of the unit
      cell as a ``(3, 3)`` matrix. Each row is one lattice vector.
    grid_sizes (Union[Tuple, List, Int[Array, '3']]): Number of grid points
      along each axis.

  Returns:
    Float[Array, 'x y z 3']: Array of shape ``(*grid_sizes, 3)`` containing
      R vectors.
  '''
  return _vector_grid(cell_vectors, grid_sizes, normalize=True)


def proper_grid_size(
  grid_sizes: Union[Int, Int[Array, ' d'], Tuple, List]
) -> Array:
  '''Adjust grid sizes for efficient FFT execution.

  This function maps each dimension to an FFT-friendly value using
  :func:`fft_factor`.

  Args:
    grid_sizes (Union[Int, Int[Array, 'd'], Tuple, List]): Input grid
      dimensions. This can be a scalar or a sequence.

  Returns:
    Array: NumPy array containing FFT-friendly grid sizes.

  Raises:
    TypeError: If ``grid_sizes`` is not a valid numeric value or sequence.
  '''
  if hasattr(grid_sizes, '__len__'):
    grid_sizes = np.array(grid_sizes)
  else:
    try:
      grid_sizes = np.ones(3, dtype=int) * int(grid_sizes)
    except (ValueError, TypeError):
      raise TypeError('mesh should be a scalar, tuple, list or np.array.')
  try:
    grid_sizes = np.array(grid_sizes, dtype=int)
  except (ValueError, TypeError):
    raise TypeError('mesh should contain integer-like values.')

  if np.any(grid_sizes <= 0):
    raise ValueError(f"mesh dimensions must be positive, got {grid_sizes}.")

  grid_sizes = np.array([fft_factor(int(i)) for i in grid_sizes], dtype=int)
  return grid_sizes


def translation_vectors(
  cell_vectors: Float[Array, '3 3'],
  cutoff: float = 1e4,
) -> Float[Array, 'num 3']:
  '''Generate lattice translation vectors for Ewald summation.

  This function builds a grid of periodic image translations used for
  long-range interaction sums. The grid extent is controlled by ``cutoff``.

  Args:
    cell_vectors (Float[Array, '3 3']): Real-space lattice vectors of the unit
      cell as a ``(3, 3)`` matrix. Each row is one lattice vector.
    cutoff (float): Scalar real-space cutoff controlling how many periodic
      images are included. Larger values improve accuracy but increase cost.

  Returns:
    Float[Array, 'num 3']: Array of shape ``(n, 3)`` containing translation
      vectors, where ``n`` depends on ``cutoff``.
  '''
  if not np.isscalar(cutoff):
    raise TypeError(f"cutoff must be a scalar float, got type {type(cutoff)}.")
  cutoff = float(cutoff)
  if cutoff <= 0:
    raise ValueError(f"cutoff must be positive, got {cutoff}.")

  dim = cell_vectors.shape[0]
  n = int(np.ceil(cutoff / np.linalg.norm(np.sum(cell_vectors, axis=0))**2))
  grid = _vector_grid(cell_vectors, [n for i in range(dim)])
  return np.reshape(grid, [-1, cell_vectors.shape[0]])


def k_vectors(
  cell_vectors: Float[Array, '3 3'],
  grid_sizes: Union[Tuple, List, Int[Array, '3']],
  *,
  symmetry_reduction: Bool = True,
  scaled_positions: Optional[Float[Array, 'atom 3']] = None,
  charges: Optional[Int[Array, ' atom']] = None,
  k_shift: Optional[Tuple[bool, bool, bool]] = None,
  return_frac_coords: bool = False,
) -> Tuple[Float[Array, 'kpt 3'], Float[Array, ' kpt']]:
  '''Generate :math:`k` vectors for Brillouin-zone sampling.

  This function uses the Monkhorst-Pack scheme to construct a uniform
  reciprocal-space :math:`k`-point grid for Brillouin-zone integration.

  .. warning::

    This function is not differentiable because it calls
    :func:`ase.dft.kpoints.monkhorst_pack`.

  Example:
  >>> from jrystal import Crystal
  >>> from jrystal.grid import k_vectors

  >>> charges = [6, 6]
  >>> positions = [[0, 0, 0], [1.5, 1.5, 1.5]]
  >>> cell_vectors = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
  >>> crystal = Crystal(charges, positions, cell_vectors)

  >>> k_mesh, k_weights = k_vectors(
  cell_vectors, grid_sizes, scaled_positions=crystal.scaled_positions,
  charges=crystal.charges)
  >>> print(k_mesh.shape)

  Args:
    cell_vectors (Float[Array, '3 3']): Real-space lattice vectors of the unit
      cell as a ``(3, 3)`` matrix. Each row is one lattice vector.
    grid_sizes (Union[Tuple, List, Int[Array, '3']]): Number of :math:`k`
      points along each reciprocal-lattice direction.

  Returns:
    Float[Array, 'kpt 3']: Array of shape ``(n, 3)`` containing :math:`k`
      vectors, where ``n`` is ``prod(grid_sizes)``.
  '''
  # TODO: implement monkhorst_pack with jax
  if symmetry_reduction:
    if scaled_positions is None or charges is None:
      raise ValueError(
        "scaled_positions and charges must be provided if symmetry_reduction "
        "is True."
      )
    k_mesh, k_weights = _get_irreducible_k_mesh(
      crystal=None,
      k_grid_sizes=grid_sizes,
      cell_vectors=cell_vectors,
      scaled_positions=scaled_positions,
      charges=charges,
      k_shift=k_shift,
      return_frac_coords=return_frac_coords,
    )
  else:
    b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
    k_mesh = monkhorst_pack(grid_sizes) @ b
    k_weights = jnp.ones(k_mesh.shape[0]) / k_mesh.shape[0]

  return k_mesh, k_weights


def spherical_mask(
  cell_vectors: Float[Array, '3 3'],
  grid_sizes: Union[List, jax.Array],
  cutoff_energy: float
) -> Bool[Array, 'x y z']:
  r'''Create a reciprocal-space spherical mask from an energy cutoff.

  The mask keeps G vectors that satisfy:

  .. math::
    \frac{\|G\|^2}{2} \leq E_\text{cutoff}

  This is commonly used in plane-wave calculations to truncate the basis.

  Args:
    cell_vectors (Float[Array, '3 3']): Real-space lattice vectors of the unit
      cell as a ``(3, 3)`` matrix. Each row is one lattice vector.
    grid_sizes (Union[List, jax.Array]): Grid dimensions in reciprocal space.
    cutoff_energy: Kinetic-energy cutoff for G vectors.

  Returns:
    Bool[Array, 'x y z']: Boolean array of shape ``(*grid_sizes)`` where
      ``True`` marks vectors within the cutoff sphere.
  '''
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  g_norm = jnp.linalg.norm(g_vector_grid, axis=-1, keepdims=False)
  mask = g_norm**2 <= cutoff_energy * 2
  return mask


def cubic_mask(grid_sizes: Union[List, jax.Array]) -> Bool[Array, 'x y z']:
  r'''Create a cubic reciprocal-space frequency mask.

  This mask keeps components in a cubic region of reciprocal space. It is
  useful for quantities such as electron density :math:`\rho = |\psi|^2`,
  which can require a broader frequency range than :math:`\psi`.

  Args:
    grid_sizes (Union[List, jax.Array]): Grid dimensions in reciprocal space.

  Returns:
    Bool[Array, 'x y z']: Boolean array of shape ``(*grid_sizes)`` where
      ``True`` marks allowed frequency components.
  '''
  masks = []
  for size in grid_sizes:
    g_max = (size - 1) // 2
    lower_bound = -g_max // 2
    upper_bound = g_max // 2
    m = np.ones((size,), dtype=bool)
    m[upper_bound + 1:lower_bound] = False
    masks.append(m)

  mask = jnp.einsum('i,j,k->ijk', *masks)
  return mask


def estimate_max_cutoff_energy(
  cell_vectors: Float[Array, '3 3'],
  mask: Bool[Array, 'x y z'],
) -> float:
  '''Estimate the effective cutoff energy of a frequency mask.

  Given a boolean mask in reciprocal space, this function computes the
  maximum kinetic energy among included G vectors.

  Args:
    cell_vectors (Float[Array, '3 3']): Real-space lattice vectors of the unit
      cell as a ``(3, 3)`` matrix. Each row is one lattice vector.
    mask (Bool[Array, 'x y z']): Boolean mask indicating which G vectors are
      included.

  Returns:
    float: Maximum kinetic energy of all G vectors selected by ``mask``.
  '''
  grid_sizes = mask.shape
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  kinetic = jnp.linalg.norm(g_vector_grid, axis=-1)**2 / 2
  return jnp.max(kinetic * mask).item()


def grid_vector_radius(grid_vector: Float[Array, 'x y z 3']):
  '''Compute vector magnitudes at each grid point.

  This function applies the Euclidean norm along the last axis of
  ``grid_vector``.

  Args:
    grid_vector (Float[Array, 'x y z 3']): Array of vectors. The final axis
      stores vector components; leading axes are grid or batch dimensions.

  Returns:
    Float[Array, 'x y z']: Array of magnitudes with shape
      ``grid_vector.shape[:-1]``.
  '''
  return jnp.linalg.norm(grid_vector, axis=-1)


def g2r_vector_grid(
  g_vector_grid: Float[Array, 'x y z 3'],
  cell_vectors: Optional[Float[Array, '3 3']] = None,
) -> Float[Array, 'x y z 3']:
  '''Convert a G-vector grid to the corresponding R-vector grid.

  If ``cell_vectors`` is not provided, it is inferred from
  ``g_vector_grid``.

  Args:
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G vectors in reciprocal
      space.
    cell_vectors (Optional[Float[Array, '3 3']], optional): Real-space lattice
      vectors. If ``None``, they are inferred from ``g_vector_grid``.

  Returns:
    Float[Array, 'x y z 3']: R-vector grid in real space with the same shape
      as the input.
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
  '''Convert an R-vector grid to the corresponding G-vector grid.

  If ``cell_vectors`` is not provided, it is inferred from
  ``r_vector_grid``.

  Args:
    r_vector_grid (Float[Array, 'x y z 3']): Grid of R vectors in real space.
    cell_vectors (Optional[Float[Array, '3 3']], optional): Real-space lattice
      vectors. If ``None``, they are inferred from ``r_vector_grid``.

  Returns:
    Float[Array, 'x y z 3']: G-vector grid in reciprocal space with the same
      shape as the input.
  '''
  if cell_vectors is None:
    cell_vectors = r2cell_vectors(r_vector_grid)
  grid_sizes = r_vector_grid.shape[:-1]
  g_vector_grid = g_vectors(cell_vectors, grid_sizes)
  return g_vector_grid


def g2cell_vectors(
  g_vector_grid: Float[Array, 'x y z 3']
) -> Float[Array, '3 3']:
  r'''Infer real-space cell vectors from a G-vector grid.

  This function solves a linear system relating the given G-vector grid to the
  canonical reciprocal basis.

  Args:
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G vectors in reciprocal
      space.

  Returns:
    Float[Array, '3 3']: Real-space lattice vectors as a ``(3, 3)`` matrix,
      one lattice vector per row.
  '''
  grid_sizes = g_vector_grid.shape[:-1]
  cardinality = g_vectors(jnp.eye(3), grid_sizes)
  a = cardinality.reshape([-1, 3])
  b = g_vector_grid.reshape([-1, 3])
  return jnp.linalg.inv(jnp.linalg.inv(a.T @ a) @ a.T @ b)


def r2cell_vectors(
  r_vector_grid: Float[Array, 'x y z 3']
) -> Float[Array, '3 3']:
  r'''Infer real-space cell vectors from an R-vector grid.

  This function solves a linear system relating the given R-vector grid to the
  canonical real-space basis.

  Args:
    r_vector_grid (Float[Array, 'x y z 3']): Grid of R vectors in real space.

  Returns:
    Float[Array, '3 3']: Real-space lattice vectors as a ``(3, 3)`` matrix,
      one lattice vector per row.
  '''
  grid_sizes = r_vector_grid.shape[:-1]
  r = r_vector_grid.reshape((-1, 3))
  d = r_vectors(jnp.eye(3), grid_sizes).reshape((-1, 3))
  return jnp.linalg.inv(d.T @ d) @ d.T @ r


def _get_irreducible_k_mesh(
  crystal: Optional[Crystal],
  k_grid_sizes: Union[Tuple, List, Int[Array, '3']],
  *,
  cell_vectors: Optional[Float[Array, '3 3']] = None,
  scaled_positions: Optional[Float[Array, 'atom 3']] = None,
  charges: Optional[Int[Array, ' atom']] = None,
  return_frac_coords: bool = False,
  k_shift: Optional[Tuple[bool, bool, bool]] = None,
) -> Tuple[Float[Array, 'kpts 3'], Float[Array, ' kpts']]:
  '''Get the irreducible k mesh for a crystal.

  Args:
    Crystal: Crystal object. This is optional, if not provided, the
    cell_vectors, positions, and charges must be provided instead.
    k_grid_sizes: Number of grid points along each reciprocal-lattice direction.
      Must be a tuple or list or an array of length 3.
    cell_vectors: Real-space cell vectors. Only required if crystal is not
      provided.
    scaled_positions: Fractional atomic positions. Only required if crystal is
      not provided.
    charges: Atomic charges. Only required if crystal is not provided.
    return_frac_coords: If True, return fractional coordinates. If False,
      return absolute coordinates.

  Returns:
    Tuple[Float[Array, 'kpts 3'], Float[Array, 'kpts']]: The irreducible
      k-mesh and the associated weights of the k-points. The weights are
      normalized to 1 (that is, ``weights.sum() == 1``). If
      ``return_frac_coords`` is True, the k-mesh is returned in fractional
      coordinates. Otherwise, the k-mesh is returned in absolute coordinates.
  '''
  k_grid_sizes = np.asarray(k_grid_sizes, dtype=np.int32)
  if k_grid_sizes.shape != (3,):
    raise ValueError(
      "k_grid_sizes must have shape (3,), "
      f"got {k_grid_sizes.shape}."
    )
  if np.any(k_grid_sizes <= 0):
    raise ValueError(f"k_grid_sizes must be positive, got {k_grid_sizes}.")

  if crystal is not None:
    cell = (
      np.asarray(crystal.cell_vectors, dtype=np.float64),
      np.asarray(crystal.scaled_positions, dtype=np.float64),
      np.asarray(crystal.charges, dtype=np.int32),
    )
  else:
    if cell_vectors is None or scaled_positions is None or charges is None:
      raise ValueError(
        "If crystal is None, cell_vectors, scaled_positions, and charges "
        "must all be provided."
      )
    cell = (
      np.asarray(cell_vectors, dtype=np.float64),
      np.asarray(scaled_positions, dtype=np.float64),
      np.asarray(charges, dtype=np.int32),
    )

  if k_shift is None:
    k_shift = np.zeros(3, dtype=np.int32)
  else:
    k_shift = np.asarray(k_shift, dtype=np.int32)
    if k_shift.shape != (3,):
      raise ValueError(f"k_shift must have shape (3,), got {k_shift.shape}.")
    if not np.all((k_shift == 0) | (k_shift == 1)):
      raise ValueError(f"k_shift entries must be 0 or 1, got {k_shift}.")

  mapping, grid = spglib.get_ir_reciprocal_mesh(
    tuple(int(i) for i in k_grid_sizes), cell, is_shift=k_shift
  )

  ir_ids = np.unique(mapping)

  counts = np.bincount(mapping, minlength=mapping.max() + 1)
  weights = counts[ir_ids] / np.prod(k_grid_sizes)
  kpts_frac = (grid[ir_ids] + 0.5 * k_shift) / np.array(k_grid_sizes)
  kpts_frac = (kpts_frac + 0.5) % 1.0 - 0.5  # shift to [-0.5, 0.5]

  if return_frac_coords:
    return jnp.array(kpts_frac), jnp.array(weights)
  else:
    if crystal is not None:
      cell_vectors = crystal.cell_vectors

    b = 2 * jnp.pi * jnp.linalg.inv(cell_vectors).T
    return jnp.array(kpts_frac) @ b, jnp.array(weights)
