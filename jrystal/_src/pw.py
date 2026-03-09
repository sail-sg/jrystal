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
"""Plane-wave parameterization and evaluation utilities."""
from typing import Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from jax.sharding import Sharding
from jaxtyping import Array, Bool, Complex, Float

from .fft import ifftn
from .grid import g_vectors
from .unitary_module import unitary_matrix, unitary_matrix_param_init
from .utils import absolute_square, expand_coefficient, volume


def param_init(
  key: Array,
  num_bands: int,
  num_kpts: int,
  freq_mask: Bool[Array, 'x y z'],
  spin_restricted: bool = True,
  sharding: Optional[Sharding] = None
) -> dict:
  r"""Initialize raw plane-wave parameters.

  Args:
    key (Array): Random key.
    num_bands (int): Number of bands.
    num_kpts (int): Number of :math:`k` points.
    freq_mask (Bool[Array, 'x y z']): Mask of active G-grid frequencies.
    spin_restricted (bool): If ``True``, use one spin channel.
    sharding (Optional[Sharding]): Optional output sharding.

  Returns:
    dict: Parameter dictionary for :func:`coeff`.
  """
  num_spin = 1 if spin_restricted else 2
  num_g = np.sum(freq_mask).item()
  shape = (num_spin, num_kpts, num_g, num_bands)
  return unitary_matrix_param_init(key, shape, complex=True, sharding=sharding)


def coeff(
  pw_param: Union[dict, Array, Tuple],
  freq_mask: Bool[Array, 'x y z'],
  sharding: Optional[Sharding] = None
) -> Complex[Array, 'spin kpt band x y z']:
  r"""Build orthonormal plane-wave coefficients on the full G grid.

  Args:
    pw_param (Union[dict, Array, Tuple]): Raw parameters.
    freq_mask (Bool[Array, 'x y z']): Mask of active G-grid frequencies.
    sharding (Optional[Sharding]): Optional sharding for orthogonalization.

  Returns:
    Complex[Array, 'spin kpt band x y z']: Coefficients on the full reciprocal
    grid.
  """
  coeff = unitary_matrix(pw_param, complex=True, sharding=sharding)
  return expand_coefficient(coeff, freq_mask)


def wave_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
) -> Complex[Array, 'spin kpt band x y z']:
  r"""Evaluate periodic wavefunctions on the real-space grid.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.

  Returns:
    Complex[Array, 'spin kpt band x y z']: Wavefunctions on the spatial grid.
  """
  grid_sizes = coeff.shape[-3:]
  wave_grid = ifftn(coeff, axes=range(-3, 0))
  wave_grid *= np.prod(grid_sizes) / jnp.sqrt(vol)
  return wave_grid


def density_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[
  Float[Array, 'spin kpt band x y z'],
  Float[Array, 'spin x y z'],
]:
  r"""Compute electron density on the real-space grid.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Float[Array, 'spin kpt band x y z'], Float[Array, 'spin x y z']]:
    Per-state density if ``occupation`` is ``None``; otherwise occupied
    spin-resolved density.
  """
  wave_grid_arr = wave_grid(coeff, vol)
  dens = absolute_square(wave_grid_arr)

  if occupation is not None:
    try:
      dens = jnp.einsum('skb...,skb->s...', dens, occupation)
    except ValueError:
      raise ValueError(
        'Occupation should have a leading dimension that is the same as coeff. '
        f'Got occupation shape: {occupation.shape}, coeff shape: {coeff.shape}'
      )
  return dens


def density_grid_reciprocal(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Union[float, Array],
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Complex[Array, 'spin kpt band x y z'], Complex[Array, 'spin x y z']]:
  r"""Compute reciprocal-space density by FFT of :func:`density_grid`.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Union[float, Array]): Unit-cell volume.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Complex[Array, 'spin kpt band x y z'], Complex[Array, 'spin x y z']]:
    Reciprocal-space density.
  """
  dens = density_grid(coeff, vol, occupation)
  return jnp.fft.fftn(dens, axes=range(-3, 0))


def wave_r(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
) -> Complex[Array, 'spin kpt band']:
  r"""Evaluate wavefunctions at one real-space point.

  Args:
    r (Float[Array, '3']): Spatial point.
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Optional[Float[Array, 'x y z 3']]): Precomputed G-vector
      grid. If ``None``, it is generated from ``cell_vectors``.

  Returns:
    Complex[Array, 'spin kpt band']: Wavefunction values at ``r``.
  """
  vol = volume(cell_vectors)
  x, y, z = coeff.shape[-3:]

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [x, y, z])

  if r.shape != (3,):
    raise ValueError('r must have shape (3,)')
  leading_dims = coeff.shape[:-3]
  coeff_ = coeff.reshape((-1, x, y, z))
  output = jnp.exp(1j * g_vector_grid @ r)
  output = jnp.einsum('lxyz,xyz->l', coeff_, output)
  output = jnp.reshape(output, leading_dims)
  return output / jnp.sqrt(vol)


def density_r(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float[Array, 'spin kpt band'], Float]:
  r"""Evaluate electron density at one real-space point.

  Args:
    r (Float[Array, '3']): Spatial point.
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Optional[Float[Array, 'x y z 3']]): Optional G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Float[Array, 'spin kpt band'], Float]:
    Per-state density if ``occupation`` is ``None``; otherwise total density at
    ``r``.
  """
  density = absolute_square(wave_r(r, coeff, cell_vectors, g_vector_grid))
  if occupation is not None:
    density = jnp.sum(density * occupation)
  return density


def nabla_density_r(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float[Array, 'spin kpt band 3'], Float[Array, '3']]:
  r"""Compute :math:`\nabla \rho(r)` at a single point.

  Args:
    r (Float[Array, '3']): Spatial point.
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Optional[Float[Array, 'x y z 3']]): Optional G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Float[Array, 'spin kpt band 3'], Float[Array, '3']]:
    Per-state density gradient if ``occupation`` is ``None``; otherwise
    occupation-weighted total density gradient at ``r``.
  """
  return nabla_density_grid(r, coeff, cell_vectors, g_vector_grid, occupation)


def nabla_density_grid(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float[Array, "spin kpt band 3"], Float[Array, "3"]]:
  r"""Compute density-gradient at a point from g vector grid.

  Args:
    r (Float[Array, '3']): Spatial point.
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Optional[Float[Array, 'x y z 3']]): Optional G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.

  Returns:
    Union[Float[Array, "spin kpt band 3"], Float[Array, "3"]]:
    Per-state density gradient if ``occupation`` is ``None``; otherwise
    occupation-weighted total density gradient at ``r``.
  """
  vol = volume(cell_vectors)
  x, y, z = coeff.shape[-3:]

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [x, y, z])

  if r.shape != (3,):
    raise ValueError('r must have shape (3,)')

  leading_dims = coeff.shape[:-3]  # [spin, kpt, band]
  coeff_ = coeff.reshape((-1, x, y, z))
  phase = jnp.exp(1j * (g_vector_grid @ r))

  psi = jnp.einsum('lxyz,xyz->l', coeff_, phase)
  psi = psi / jnp.sqrt(vol)

  # grad psi = i / sqrt(V) * sum_G G * c_G * exp(i G dot r)
  grad_psi = jnp.einsum(
    'xyzd,lxyz,xyz->ld',
    1j * g_vector_grid,
    coeff_,
    phase,
  )
  grad_psi = grad_psi / jnp.sqrt(vol)

  grad_density = 2.0 * jnp.real(jnp.conj(psi)[:, None] * grad_psi)
  grad_density = grad_density.reshape(leading_dims + (3,))

  if occupation is not None:
    if occupation.shape != leading_dims:
      raise ValueError(
        'Occupation should have shape [spin, kpt, band]. '
        f'Got occupation shape: {occupation.shape}, expected {leading_dims}.'
      )
    return jnp.einsum('skbq,skb->q', grad_density, occupation)

  return grad_density
