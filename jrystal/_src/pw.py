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

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Sharding
from jaxtyping import Array, Bool, Complex, Float

from .fft import fftn, ifftn
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

  This function generates a random tensor of shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, where :code:`num_g` is the
  number of :code:`True` items in the :code:`freq_mask`.

  In planewave-based calculation, a wave function is represented as a
  linear combination of the Fourier series in 3D. Therefore, to create one
  wave function we need a 3D shaped tensor to represent the mixing
  coefficients on each frequency component (denoted as :code:`G`).
  :code:`freq_mask` provides a 3D mask to decide which frequency components
  are selected, the number of selected components is denoted as :code:`num_g`.

  The :code:`num_bands` & :code:`num_kpts` are a bit hard to explain.
  Intuitively, the wave functions consist of high frequency components that
  have a period smaller than the unit cell (denoted :math:`G`) and components
  that have a period larger than the unit cell (denoted :math:`k`).

  The form of wave function under solid state is:

  .. math::

    \psi(r) = e^{i\vb{k}^\top \vb{r}}\sum_G c_{kG} e^{i\vb{G}^\top \vb{r}}

  This function generates a raw parameter, which after processing by
  :py:func:`coeff` can be used as the :math:`c_{kG}` part of the above equation.

  Extension reads:
  1. Why and how to mask the frequency components.
  2. Bloch theorem.

  As far as this function is concerned, it simply returns a randomly
  initialized parameter of shape :code:`(num_spin, num_kpts, num_g, num_bands)`.
  The input arguments to this function are only used to determine the shape.

  Note that this function returns the raw parameter that cannot be used
  directly to weight the frequency components, as in quantum chemistry we
  require the wave functions to be orthogonal to each other.
  Check :py:func:`coeff` for converting the raw parameter into a unitary
  tensor.

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

  This function takes a raw parameter of shape
  :code:`(num_spin, num_kpts, num_gpts, num_bands)`, orthogonalizes for the last
  two dimensions, so that the resulting tensor satisfies the unitary constraint
  :code:`einsum('kabc,labc->kl', ret[i, j], ret[i, j]) == eye(num_bands)`.

  The :code:`pw_param` should be created from :py:func:`param_init`, and the
  same :code:`freq_mask` used in :py:func:`param_init` should be used here. As
  mentioned in :py:func:`param_init`, we use linear combination over 3D Fourier
  components for creating wave functions. Some extra requirements are:

  1. The wave functions that have the same spin and same k component need
     to be orthogonal to each other.
  2. We only activate some of the frequency components with the
  :code:`freq_mask`.

  As the raw parameter returned from :py:func:`param_init` has the shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, where :code:`num_g` is the
  number of activated frequencies flattened from the activated entries in the
  :code:`freq_mask`, this function first orthogonalizes over the last two
  dimensions and reorganizes the orthogonalized parameter into a 3D grid the
  same shape as the frequency mask.

  Extension reads:
  1. Why and how to mask the frequency components.
  2. Bloch theorem.

  Args:
    pw_param (Union[dict, Array, Tuple]): Raw parameters.
    freq_mask (Bool[Array, 'x y z']): Mask of active G-grid frequencies.
    sharding (Optional[Sharding]): Optional sharding for orthogonalization.

  Returns:
    Complex[Array, 'spin kpt band x y z']: Coefficients on the full reciprocal
    grid.
  """
  c = unitary_matrix(pw_param, complex=True, sharding=sharding)
  return expand_coefficient(c, freq_mask)


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
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[
  Float[Array, 'spin kpt band x y z'],
  Float[Array, 'spin x y z'],
]:
  r"""Compute electron density on the real-space grid.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Float[Array, 'spin kpt band x y z'], Float[Array, 'spin x y z']]:
    Per-state density if ``occupation`` is ``None``; otherwise occupied
    spin-resolved density.
  """
  wave_grid_arr = wave_grid(coeff, vol)
  dens = absolute_square(wave_grid_arr)

  if occupation is not None:

    if k_weights is not None:
      assert occupation.shape[1] == k_weights.shape[0], (
        f"occupation.shape[1] ({occupation.shape[1]}) must be equal to "
        f"k_weights.shape[0] ({k_weights.shape[0]})."
      )
      occupation = occupation * k_weights[None, :, None]

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
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Complex[Array, 'spin kpt band x y z'], Complex[Array, 'spin x y z']]:
  r"""Compute reciprocal-space density by FFT of :func:`density_grid`.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Union[float, Array]): Unit-cell volume.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Complex[Array, 'spin kpt band x y z'], Complex[Array, 'spin x y z']]:
    Reciprocal-space density.
  """
  dens = density_grid(coeff, vol, occupation, k_weights)
  return fftn(dens, axes=range(-3, 0))


def grad_density_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
  g_vector_grid: Float[Array, 'x y z 3'],
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Float[Array, 'spin x y z 3'], Float[Array, 'spin kpt band x y z 3']]:
  r"""Compute density gradient :math:`\nabla\rho` on the real-space grid.

  Uses the reciprocal-space relation
  :math:`(\nabla\rho)_d(G) = i G_d \hat\rho(G)`.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Float[Array, 'spin x y z 3'], Float[Array, 'spin kpt band x y z 3']]:
    Density gradient with a trailing direction axis of size 3.
  """
  dens_recip = density_grid_reciprocal(coeff, vol, occupation, k_weights)
  grads = []
  for d in range(3):
    grad_recip_d = 1j * g_vector_grid[..., d] * dens_recip
    grads.append(jnp.real(ifftn(grad_recip_d, axes=range(-3, 0))))
  return jnp.stack(grads, axis=-1)


def sigma_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
  g_vector_grid: Float[Array, 'x y z 3'],
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
  r"""Compute contracted density gradient :math:`\sigma = |\nabla\rho|^2`.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
    Contracted gradient squared on the real-space grid.
  """
  grad_dens = grad_density_grid(
    coeff, vol, g_vector_grid, occupation, k_weights
  )
  return jnp.sum(grad_dens**2, axis=-1)


def tau_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
  g_vector_grid: Float[Array, 'x y z 3'],
  kpts: Optional[Float[Array, 'kpt 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
  r"""Compute kinetic energy density on the real-space grid.

  .. math::

    \tau(\mathbf{r}) = \frac{1}{2}\sum_i f_i\,|\nabla\psi_i(\mathbf{r})|^2

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    kpts (Optional[Float[Array, 'kpt 3']]): k-point coordinates. Defaults to
      Gamma point.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
    Kinetic energy density on the real-space grid.
  """
  grid_sizes = coeff.shape[-3:]
  num_grid_points = np.prod(grid_sizes)

  nabla_psi_sq = jnp.zeros(coeff.shape)
  for d in range(3):
    gk_d = g_vector_grid[..., d]
    if kpts is not None:
      gk_d = gk_d + jnp.reshape(kpts[:, d], (1, -1, 1, 1, 1, 1))
    nabla_coeff_d = coeff * (1j * gk_d)
    nabla_psi_d = ifftn(nabla_coeff_d,
                        axes=range(-3, 0)) * num_grid_points / jnp.sqrt(vol)
    nabla_psi_sq = nabla_psi_sq + absolute_square(nabla_psi_d)

  tau = 0.5 * nabla_psi_sq

  if occupation is not None:
    if k_weights is not None:
      occupation = occupation * k_weights[None, :, None]
    tau = jnp.einsum('skb...,skb->s...', tau, occupation)

  return tau


def lapl_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
  g_vector_grid: Float[Array, 'x y z 3'],
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
  r"""Compute Laplacian of density :math:`\nabla^2\rho` on the real-space grid.

  Uses the reciprocal-space relation
  :math:`\widehat{\nabla^2\rho}(G) = -|G|^2 \hat\rho(G)`.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    vol (Float): Unit-cell volume.
    g_vector_grid (Float[Array, 'x y z 3']): Reciprocal-space G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Float[Array, 'spin x y z'], Float[Array, 'spin kpt band x y z']]:
    Laplacian of density on the real-space grid.
  """
  dens_recip = density_grid_reciprocal(coeff, vol, occupation, k_weights)
  g_sq = jnp.sum(g_vector_grid**2, axis=-1)
  lapl_recip = -g_sq * dens_recip
  return jnp.real(ifftn(lapl_recip, axes=range(-3, 0)))


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
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Float[Array, 'spin kpt band'], Float]:
  r"""Evaluate electron density at one real-space point.

  Args:
    r (Float[Array, '3']): Spatial point.
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Optional[Float[Array, 'x y z 3']]): Optional G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Float[Array, 'spin kpt band'], Float]:
    Per-state density if ``occupation`` is ``None``; otherwise total density at
    ``r``.
  """
  density = absolute_square(wave_r(r, coeff, cell_vectors, g_vector_grid))
  if occupation is not None:
    if k_weights is not None:
      assert occupation.shape[1] == k_weights.shape[0], (
        f"occupation.shape[1] ({occupation.shape[1]}) must be equal to "
        f"k_weights.shape[0] ({k_weights.shape[0]})."
      )
      occupation = occupation * k_weights[None, :, None]
    density = jnp.sum(density * occupation)
  return density


def nabla_density_r(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Float[Array, 'spin kpt band 3'], Float[Array, '3']]:
  r"""Compute :math:`\nabla \rho(r)` at a single point.

  Args:
    r (Float[Array, '3']): Spatial point.
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Optional[Float[Array, 'x y z 3']]): Optional G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

  Returns:
    Union[Float[Array, 'spin kpt band 3'], Float[Array, '3']]:
    Per-state density gradient if ``occupation`` is ``None``; otherwise
    occupation-weighted total density gradient at ``r``.
  """

  def den(r):
    return density_r(
      r, coeff, cell_vectors, g_vector_grid, occupation, k_weights
    )

  return jax.grad(den)(r)


def nabla_density_grid(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
  k_weights: Optional[Float[Array, ' kpts']] = None,
) -> Union[Float[Array, "spin kpt band 3"], Float[Array, "3"]]:
  r"""Compute density-gradient at a point from g vector grid.

  Args:
    r (Float[Array, '3']): Spatial point.
    coeff (Complex[Array, 'spin kpt band x y z']): Plane-wave coefficients.
    cell_vectors (Float[Array, '3 3']): Real-space cell vectors.
    g_vector_grid (Optional[Float[Array, 'x y z 3']]): Optional G-vector grid.
    occupation (Optional[Float[Array, 'spin kpt band']]): Occupation numbers.
    k_weights (Optional[Float[Array, ' kpts']]): Weights for each k-point.

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

    if k_weights is not None:
      assert occupation.shape[1] == k_weights.shape[0], (
        f"occupation.shape[1] ({occupation.shape[1]}) must be equal to "
        f"k_weights.shape[0] ({k_weights.shape[0]})."
      )
      occupation = occupation * k_weights[None, :, None]

    return jnp.einsum('skbq,skb->q', grad_density, occupation)

  return grad_density
