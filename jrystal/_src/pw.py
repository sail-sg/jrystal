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
'''Planewave module.'''
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Sharding
from jaxtyping import Array, Bool, Complex, Float

from .grid import g_vectors
from .spmd.fft import ifftn3d
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
  r'''Initialize the raw parameters.

  This function generates a random tensor of shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, where :code:`num_g` is the
  number of :code:`True` items in the :code:`freq_mask`.

  In planewave-based calculation, a wave function is represented as a
  linear combination of the Fourier series in 3D. Therefore, to create one
  wave function we need a 3D shaped tensor to represent the mixing
  coefficients on each frequency component (denoted as :code:`G`).
  :code:`freq_mask` provides a 3D mask to decide which frequency components
  are selected, the number of selected components is denoted as :code:`num_g`.

  The :code:`num_bands` & :code:`num_kpts` are a bit hard to explain. Intuitively,
  the wave functions consist of high frequency components that have a period
  smaller than the unit cell (denoted :math:`G`) and components that have a period
  larger than the unit cell (denoted :math:`k`).

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
    key (Array): Random key for initializing the parameters.
    num_bands (int): The number of bands.
    num_kpts (int): The number of k points.
    freq_mask (Bool[Array, 'x y z']): A 3D mask that denotes which frequency components are selected.
    spin_restricted (bool): If :code:`True`, :code:`num_spin=2` else :code:`num_spin=1`.

  Returns:
    Complex[Array, 'spin kpt gpt band']: A complex type raw parameter of shape :code:`(num_spin, num_kpts, num_g, num_bands)`.
  '''
  num_spin = 1 if spin_restricted else 2
  num_g = np.sum(freq_mask).item()
  shape = (num_spin, num_kpts, num_g, num_bands)
  return unitary_matrix_param_init(key, shape, complex=True, sharding=sharding)


def coeff(
  pw_param: Union[dict, Array, Tuple],
  freq_mask: Bool[Array, 'x y z'],
  sharding: Optional[Sharding] = None
) -> Complex[Array, 'spin kpt band x y z']:
  r'''Create the linear coefficients to combine the frequency components.

  This function takes a raw parameter of shape
  :code:`(num_spin, num_kpts, num_gpts, num_bands)`, orthogonalizes for the last
  two dimensions, so that the resulting tensor satisfies the unitary constraint
  :code:`einsum('kabc,labc->kl', ret[i, j], ret[i, j]) == eye(num_bands)`.

  The :code:`pw_param` should be created from :py:func:`param_init`, and the same
  :code:`freq_mask` used in :py:func:`param_init` should be used here. As mentioned
  in :py:func:`param_init`, we use linear combination over 3D Fourier
  components for creating wave functions. Some extra requirements are:

  1. The wave functions that have the same spin and same k component need
     to be orthogonal to each other.
  2. We only activate some of the frequency components with the :code:`freq_mask`.

  As the raw parameter returned from :py:func:`param_init` has the shape
  :code:`(num_spin, num_kpts, num_g, num_bands)`, where :code:`num_g` is the number of
  activated frequencies flattened from the activated entries in the :code:`freq_mask`,
  this function first orthogonalizes over the last two dimensions and
  reorganizes the orthogonalized parameter into a 3D grid the same shape as
  the frequency mask.

  Extension reads:
  1. Why and how to mask the frequency components.
  2. Bloch theorem.

  Args:
    pw_param (Union[Array, Tuple]): The raw parameter, maybe created from :py:func:`param_init`.
    freq_mask (Bool[Array, 'x y z']): A 3D mask to select the frequency components.

  Returns:
    Complex[Array, 'spin kpt band x y z']: Complex array of shape
    :code:`(num_spin, num_kpts, num_band, x y z)`. It satisfies the unitary
    constraint that for any :code:`i,j`, :code:`einsum('kabc,labc->kl', ret[i, j], ret[i, j])`
    is an identity matrix.
  '''
  coeff = unitary_matrix(pw_param, complex=True, sharding=sharding)
  return expand_coefficient(coeff, freq_mask)


def wave_grid(
  coeff: Complex[Array, 'spin kpt band x y z'],
  vol: Float,
) -> Complex[Array, 'spin kpt band x y z']:
  r'''Wave function evaluated at a grid of spatial locations.

  This function implements the :math:`u(r)` part of the Bloch wave function:

  .. math::

    u(r)=\frac{1}{\sqrt{\Omega_\text{cell}}} \sum_G c_{G} e^{iG^\top r}

  where :math:`G` is the 3D frequency components, :math:`\Omega_\text{cell}` is the
  volume of the crystal unit cell, which is to make sure the wave function is
  normalized within the cell.

  where :math:`c` is the linear coefficient. It combines over different
  :math:`G` components that is generated with :py:func:`jrystal.grid.g_vectors`.
  We can evaluate the wave function at any spatial location :math:`r` which takes
  :math:`O(|G|)` computation. However, if we evaluate this function on a specific
  spatial grid of size :math:`|G|`, we can be faster than :math:`O(|G|^2)` by using
  fourier transform. IFFT gives us an :math:`O(|G|\log(|G|))` implementation of the
  above equation. The :math:`G` and :math:`R` grid can be obtained from
  :py:func:`jrystal.grid.g_vectors` and :py:func:`jrystal.grid.r_vectors`
  correspondingly.

  .. code:: python

    G = jrystal.grid.g_vectors(*args)  # (x y z, 3)
    R = jrystal.grid.r_vectors(*args)  # (x y z, 3)
    coefficients = ...  # (x y z)
    vol = ...

    def wave_function(r):
      return (coefficients * jnp.exp(1j * G @ r)).sum() / jnp.sqrt(vol)

    # The following is O(|G|^2)
    wave_at_R_naive = jax.vmap(jax.vmap(jax.vmap(wave_function)))(R)
    # The following is O(|G|log|G|)
    wave_at_R_fft = wave_grid(coefficients, vol)

  As IFFT implements

  .. math::

    x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{i\frac{2\pi}{N}kn}

  It is a bit different from the definition of the wave function,
  if you check the code, we do two things to align them,

  1. we multiply back the :math:`N` to cancel the :math:`\frac{1}{N}`
  factor in the IFFT (in 3D the :code:`np.prod(grid_sizes)`).
  2. we divide by the :math:`\sqrt{\Omega_\text{cell}}`.

  The :code:`coeff` passed to this function has shape :code:`(..., x y z)`,
  it can have any leading dimension.
  It can be created using :py:func:`param_init` and :py:func:`coeff`.
  :py:func:`param_init` creates a raw parameter and :py:func:`coeff` converts
  that parameter into coefficients that are used to linearly weight the 3D
  fourier components.

  Args:
    coeff (Complex[Array, 'spin kpt band x y z']): Wave function coefficients, which has a shape of :code:`(spin, kpt, band, x, y, z)`.
    vol (float): Volume of the unit cell.

  Returns:
    Complex[Array, 'spin kpt band x y z']: Wave function evaluated at the spatial grid.
  '''
  grid_sizes = coeff.shape[-3:]
  wave_grid = ifftn3d(coeff)
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
  r'''Compute the density at the spatial grid.

  In a system with electrons, the density of the electron is the result of
  multiple wave functions overlapping in space. As mentioned in
  :py:func:`wave_grid`, the wave function is a linear combination of 3D Fourier
  components. To compute the density, usually we only need to take the absolute
  square of each wave function and sum them up.

  .. math::

    \rho(\vb{r}) = \sum_i |\psi_i(\vb{r})|^2

  This function evaluates the density :math:`\rho(r)` at the spatial grid
  generated from :py:func:`jrystal.grid.r_vectors`.

  In crystals, this is a little bit more complicated. The form of the wave
  function is:

  .. math::

    \psi(\vb{r}) = \frac{1}{\sqrt{\Omega_\text{cell}}} e^{i\vb{k}^\top \vb{r}} \sum_G c_{kG}
    e^{i\vb{G}^\top \vb{r}}

  The :math:`c_{kG}` can be computed from :py:func:`param_init`
  and :py:func:`coeff`. For calculation of density, we only need
  the :math:`c_{kG}` and the occupation :math:`o_k` over :math:`k`.

  .. math::

    \rho(\vb{r}) = \frac{1}{\Omega_\text{cell}} \sum_{G, G'} c_{kG}c_{kG'}^*
    e^{i\vb{(G - G')}^\top \vb{r}}

  Args:
    coeff: :math:`c_{kG}` part of the parameter. It can have a leading batch
      dimension which will be summed to get the overall density. Therefore the
      shape is :code:`(spin, kpt, band, x, y, z)`.
    vol: Volume of the unit cell, a real scalar.
    occupation: The occupation over different k frequencies. The shape is
      :code:`(spin, kpt, band)`, it should have the same leading dimension as
      :code:`coeff`. This is an optional argument. When
      :code:`occupation=None`, we compute the density contribution from each
      :math:`k` without summing them. If :code:`occupation` is provided, we sum
      up all the density from each :math:`k` weighted by the occupation.

  Returns:
    Union[Float[Array, "spin kpt band x y z"], Float[Array, "spin x y z"], Float
      [Array, "x y z"]]: A real-valued tensor that represents the density at
      the spatial grid computed from :py:func:`jrystal.grid.r_vectors`. The
      shape is :code:`(spin, x, y, z)` if :code:`occupation` is provided, else
      the shape is :code:`(spin, kpt, band, x, y, z)`.
  '''
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
  r'''Fourier transform of the density grid.

  In a system with electrons, the density of the electron is the result of
  multiple wave functions overlapping in the space. As we mention in
  :py:func:`wave_grid`, the wave function is a linear combination of 3D fourier
  components. To compute the density, usually we only need to take the absolute
  square of each wave function :math:`\psi_i`, multiply by the occupation :math:`f_i`
  and sum them up:

  .. math::

    \rho(\vb{r}) = \sum_i f_i |\psi_i(\vb{r})|^2

  :py:func:`density_grid` computes the density :math:`\rho(\vb{r})` at the spatial grid
  generated from :py:func:`jrystal.grid.r_vectors`.

  The discrete fourier transformation of :math:`\rho(\vb{r})` is

  .. math::

    \tilde{\rho}(\vb{G}) = \frac{1}{\Omega} \int_\Omega \rho(\vb{r}) e^{-\text{i} \vb{G}^\top \vb{r}} \dd{\vb{r}}

  We can also evaluate the :math:`\tilde{\rho}(G)` at any :math:`G`, but this function
  computes the :math:`\tilde{\rho}(G)` evaluated at the grid generated from
  :py:func:`jrystal.grid.g_vectors`. It is equivalent to computing the :math:`\rho(\vb{r})`
  at :py:func:`jrystal.grid.r_vectors` and then do the discrete fourier
  transform.

  Since this function is just composing FFT with :py:func:`density_grid`,
  we refer you to :py:func:`density_grid` for more details.

  Args:
    coeff: coefficients of the wave functions.
    vol: volume of the unit cell.
    occupation: occupation over the different :math:`k` components, see more in
      :py:func:`density_grid`.

  Returns:
    A complex valued tensor representing :math:`\tilde{\rho}(G)` evaluated at
    :math:`G` generated from :py:func:`jrystal.grid.g_vectors`.
  '''
  dens = density_grid(coeff, vol, occupation)
  return jnp.fft.fftn(dens, axes=range(-3, 0))


def wave_r(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
) -> Complex[Array, 'spin kpt band']:
  r'''Evaluate plane wave functions at location r.

  This function computes the :math:`\psi(r)` following the equation:

  .. math::

    \psi(r) = \frac{1}{\sqrt{\Omega_\text{cell}}} \sum_G c_{G} e^{i\vb{G}^\top \vb{r}}

  The :code:`coeff` provided is the :math:`c_{G}`, the :code:`cell_vectors` is
  used to generate the grid of frequency components :math:`G` by calling the function
  :py:func:`jrystal.grid.g_vectors`. The :code:`r` is the location where we
  evaluate the wave function.

  Args:
    r: Spatial location to evaluate the wave function, shape: (3,).
    coeff: Wave function coefficients, which has a shape of
      :code:`(spin, kpt, band, x, y, z)`.
      It can be created from :py:func:`param_init` followed by :py:func:`coeff`.
    cell_vectors: The cell vectors of the crystal unit cell.
    g_vector_grid: The G vectors computed from :py:func:`jrystal.grid.g_vectors`.
      If None, will be computed from cell_vectors.

  Returns:
    Complex[Array, 'spin kpt band']: Complex tensor that represents wave functions evaluated at location r.
  '''
  vol = volume(cell_vectors)
  x, y, z = coeff.shape[-3:]

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [x, y, z])

  if r.shape != (3,):
    raise ValueError('r must have shape (3,)')
  leading_dims = coeff.shape[:-3]
  coeff_ = coeff.reshape((-1, x, y, z))
  output = jnp.exp(1j * g_vector_grid @ r)
  output = jnp.einsum('lxyz,xyz->skb', coeff_, output)
  output = jnp.reshape(output, leading_dims)
  return output / jnp.sqrt(vol)


def density_r(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Float:
  r'''Compute the electron density at location r.

  This function computes the density at location :math:`r`. If occupation is
  not provided, it simply returns the absolute square of the :py:func:`wave_r`.
  If occupation is provided, the :code:`coeff` needs to have a shape of
  :code:`(spin, kpt, band, x, y, z)`, the :code:`occupation`
  needs to have the dimension of :code:`(spin, kpt, band)`.

  Args:
    r: Spatial location to evaluate the density, shape: (3,).
    coeff: Wave function coefficients, which has a shape of
      :code:`(spin, kpt, band, x, y, z)`.
      It can be created from :py:func:`param_init` followed by :py:func:`coeff`.
    cell_vectors: The cell vectors of the crystal unit cell.
    g_vector_grid: The G vectors computed from :py:func:`jrystal.grid.g_vectors`.
    occupation: Occupation over different k frequencies.
      Refer to :py:func:`density_grid` for more information.

  Returns:
    Float: A real scalar that represents the density at location r.
  '''
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
) -> Float[Array, '3']:
  r'''Compute the first order derivative of the density at location r.

  Refer to :py:func:`density_r` for more information.

  Args:
    r: Spatial location to evaluate the density, shape: (3,).
    coeff: Wave function coefficients, which has a shape of
      :code:`(spin, kpt, band, x, y, z)`.
      It can be created from :py:func:`param_init` followed by :py:func:`coeff`.
    cell_vectors: The cell vectors of the crystal unit cell.
    g_vector_grid: The G vectors computed from :py:func:`jrystal.grid.g_vectors`.
    occupation: Occupation over different k frequencies.
      Refer to :py:func:`density_grid` for more information.

  Returns:
    Float[Array, '3']: A real vector that represents the density derivative at
    location :code:`r`.
  '''

  def den(r):
    return density_r(r, coeff, cell_vectors, g_vector_grid, occupation)

  return jax.grad(den)(r)


def nabla_density_grid(
  r: Float[Array, '3'],
  coeff: Complex[Array, 'spin kpt band x y z'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Optional[Float[Array, 'x y z 3']] = None,
  occupation: Optional[Float[Array, 'spin kpt band']] = None,
) -> Union[Float[Array, "spin kpt band x y z"], Float[Array, "spin kpt band"]]:
  r'''Compute the first order derivative of the density at a specific grid of
  spatial locations.

  Refer to :py:func:`density_grid` for more information.

  Args:
    r: Spatial location to evaluate the density, shape: (3,).
    coeff: Wave function coefficients, which has a shape of
      :code:`(spin, kpt, band, x, y, z)`.
      It can be created from :py:func:`param_init` followed by :py:func:`coeff`.
    cell_vectors: The cell vectors of the crystal unit cell.
    g_vector_grid: The G vectors computed from :py:func:`jrystal.grid.g_vectors`.
    occupation: Occupation over different k frequencies.
      Refer to :py:func:`density_grid` for more information.

  Returns:
    Union[Float[Array, "spin kpt band x y z"], Float[Array, "spin kpt band"]]: A real-valued tensor that represents the density derivative at the spatial grid computed from :py:func:`jrystal.grid.r_vectors`. The shape is :code:`(x, y, z)` if :code:`occupation` is provided, else the shape is :code:`(spin, kpt, band, x, y, z)`.
  '''
  r = jnp.reshape(r, (-1))

  if r.shape[0] != 3:
    raise ValueError('r must have shape (3,)')

  vol = volume(cell_vectors)
  nx, ny, nz = coeff.shape[-3:]

  if g_vector_grid is None:
    g_vector_grid = g_vectors(cell_vectors, [nx, ny, nz])

  gr = jnp.dot(g_vector_grid, r)

  cosgr = jnp.cos(gr)
  singr = jnp.sin(gr)
  cr = jnp.real(coeff)
  ci = jnp.imag(coeff)

  rcos = jnp.einsum('skbxyz,xyz->skbxyz', cr, cosgr)
  rsin = jnp.einsum('skbxyz,xyz->skbxyz', cr, singr)
  icos = jnp.einsum('skbxyz,xyz->skbxyz', ci, cosgr)
  isin = jnp.einsum('skbxyz,xyz->skbxyz', ci, singr)

  o1 = jnp.sum(rcos - isin, axis=range(-3, 0))
  o1 = jnp.expand_dims(o1, -1)
  o2 = -jnp.einsum('skbxyz,xyzd->skbd', rsin + icos, g_vector_grid)
  o3 = jnp.sum(rsin + icos, axis=range(-3, 0))
  o3 = jnp.expand_dims(o3, -1)
  o4 = jnp.einsum('skbxyz,xyzd->skbd', rcos - isin, g_vector_grid)

  output = 2 * (o1 * o2 + o3 * o4)

  if occupation is not None:
    occupation = jnp.expand_dims(occupation, -1)
    output = jnp.sum(output * occupation, axis=range(0, 3)) / vol

  return output
