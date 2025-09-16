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
"""Spherical coordinate system utilities and spherical harmonics
transformations.

This module provides functions for working with spherical coordinates and
spherical harmonics, particularly useful for quantum mechanical calculations
and pseudopotential transformations.
"""
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum
from jax.scipy.special import sph_harm, sph_harm_y
from jaxtyping import Array, Complex, Float

from .._src.utils import vmapstack


def compute_spherical_harmonics_grid(
  r_vector_grid: Float[Array, "*n 3"], l_max: int
) -> Complex[Array, "l *n m"]:
  """Compute spherical harmonics on a grid for all angular momenta up to l_max.

  Evaluates real spherical harmonics Y_l^m for all l from 0 to l_max and
  all corresponding m values (-l ≤ m ≤ l) at each point in the input grid.

  Args:
    r_vector_grid: Cartesian coordinates with shape (..., 3) where the last
    dimension contains [x, y, z] coordinates
    l_max: Maximum angular momentum quantum number (inclusive)

  Returns:
    Complex array with shape (l_max+1, ..., 2*l_max+1) where:
      - First dimension indexes angular momentum l = 0, 1, ..., l_max
      - Last dimension indexes magnetic quantum numbers in a padded format
      - For each l, only the central (2*l+1) elements contain valid Y_l^m values
      - The magnetic quantum numbers are ordered as m = -l, -l+1, ..., 0, ...,
      l-1, l

  Note:
    The output uses a padded format where all l-values share the same
    m-dimension size (2*l_max+1). For l < l_max, the unused entries are
    zero-padded.
  """
  batch_shape = r_vector_grid.shape[:-1]
  spherical_coords = cartesian_to_spherical(r_vector_grid)
  theta, phi = spherical_coords[..., 1], spherical_coords[..., 2]

  # Compute spherical harmonics for each l and stack them
  harmonics_by_l = []
  for li in range(l_max + 1):
    # Compute Y_l^m for all m values for this l
    ylm_l = batch_sph_harm_real(li, theta, phi)  # shape: (*batch, 2*l+1)

    # Pad to consistent size (2*l_max+1) for stacking
    padding_needed = 2 * l_max + 1 - (2 * li + 1)
    left_pad = padding_needed // 2
    right_pad = padding_needed - left_pad

    ylm_l_padded = jnp.pad(
      ylm_l, (*[(0, 0)] * len(batch_shape), (left_pad, right_pad)),
      mode='constant',
      constant_values=0.0
    )
    harmonics_by_l.append(ylm_l_padded)

  return jnp.stack(harmonics_by_l, axis=0)


def cartesian_to_spherical(x: Float[Array, "*n 3"],
                           eps=1e-10) -> Float[Array, "*n 3"]:
  """Convert Cartesian coordinates to spherical coordinates.

  Transforms 3D Cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ), where:

  - r is the radial distance from the origin
  - θ (theta) is the azimuthal angle in the x-y plane from the x-axis (0 ≤ θ < 2π)
  - φ (phi) is the polar angle from the z-axis (0 ≤ φ ≤ π)

  For the special case of the origin (0, 0, 0), returns (0, NaN, π/2).

  Args:
    x: Float[Array, "*n 3"]: Cartesian coordinates with shape (..., 3) where ... represents arbitrary batch dimensions

  Warning:
    The definition of theta and phi is different from the jax convention.

  Returns:
    Float[Array, "*n 3"]: Spherical coordinates (r, θ, φ) with same batch shape
  """
  r = jnp.linalg.norm(x, axis=-1)  # Radial distance
  r = jnp.where(r == 0., eps, r)

  # Polar angle (phi)
  phi = jnp.arccos(jnp.clip(x[..., 2] / r, -1.0, 1.0))

  # Azimuthal angle (theta) and shift to range [0, 2*pi)
  theta = jnp.arctan2(x[..., 1], x[..., 0])
  theta = jnp.mod(theta + 2 * jnp.pi, 2 * jnp.pi)

  return jnp.stack((r, theta, phi), axis=-1)


def batch_sph_harm_real(
  l: int, theta: Float[Array, "*batch"], phi: Float[Array, "*batch"]
) -> Float[Array, "*batch m"]:
  """
  Compute the real form of spherical harmonics for a batch of points.


  """
  _sph_harm1 = batch_sph_harm(l, theta, phi)  # [*batch m]
  m = jnp.arange(-l, l + 1)
  _sph_harm2 = einsum(_sph_harm1.conj(), (-1)**m, "... m, m -> ... m")

  output = jnp.where(
    m >= 0,
    _sph_harm1.real * jnp.sqrt(2) * (-1)**m,
    _sph_harm2.imag * jnp.sqrt(2) * (-1)**m,
  )  # [m, *batch]

  output = output.at[..., l].set(_sph_harm1[..., l].real)
  return output


def batch_sph_harm(
  l: int,
  theta: Float[Array, "*batch"],
  phi: Float[Array, "*batch"],
) -> Float[Array, "*batch m"]:
  """
    Compute the spherical harmonics for a batch of points.

    this function is used to compute the sum of spherical harmonics:

    .. math::

      Y_{l, m}(theta, phi)

  Args:
    l (int): The angular momentum quantum number.
    theta (Float[Array, "*batch"]): The azimuthal angle.
    phi (Float[Array, "*batch"]): The polar angle.

  Returns:
    Float[Array, "*batch m"]: The spherical harmonics, where the last dimension
    is the magnetic quantum number.

  """
  dim = theta.ndim
  m = np.arange(-int(l), int(l) + 1)
  n = np.array([l])

  @vmapstack(dim)
  def _sph_harm_fun(theta, phi):
    return sph_harm_y(n, m, phi, theta)

  return _sph_harm_fun(theta, phi)


# def legendre_to_sph_harm(
#   l: int = 0,
#   l_max: int = 4
# ) -> Callable[[Float[Array, "*batch 3"]], Float[Array, "*batch m"]]:
#   """Convert Legendre polynomials to spherical harmonics decomposition.

#   Implements the decomposition of Legendre polynomials into spherical harmonics
#   according to the formula:

#   .. math::
#     (2l+1) P_l(x^Ty) = 4\pi \sum_m Y_{l, m}(x) Y^*_{l, m}(y)

#   where:

#   - l is the angular momentum quantum number
#   - P_l is the Legendre polynomial of order l
#   - Y_{l,m} are the spherical harmonics
#   - m is the magnetic quantum number ranging from -l to +l

#   This transformation is particularly useful for efficient computation of P_l(G^T G') in quantum mechanical calculations, as it allows replacing expensive pairwise inner products of G vectors with faster spherical harmonics calculations.

#   Args:
#     l (int): Angular momentum quantum number (default: 0)

#   Returns:
#       Callable that takes Cartesian coordinates of shape (*batch, 3) and
#       returns spherical harmonics coefficients of shape (*batch, m), where m =
#       2l+1
#   """
#   m = np.arange(-l, l + 1)
#   n = np.array([l])

#   def fun(x):

#     @vmapstack(x.ndim - 1)
#     def _f(x_i):
#       x_spherical = cartesian_to_spherical(x_i)
#       theta = x_spherical[..., 1:2]  # azimuthal angle
#       phi = x_spherical[..., 2:3]  # polar angle

#       y_lm = jax.vmap(
#         sph_harm, in_axes=[0, None, None, None]
#       )(m, n, theta, phi).reshape([-1])  # [m]

#       y_lm = jnp.pad(y_lm, (0, (l_max - l) * 2), constant_values=0)
#       # pad the y_lm to length l_max with zeros
#       return y_lm * 4 * jnp.pi  # [m]

#     return _f(x)

#   return fun
