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
from jaxtyping import Array, Float

from .._src.utils import vmapstack


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
  l: int,
  theta: Float[Array, "*batch"],
  phi: Float[Array, "*batch"]
) -> Float[Array, "*batch m"]:
  """
  Compute the real form of spherical harmonics for a batch of points.
  """
  _sph_harm1 = batch_sph_harm(l, theta, phi)  # [*batch m]
  m = jnp.arange(-l, l + 1)
  _sph_harm2 = einsum(_sph_harm1.conj(), (-1) ** m, "... m, m -> ... m")

  output = jnp.where(
    m >= 0,
    _sph_harm1.real * jnp.sqrt(2) * (-1) ** m,
    _sph_harm2.imag * jnp.sqrt(2) * (-1) ** m,
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
    Float[Array, "*batch m"]: The spherical harmonics, where the last dimension is the magnetic quantum number.

  """
  dim = theta.ndim
  m = np.arange(-int(l), int(l) + 1)
  n = np.array([l])

  @vmapstack(dim)
  def _sph_harm_fun(theta, phi):
    return sph_harm_y(n, m, phi, theta)
    # note that the definitions of theta and phi are swapped in sph_harm_y.

  return _sph_harm_fun(theta, phi)


def legendre_to_sph_harm(
  l: int = 0,
  l_max: int = 4
) -> Callable[[Float[Array, "*batch 3"]], Float[Array, "*batch m"]]:
  """Convert Legendre polynomials to spherical harmonics decomposition.

  Implements the decomposition of Legendre polynomials into spherical harmonics
  according to the formula:

  .. math::
    (2l+1) P_l(x^Ty) = 4\pi \sum_m Y_{l, m}(x) Y^*_{l, m}(y)

  where:

  - l is the angular momentum quantum number
  - P_l is the Legendre polynomial of order l
  - Y_{l,m} are the spherical harmonics
  - m is the magnetic quantum number ranging from -l to +l

  This transformation is particularly useful for efficient computation of P_l(G^T G') in quantum mechanical calculations, as it allows replacing expensive pairwise inner products of G vectors with faster spherical harmonics calculations.

  Args:
    l (int): Angular momentum quantum number (default: 0)

  Returns:
      Callable that takes Cartesian coordinates of shape (*batch, 3) and
      returns spherical harmonics coefficients of shape (*batch, m), where m =
      2l+1
  """
  m = np.arange(-l, l + 1)
  n = np.array([l])

  def fun(x):

    @vmapstack(x.ndim - 1)
    def _f(x_i):
      x_spherical = cartesian_to_spherical(x_i)
      theta = x_spherical[..., 1:2]  # azimuthal angle
      phi = x_spherical[..., 2:3]  # polar angle

      y_lm = jax.vmap(
        sph_harm, in_axes=[0, None, None, None]
      )(m, n, theta, phi).reshape([-1])  # [m]

      y_lm = jnp.pad(y_lm, (0, (l_max - l) * 2), constant_values=0)
      # pad the y_lm to length l_max with zeros
      return y_lm * 2 * jnp.sqrt(jnp.pi)  # [m]

    return _f(x)

  return fun


def legendre_kernel_trick(l: int = 0) -> Callable:  # noqa
  """Decompose legendre polynomials via kernel trick:

      (2l+1) P_l (x^Ty) = \phi(x)^T \phi(y)

  Return \phi

  Args:
      l (int, optional): The degree of the legendre polynomial. Defaults to 0.

  Returns:
      callable: return a function that map from shape [n1 n2 n3 3] to
      [n1 n2 n3 new_dim] where new_dim is decided using kernel trick.
  """
  if l == 0:

    def phi(x):
      return jnp.sqrt(3) / 3 + jnp.zeros_like(x)

    return phi
  elif l == 1:

    def phi(x):
      x = x / jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))
      x = jnp.where(jnp.isnan(x), 0, x)
      return jnp.sqrt(3) * x

    # sqrt(3) is due to the (2l+1) factor
    return phi
  else:
    raise NotImplementedError(
      "The decomposition of legendre has not been implemented for l > 1."
    )
