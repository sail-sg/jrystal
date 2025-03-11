"""Spherical coordinate system utilities and spherical harmonics transformations.

This module provides functions for working with spherical coordinates and spherical harmonics, particularly useful for quantum mechanical calculations and pseudopotential transformations.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import sph_harm
from jaxtyping import Float, Array
from typing import Callable
from .._src.utils import vmapstack


def cartesian_to_spherical(x: Float[Array, "*n 3"]) -> Float[Array, "*n 3"]:
  """Convert Cartesian coordinates to spherical coordinates.

  Transforms 3D Cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ), where:
  
  - r is the radial distance from the origin
  - θ (theta) is the azimuthal angle in the x-y plane from the x-axis (0 ≤ θ < 2π)
  - φ (phi) is the polar angle from the z-axis (0 ≤ φ ≤ π)

  For the special case of the origin (0, 0, 0), returns (0, NaN, π/2).

  Args:
    x: Float[Array, "*n 3"]: Cartesian coordinates with shape (..., 3) where ... represents arbitrary batch dimensions

  Returns:
    Float[Array, "*n 3"]: Spherical coordinates (r, θ, φ) with same batch shape
  """
  r = jnp.linalg.norm(x, axis=-1)  # Radial distance

  # Polar angle (phi)
  phi = jnp.arccos(jnp.clip(x[2] / r, -1.0, 1.0))

  # Azimuthal angle (theta) and shift to range [0, 2*pi)
  theta = jnp.arctan2(x[1], x[0])
  theta = jnp.mod(theta + 2 * jnp.pi, 2 * jnp.pi)

  return jnp.stack((r, theta, phi), axis=-1)


def legendre_to_sph_harm(
  l: int = 0,
) -> Callable[[Float[Array, "*batch 3"]], Float[Array, "*batch m"]]:
  """Convert Legendre polynomials to spherical harmonics decomposition.

  Implements the decomposition of Legendre polynomials into spherical harmonics according to the formula:

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
      Callable that takes Cartesian coordinates of shape (*batch, 3) and returns spherical harmonics coefficients of shape (*batch, m), where m = 2l+1
  """
  m = jnp.arange(-l, l + 1)
  n = jnp.array([l])
  L_MAX = 4

  def fun(x):

    @vmapstack(x.ndim - 1)
    def _f(x_i):
      x_spherical = cartesian_to_spherical(x_i)
      theta = x_spherical[..., 1:2]  # azimuthal angle
      phi = x_spherical[..., 2:3]  # polar angle

      y_lm = jax.vmap(
        sph_harm, in_axes=[0, None, None, None]
      )(m, n, theta, phi).reshape([-1])  # [m]

      y_lm = jnp.pad(y_lm, (0, (L_MAX - l) * 2), constant_values=0)
      # pad the y_lm to length l_max with zeros
      return y_lm * 2 * jnp.sqrt(jnp.pi)  # [m]

    return _f(x)

  return fun
