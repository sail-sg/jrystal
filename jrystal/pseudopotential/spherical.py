"""Functions related to spherical coordinate system. """
import jax
import jax.numpy as jnp
from jax.scipy.special import sph_harm
from jaxtyping import Float, Array
from typing import Callable
from .._src.utils import vmapstack


def cartesian_to_spherical(x: Float[Array, "*nd 3"]) -> Float[Array, "*nd 3"]:
  """Transform the cartesian coordinate to spherical one.

    the spherical coordinates for the origin (0, 0, 0) are (0, nan, \pi/2).

  Returns:
      Float[Array, "*nd 3"]: the spherical coordinate (r, theta, phi)
      theta is the azimuthal angle, and phi is the polar angle.
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
) -> Callable[[Float[Array, "*batch 3"]], Float[Array, "*batch m"]]:  # noqa
  """
    Decompose the legendre polynomial into spherical harmonics:

     (2l+1) P_l(x^Ty) = 4\pi \sum_m Y_{l, m}(x) Y*_{l, m}(y)

    where l is the index for angular momentum, and Y are the sperical harmonics.

    - When we need this function?

      when we need to calculate P_l(G^T G'). If we compute the every pair of the
      inner products between two G vections, it is very expensive. Instead, we 
      can decompose the Legendre polynomial into spherical harmonics, and then 
      compute the spherical harmonics for each G vector and calculate the inner 
      products of these spherical harmonics. This is much faster.

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
