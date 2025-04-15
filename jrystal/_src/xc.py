import importlib
from typing import Callable

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax_xc.utils import get_p
from jaxtyping import Array, Float

from .utils import absolute_square, safe_real


def get_xc_functional(xc: str = 'gga_x_pbe', polarized: bool = False):
  """Dynamically import XC functional implementation.

    Args:
        functional_type: String like 'pbe', 'b3lyp', etc.
        polarization: String, either 'unpol' or 'pol' for unpolarized/polarized

    Returns:
        The requested functional implementation
    """
  polarization = 'pol' if polarized else 'unpol'
  try:
    module_path = f"jax_xc.impl.{xc}"
    module = importlib.import_module(module_path)
    functional = getattr(module, polarization)
    return functional
  except (ImportError, AttributeError) as e:
    raise ImportError(
      f"Could not import {polarization} from {module_path}: {e}"
    )


# Constants in libxc
XC_UNPOLARIZED = 0
XC_POLARIZED = 1


def _lda(xc_type: str, rho_r: Float[Array, 's x y z']):
  polarized = rho_r.shape[0] == 2
  p = get_p(xc_type, XC_POLARIZED if polarized else XC_UNPOLARIZED)
  exc = get_xc_functional(xc_type, polarized=polarized)
  grid_shape = rho_r.shape
  rho_r_flat = rho_r.reshape(-1)
  out = jax.vmap(exc, (None, 0), 0)(p, rho_r_flat)
  return out.reshape(grid_shape)


def _gga(
  xc_type: str,
  rho_r: Float[Array, 's x y z'],
  rho_r_grad: Float[Array, 's x y z']
):
  polarized = rho_r.shape[0] == 2
  p = get_p(xc_type, XC_POLARIZED if polarized else XC_UNPOLARIZED)
  exc = get_xc_functional(xc_type, polarized=polarized)
  grid_shape = rho_r.shape
  rho_r_flat = rho_r.reshape(-1)
  rho_r_grad_flat = rho_r_grad.reshape(-1)
  out = jax.vmap(exc, (None, 0, 0), 0)(p, rho_r_flat, rho_r_grad_flat)
  return out.reshape(grid_shape)


def vxc_lda(exc: Callable, rho_r: Float[Array, 's x y z']):
  polarized = rho_r.shape[0] == 2
  grid_sizes = rho_r.shape

  if polarized:
    pass
  else:
    rho_r_flat = rho_r.reshape(-1)
    dexc_drho_flat = jax.vmap(jax.grad(exc))(rho_r_flat)
    t1 = dexc_drho_flat.reshape(grid_sizes) * rho_r
    t2 = exc(rho_r)
    vxc_r = t1 + t2

  return vxc_r


def vxc_gga_recp(
  exc: Callable,
  rho_r: Float[Array, 'x y z'],
  rho_r_grad_norm: Float[Array, 'x y z'],
  gs: Float[Array, 'x y z 3']
):
  rho_G = jnp.fft.fftn(rho_r)
  rho_r_flat = rho_r.reshape(-1)
  rho_r_grad_norm_flat = rho_r_grad_norm.reshape(-1)
  dexc_drho_flat = jax.vmap(jax.grad(exc))(rho_r_flat, rho_r_grad_norm_flat)
  dexc_drho_grad_norm_flat = jax.vmap(jax.grad(exc, argnums=1)
                                     )(rho_r_flat, rho_r_grad_norm_flat)
  lapl_rho_G = -1 * (gs**2).sum(-1) * rho_G
  lapl_rho_r = jnp.fft.ifftn(lapl_rho_G)

  grid_sizes = gs.shape[:-1]
  t = dexc_drho_grad_norm_flat.reshape(grid_sizes) * rho_r
  t = jnp.where(rho_r_grad_norm > 0, t, 0)

  g_axes = list(range(gs.ndim - 1))

  t1 = dexc_drho_flat.reshape(grid_sizes) * rho_r
  t2 = exc(rho_r, rho_r_grad_norm)
  t3 = (
    jnp.fft.ifftn(1j * gs * jnp.fft.fftn(t)[..., None], axes=g_axes) *
    jnp.fft.ifftn(1j * gs * rho_G[..., None], axes=g_axes)
  ).sum(-1)
  t4 = t * lapl_rho_r
  integrand = t1 + t2 - 2 * (t3 + t4)
  vxc_G = jnp.fft.fftn(integrand)
  return vxc_G


def vxc_gga(
  exc: Callable,
  rho_r: Float[Array, 'x y z'],
  rho_r_grad_norm: Float[Array, 'x y z'],
  gs: Float[Array, 'x y z 3']
):
  return jnp.fft.ifftn(vxc_gga_recp(exc, rho_r, rho_r_grad_norm, gs))


def rho_r_grad_fn(
  density_grid: Float[Array, 's x y z'], gs: Float[Array, 'x y z 3']
) -> Float[Array, 's x y z']:
  """If polarized, s in the return shape is 3, else 1"""
  polarized = density_grid.shape[0] == 2
  rho_G = jnp.fft.fftn(density_grid)
  grad_rho_G = rho_G[..., None] * 1j * gs
  grad_rho_r = jnp.fft.ifftn(grad_rho_G, axes=range(-3, 0))
  if polarized:
    s_alpha, s_beta = grad_rho_r
    return safe_real(
      jnp.stack([
        s_alpha * s_alpha,
        s_alpha * s_beta,
        s_beta * s_beta,
      ])
    ).sum(-1)
  else:
    return absolute_square(grad_rho_r).sum(-1)


def xc_density(
  density_grid: Float[Array, 's x y z'],
  g_vector_grid,
  kohn_sham: bool = False,
  xc_type: str = "lda_x"
):
  if "gga" in xc_type:
    exc_fn = lambda density, grad: sum(
      [_gga(xc_type_, density, grad) for xc_type_ in xc_type.split('+')]
    )
    grad = rho_r_grad_fn(density_grid, g_vector_grid)

    if kohn_sham:
      density_grid = stop_gradient(density_grid)
      vxc_r = vxc_gga(exc_fn, density_grid, grad, g_vector_grid)
      return vxc_r

    else:
      return exc_fn(density_grid, grad)

  else:  # LDA
    exc_fn = lambda density: sum(
      [_lda(xc_type_, density) for xc_type_ in xc_type.split('+')]
    )

    if kohn_sham:
      density_grid = stop_gradient(density_grid)
      vxc_r = vxc_lda(exc_fn, density_grid)
      return vxc_r

    else:
      return exc_fn(density_grid)
