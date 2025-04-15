"""PBE exchange-correlation functional."""

import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax_xc.impl.gga_x_pbe import unpol as pbe_jac_xc
from jax_xc.utils import get_p
from jaxtyping import Array, Float

from ..utils import absolute_square


def vxc_gga_recp(exc, rho_r, rho_r_grad_norm, gs):
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


def _pbe_x(rho_r, rho_r_grad_norm):
  """wrapper to low level API of jax_xc"""
  p = get_p("gga_x_pbe", 1)
  grid_shape = rho_r.shape
  rho_r_flat = rho_r.reshape(-1)
  rho_r_grad_norm_flat = rho_r_grad_norm.reshape(-1)
  out = jax.vmap(pbe_jac_xc, (None, 0, 0),
                 0)(p, rho_r_flat, rho_r_grad_norm_flat)
  return out.reshape(grid_shape)


def rho_r_grad_norm_fn(density_grid, g_vector_grid):
  rho_G = jnp.fft.fftn(density_grid)
  grad_rho_G = rho_G[..., None] * 1j * g_vector_grid
  grad_rho_r = jnp.fft.ifftn(grad_rho_G, axes=(0, 1, 2))
  return absolute_square(grad_rho_r).sum(-1)


def pbe_x(
  density_grid: Float[Array, 'x y z'],
  g_vector_grid,
  kohn_sham: bool = False
) -> Float[Array, 'x y z']:
  """
  Calculate the PBE exchange-correlation potential.

  Warning: currently only support spin-restricted calculation.

  Args:
    density_grid (Float[Array, 'x y z']): Real-space electron density.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.
  """
  assert density_grid.ndim == 3, "Currently only support spin-restricted calculation."

  grad_norm = rho_r_grad_norm_fn(density_grid, g_vector_grid)

  if kohn_sham:  # return vxc
    # NOTE: v_eff can be calculated in reciprocal space
    density_grid = stop_gradient(density_grid)
    vxc_G = vxc_gga_recp(_pbe_x, density_grid, grad_norm, g_vector_grid)
    vxc_r = jnp.fft.ifftn(vxc_G)
    return vxc_r
  else:
    return _pbe_x(density_grid, grad_norm)
