import importlib
from typing import Callable, Optional

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


def _parse_xc_type(xc_type: str, rho_r: Float[Array, 's x y z']):
  polarized = rho_r.ndim > 0 and rho_r.shape[0] == 2
  is_exchange = '_x' in xc_type  # HACK
  if is_exchange:  # NOTE: libxc's polarized exchange functionals might be problematic
    p = get_p(xc_type, XC_UNPOLARIZED)
    f = get_xc_functional(xc_type, polarized=False)
  else:
    p = get_p(xc_type, XC_POLARIZED if polarized else XC_UNPOLARIZED)
    f = get_xc_functional(xc_type, polarized=polarized)
  grid_shape = rho_r.shape[1:]
  rho_r_flat = rho_r.reshape(2 if polarized else 1, -1)
  return polarized, is_exchange, p, f, grid_shape, rho_r_flat


def _lda(xc_type: str, rho_r: Float[Array, 's x y z']):
  polarized, is_exchange, p, f, grid_shape, rho_r_flat = _parse_xc_type(xc_type, rho_r)
  if polarized and is_exchange:  # NOTE: exact scaling relation
    f_up = jax.vmap(f, (None, 0), 0)(p, 2 * rho_r_flat[0])
    f_dn = jax.vmap(f, (None, 0), 0)(p, 2 * rho_r_flat[1])
    f_val = 0.5 * (f_up + f_dn)
  elif polarized:  # and is correlation
    f_val = jax.vmap(f, (None, 1), 0)(p, rho_r_flat)
  else:  # not polarized
    f_val = jax.vmap(f, (None, 0), 0)(p, rho_r_flat[0])
  return f_val.reshape(grid_shape)


def _gga(
  xc_type: str,
  rho_r: Float[Array, 's x y z'],
  sigma_r: Float[Array, 's x y z']
):
  polarized, is_exchange, p, f, grid_shape, rho_r_flat = _parse_xc_type(xc_type, rho_r)
  sigma_r_flat = sigma_r.reshape(3 if polarized else 1, -1)
  if polarized and is_exchange:  # NOTE: exact scaling relation
    f_up = jax.vmap(f, (None, 0, 0),
                    0)(p, 2 * rho_r_flat[0], 4 * sigma_r_flat[0])
    # NOTE: sigma_r_flat[1] is the cross term
    f_dn = jax.vmap(f, (None, 0, 0),
                    0)(p, 2 * rho_r_flat[1], 4 * sigma_r_flat[2])
    f_val = 0.5 * (f_up + f_dn)
  elif polarized:  # and is correlation
    f_val = jax.vmap(f, (None, 1, 1), 0)(p, rho_r_flat, sigma_r_flat)
  else:  # not polarized
    f_val = jax.vmap(f, (None, 0, 0), 0)(p, rho_r_flat[0], sigma_r_flat[0])
  return f_val.reshape(grid_shape)


def sigma_r_fn(
  density_grid: Float[Array, 's x y z'], gs: Float[Array, 'x y z 3']
) -> Float[Array, 's x y z']:
  """Calculate the gradient norms of the density.
  If polarized, s in the return shape is 3, else 1, as for polarized density
  we compute the gradient norm of up and down spin, and the cross term."""
  polarized = density_grid.shape[0] == 2
  rho_G = jnp.fft.fftn(density_grid, axes=range(-3, 0))
  grad_rho_G = rho_G[..., None] * 1j * gs
  grad_rho_r = jnp.fft.ifftn(grad_rho_G, axes=range(-4, -1))
  if polarized:
    s_up, s_dn = grad_rho_r
    return jnp.stack([
      s_up.conj() * s_up,
      s_up.conj() * s_dn,
      s_dn.conj() * s_dn,
    ]).sum(-1).real
  else:
    return absolute_square(grad_rho_r).sum(-1)


def vxc_lda(exc: Callable, rho_r: Float[Array, 's x y z']):
  polarized = rho_r.shape[0] == 2
  grid_sizes = rho_r.shape[1:]

  rho_r_flat = rho_r.reshape(rho_r.shape[0], -1)
  if polarized:
    dexc_drho_flat = jax.vmap(jax.jacfwd(exc), 1, 1)(rho_r_flat)
    t1 = dexc_drho_flat.reshape(2, *grid_sizes) * rho_r
  else:
    dexc_drho_flat = jax.vmap(jax.grad(exc))(rho_r_flat[0])
    t1 = dexc_drho_flat.reshape(1, *grid_sizes) * rho_r

  t2 = exc(rho_r)
  vxc_r = t1 + t2

  return vxc_r


def vxc_gga_recp(
  exc: Callable,
  rho_r: Float[Array, 's x y z'],
  sigma_r: Float[Array, 's x y z'],
  gs: Float[Array, 'x y z 3']
):
  """Calculate the function derivative of the XC functional in the
  reciprocal space.

  Note that the correlation part are treated differently

  Args:
    sigma_r: gradient norm of the density, evaluated on the grid
  """
  polarized = rho_r.shape[0] == 2
  grid_sizes = rho_r.shape[1:]

  # local term
  rho_r_flat = rho_r.reshape(rho_r.shape[0], -1)
  sigma_r_flat = sigma_r.reshape(sigma_r.shape[0], -1)
  if polarized:
    dexc_drho_flat = jax.vmap(jax.jacfwd(exc), (1, 1),
                              1)(rho_r_flat, sigma_r_flat)
    t1 = dexc_drho_flat.reshape(2, *grid_sizes) * rho_r
  else:
    dexc_drho_flat = jax.vmap(jax.grad(exc))(rho_r_flat[0], sigma_r_flat[0])
    t1 = dexc_drho_flat.reshape(1, *grid_sizes) * rho_r

  t2 = exc(rho_r, sigma_r)
  local_term = t1 + t2

  # gradient correction
  axes = list(range(-3, 0))
  rho_G: Float[Array, 's x y z'] = jnp.fft.fftn(rho_r, axes=axes)
  grad_rho_r = jnp.fft.ifftn(rho_G[..., None] * 1j * gs, axes=axes)
  lapl_rho_G = -1 * (gs**2).sum(-1) * rho_G
  lapl_rho_r: Float[Array, 's x y z'] = jnp.fft.ifftn(lapl_rho_G, axes=axes)

  grad_fft = lambda x: jnp.fft.ifftn(1j * gs * jnp.fft.fft(x)[..., None], axes=axes)

  if polarized:
    dexc_dsigma_flat: Float[Array, '3 num_g']
    dexc_dsigma_flat = jax.vmap(jax.jacfwd(exc, argnums=1), (1, 1),
                                1)(rho_r_flat, sigma_r_flat)

    t_up = dexc_dsigma_flat[0].reshape(grid_sizes) * rho_r[0]
    t_up = jnp.where(sigma_r[0] > 0, t_up, 0)
    t3_up = (grad_fft(t_up) * grad_rho_r[0]).sum(-1)
    t4_up = t_up * lapl_rho_r[0]

    t_cross_up = dexc_dsigma_flat[1].reshape(grid_sizes) * rho_r[1]
    t5_up = (grad_fft(t_cross_up) * grad_rho_r[1]).sum(-1)
    t6_up = t_cross_up * lapl_rho_r[1]

    grad_correction_up = 2 * (t3_up + t4_up) + t5_up + t6_up

    t_dn = dexc_dsigma_flat[2].reshape(grid_sizes) * rho_r[1]
    t_dn = jnp.where(sigma_r[2] > 0, t_dn, 0)
    t3_dn = (grad_fft(t_dn) * grad_rho_r[1]).sum(-1)
    t4_dn = t_dn * lapl_rho_r[1]

    t_cross_dn = dexc_dsigma_flat[1].reshape(grid_sizes) * rho_r[0]
    t5_dn = (grad_fft(t_cross_dn) * grad_rho_r[0]).sum(-1)
    t6_dn = t_cross_dn * lapl_rho_r[0]

    grad_correction_dn = 2 * (t3_dn + t4_dn) + t5_dn + t6_dn

    grad_correction = jnp.stack([grad_correction_up, grad_correction_dn])

  else:
    dexc_dsigma_flat: Float[Array, 'num_g']
    dexc_dsigma_flat = jax.vmap(jax.grad(exc, argnums=1)
                               )(rho_r_flat[0], sigma_r_flat[0])
    t = dexc_dsigma_flat.reshape(grid_sizes) * rho_r[0]
    t = jnp.where(sigma_r[0] > 0, t, 0)
    t3 = (grad_fft(t) * grad_rho_r[0]).sum(-1)
    t4 = t * lapl_rho_r
    grad_correction = 2 * (t3 + t4)

  integrand = local_term - grad_correction

  vxc_G = jnp.fft.fftn(integrand, axes=axes)

  return vxc_G


def vxc_gga(
  exc: Callable,
  rho_r: Float[Array, 'x y z'],
  sigma_r: Float[Array, 'x y z'],
  gs: Float[Array, 'x y z 3']
):
  return jnp.fft.ifftn(vxc_gga_recp(exc, rho_r, sigma_r, gs), axes=list(range(-3, 0)))


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
    grad = sigma_r_fn(density_grid, g_vector_grid)

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
