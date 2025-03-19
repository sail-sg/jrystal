#!/usr/bin/env python3

import timeit
from functools import partial

import jax
import jax.numpy as jnp
import jrystal
import numpy as np
from dpw.bloch import u
from dpw.utils import post_proc_r_grid, vmap_vector_grid
from dpw.xc import lda_x
from folx import forward_laplacian
from jrystal._src.grid import (
  _half_frequency_pad_to,
  g_vectors,
  half_frequency_shape,
  proper_grid_size,
  r_vectors
)
from jrystal._src.utils import absolute_square
from jrystal.training_utils import create_crystal

config = jrystal.config.get_config()

nO = 2
nG = 5

config.crystal = 'diamond1'
config.grid_sizes = nG

crystal = create_crystal(config)
grid_sizes = proper_grid_size(config.grid_sizes)
half_grid_sizes = half_frequency_shape(grid_sizes)
gs = g_vectors(crystal.cell_vectors, grid_sizes)
rs = r_vectors(crystal.cell_vectors, grid_sizes)

den = lambda c: jnp.fft.fftn(c, axes=(0, 1, 2))

w = np.random.randn(np.prod(half_grid_sizes), nO)
c_dense = jnp.linalg.qr(w)[0].T.reshape((nO, *half_grid_sizes))
c = _half_frequency_pad_to(c_dense, grid_sizes)

orbs = u(crystal.cell_vectors, c, rs, force_fft=True)
# sum over the orbitals densities
rho_r_fn = lambda c_, r_: absolute_square(
  u(crystal.cell_vectors, c_, r_, force_fft=True)
).sum(0)


def rho_r_grad_norm_fn(c_, r_, gs):
  rho_G = jnp.fft.fftn(rho_r_fn(c_, r_))
  grad_rho_G = rho_G[..., None] * 1j * gs.at[0, 0, 0].set(1.)
  grad_rho_r = jnp.fft.ifftn(grad_rho_G, axes=(0, 1, 2))
  return absolute_square(grad_rho_r).sum(-1)


def ex_pbe(rho_r, rho_r_grad_norm):
  kappa, mu = 0.804, 0.21951
  kf = jnp.cbrt(3 * mu**2 * rho_r)
  # reduced_density_gradient
  s = jnp.sqrt(rho_r_grad_norm / (2 * kf * rho_r))
  # enhancement factor
  e_f = 1 + kappa - kappa / (1 + mu * s**2 / kappa)
  return lda_x(rho_r) * e_f


def lapl_r(fn, rs):
  lapl_fn = forward_laplacian(fn)
  result = lapl_fn(rs)
  lapl = result.laplacian
  return lapl


def lapl_G(fn, rs):
  f_G = jnp.fft.fftn(fn(rs))
  return -1 * (gs**2).sum(-1) * f_G


fn = partial(rho_r_fn, c)
lapl_G_1 = jnp.fft.fftn(lapl_r(fn, rs))
lapl_G_2 = lapl_G(fn, rs)
np.allclose(lapl_G_1, lapl_G_2, atol=1e-4)


def vxc_gga_pw_integrand_recp(
  c, rs, gs, exc, rho_r_fn, rho_r_grad_norm_fn, lapl_recp=True
):
  rho_r_ = rho_r_fn(c, rs)
  rho_G = jnp.fft.fftn(rho_r_)
  rho_r_flat = rho_r_.reshape(-1)
  rho_r_grad_norm_ = rho_r_grad_norm_fn(c, rs, gs)
  rho_r_grad_norm_flat = rho_r_grad_norm_.reshape(-1)
  dexc_drho_flat = jax.vmap(jax.grad(exc))(rho_r_flat, rho_r_grad_norm_flat)
  dexc_drho_grad_norm_flat = jax.vmap(jax.grad(exc, argnums=1)
                                     )(rho_r_flat, rho_r_grad_norm_flat)
  if lapl_recp:
    lapl_rho_G = -1 * (gs**2).sum(-1) * rho_G
    lapl_rho_r = jnp.fft.ifftn(lapl_rho_G)
  else:
    lapl_rho_r = lapl_r(partial(rho_r_fn, c), rs)

  t = dexc_drho_grad_norm_flat.reshape(grid_sizes) * rho_r_

  g_axes = list(range(gs.ndim - 1))

  t1 = dexc_drho_flat.reshape(grid_sizes) * rho_r_
  t2 = exc(rho_r_, rho_r_grad_norm_)
  t3 = (
    jnp.fft.ifftn(1j * gs * jnp.fft.fftn(t)[..., None], axes=g_axes) *
    jnp.fft.ifftn(1j * gs * rho_G[..., None], axes=g_axes)
  ).sum(-1)
  t4 = t * lapl_rho_r
  integrand = t1 + t2 - 2 * (t3 + t4)
  vxc_G = jnp.fft.fftn(integrand)
  return vxc_G


vxc_G_1 = vxc_gga_pw_integrand_recp(
  c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn
)
vxc_G_2 = vxc_gga_pw_integrand_recp(
  c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn, lapl_recp=False
)

np.allclose(vxc_G_1, vxc_G_2)


def vxc_gga_pw_integrand_recp_fwd(
  c, rs, gs, exc, rho_r_fn, rho_r_grad_norm_fn, grad_recp=True
):
  rho_r_ = rho_r_fn(c, rs)
  rho_r_flat = rho_r_.reshape(-1)
  rho_r_grad_norm_ = rho_r_grad_norm_fn(c, rs, gs)
  rho_r_grad_norm_flat = rho_r_grad_norm_.reshape(-1)
  dexc_drho_flat = jax.vmap(jax.grad(exc))(rho_r_flat, rho_r_grad_norm_flat)
  dexc_drho_grad_norm_flat = jax.vmap(jax.grad(exc, argnums=1)
                                     )(rho_r_flat, rho_r_grad_norm_flat)

  g_axes = list(range(gs.ndim - 1))

  if grad_recp:
    rho_G = jnp.fft.fftn(rho_r_)
    rho_r_grad = jnp.fft.ifftn(1j * gs * rho_G[..., None], axes=g_axes)
  else:
    rho_r_grad_flat = jax.vmap(jax.grad(partial(rho_r_fn, c)))(
      rs.reshape(-1, 3)
    )
    rho_r_grad = rho_r_grad_flat.reshape((*grid_sizes, 3))

  t = dexc_drho_grad_norm_flat.reshape(grid_sizes) * rho_r_

  t1 = dexc_drho_flat.reshape(grid_sizes) * rho_r_
  t2 = exc(rho_r_, rho_r_grad_norm_)
  t5 = t[..., None] * rho_r_grad

  t5_ = 2 * (jnp.fft.fftn(t5, axes=g_axes) * 1j * gs).sum(-1)
  vxc_G = jnp.fft.fftn(t1 + t2) - t5_

  return vxc_G


vxc_G_3 = vxc_gga_pw_integrand_recp_fwd(
  c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn
)
vxc_G_4 = vxc_gga_pw_integrand_recp_fwd(
  c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn, grad_recp=False
)

np.allclose(vxc_G_3, vxc_G_4, atol=1e-5)

np.allclose(vxc_G_1, vxc_G_3, atol=1e-1)

# benchmark time
f1 = jax.jit(
  lambda c_:
  vxc_gga_pw_integrand_recp(c_, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn)
)
f2 = jax.jit(
  lambda c_: vxc_gga_pw_integrand_recp(
    c_, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn, lapl_recp=False
  )
)
f3 = jax.jit(
  lambda c_: vxc_gga_pw_integrand_recp_fwd(
    c_, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn
  )
)
f4 = jax.jit(
  lambda c_: vxc_gga_pw_integrand_recp_fwd(
    c_, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn, grad_recp=False
  )
)


def get_avg_time(f, x, N=10):
  f(x).block_until_ready()
  avg_time = timeit.timeit(lambda: f(x).block_until_ready(), number=N) / N
  return avg_time


print(get_avg_time(f1, c))
print(get_avg_time(f2, c))
print(get_avg_time(f3, c))
print(get_avg_time(f4, c))
