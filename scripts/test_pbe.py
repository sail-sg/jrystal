import timeit
from functools import partial

import jax
import jax.numpy as jnp
import jrystal
import numpy as np
from folx import forward_laplacian
from jaxtyping import Array, Float
from jrystal._src.grid import (
  _half_frequency_pad_to,
  g_vectors,
  half_frequency_shape,
  proper_grid_size,
  r_vectors
)
from jrystal._src.utils import absolute_square
from jrystal.bloch import u
from jrystal.calc.opt_utils import create_crystal


def lda_x(r0: Float[Array, "*nd"]):
  t3 = 3**(0.1e1 / 0.3e1)
  t4 = jnp.pi**(0.1e1 / 0.3e1)
  t8 = 2.220446049250313e-16**(0.1e1 / 0.3e1)
  t10 = jnp.where(0.1e1 <= 2.22044604925e-16, t8 * 2.22044604925e-16, 1)
  t11 = r0**(0.1e1 / 0.3e1)
  t15 = jnp.where(r0 / 0.2e1 <= 1e-15, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
  res = 0.2e1 * 1. * t15
  return res


def post_proc_r_grid(A, r_vector_grid):
  """Ensures that the r vector grid do not go outside of the unit cell defined
  by the cell vector `a`. This is crucial for ensuring that the flow density
  does not vanish as evaluation point goes out of bound.
  """
  dim = len(A)
  coeffs = jnp.linalg.solve(jnp.expand_dims(A.T, range(0, dim)), r_vector_grid)

  # NOTE: due to the way r grid is constructed, it always goes out of bound
  # at the negative side
  if coeffs.min() < -0.5:
    correction = A.T @ jnp.ones(dim) * (coeffs.min() % -0.5)
    return r_vector_grid - correction
  return r_vector_grid


rho_r_fn = lambda c_, r_: absolute_square(
  u(crystal.cell_vectors, c_, r_, force_fft=True)
).sum(0)


def rho_r_grad_norm_fn_ad(c_, r_):
  """ad version"""
  grid_shape = r_.shape[:-1]
  r_flat = r_.reshape(-1, 3)
  grad_norm_flat = jax.vmap(
    lambda r__: jnp.linalg.norm(
      jax.grad(
        lambda r:
        absolute_square(u(crystal.cell_vectors, c_, r, force_fft=False)).sum(0)
      )(r__)
    )**2
  )(
    r_flat
  )
  grad_norm = grad_norm_flat.reshape(grid_shape)
  return grad_norm


def rho_r_grad_norm_fn(c_, r_, gs):
  rho_G = jnp.fft.fftn(rho_r_fn(c_, r_))
  grad_rho_G = rho_G[..., None] * 1j * gs
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


if __name__ == '__main__':
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

  grad_norm_ad = rho_r_grad_norm_fn_ad(c, rs)
  grad_norm = rho_r_grad_norm_fn(c, rs, gs)

  np.allclose(grad_norm, grad_norm_ad)

  orbs = u(crystal.cell_vectors, c, rs, force_fft=True)
  # sum over the orbitals densities

  fn = partial(rho_r_fn, c)
  lapl_G_1 = jnp.fft.fftn(lapl_r(fn, rs))
  lapl_G_2 = lapl_G(fn, rs)
  np.allclose(lapl_G_1, lapl_G_2, atol=1e-4)

  vxc_G_1 = vxc_gga_pw_integrand_recp(
    c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn
  )
  vxc_G_2 = vxc_gga_pw_integrand_recp(
    c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn, lapl_recp=False
  )

  vxc_G_3 = vxc_gga_pw_integrand_recp_fwd(
    c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn
  )
  vxc_G_4 = vxc_gga_pw_integrand_recp_fwd(
    c, rs, gs, ex_pbe, rho_r_fn, rho_r_grad_norm_fn, grad_recp=False
  )

  np.allclose(vxc_G_1, vxc_G_2)

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
