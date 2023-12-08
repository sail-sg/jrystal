"""The energy density module of exchange-correlation functions.

This module is the the \epsilon(r) as in

.. math::
    E_xc = \int  \epsilon(r) \rho(r) dr

"""
import numpy as np
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Callable

from jrystal._src.jrystal_typing import RealVecterGrid, RealScalar
import jax_xc


def lda_x(r0: Float[Array, "*nd"]):
  t3 = 3**(0.1e1 / 0.3e1)
  t4 = jnp.pi**(0.1e1 / 0.3e1)
  t8 = 2.220446049250313e-16**(0.1e1 / 0.3e1)
  t10 = jnp.where(0.1e1 <= 2.22044604925e-16, t8 * 2.22044604925e-16, 1)
  t11 = r0**(0.1e1 / 0.3e1)
  t15 = jnp.where(r0 / 0.2e1 <= 1e-15, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
  res = 0.2e1 * 1. * t15
  return res


def xc_density(
  density_fn: Callable,
  r_vector_grid: RealVecterGrid,
  vol: RealScalar,
  xc: str = 'lda_x'
) -> Float[Array, '... d']:

  epsilon_xc = getattr(jax_xc, xc, None)
  if epsilon_xc:
    epsilon_xc = epsilon_xc(polarized=True)

  else:
    raise NotImplementedError('xc functional is not implemented')

  num_grid = np.prod(r_vector_grid.shape)

  def e_xc(r):
    return epsilon_xc(density_fn, r)

  rs = jnp.reshape(r_vector_grid, (-1, r_vector_grid.shape[-1]))
  e_xc_grid = jax.vmap(e_xc)(rs)
  e_xc_grid = e_xc_grid.reshape(
    (*r_vector_grid.shape[:-1], *e_xc_grid.shape[1:])
  )
  return e_xc_grid * vol / num_grid
