"""the energy density of exchange-correlation functions.

This module is the the \epsilon(r) as in

.. math::
    E_xc = \int  \epsilon(r) \rho(r) dr

"""
import jax.numpy as jnp
from jaxtyping import Float, Array


def lda_x(r0: Float[Array, "*nd"]):
  t3 = 3**(0.1e1 / 0.3e1)
  t4 = jnp.pi**(0.1e1 / 0.3e1)
  t8 = 2.220446049250313e-16**(0.1e1 / 0.3e1)
  t10 = jnp.where(0.1e1 <= 2.22044604925e-16, t8 * 2.22044604925e-16, 1)
  t11 = r0**(0.1e1 / 0.3e1)
  t15 = jnp.where(r0 / 0.2e1 <= 1e-15, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
  res = 0.2e1 * 1. * t15
  return res
