"""Occupation Modules
"""
import jax.numpy as jnp
import flax.linen as nn
from jrystal._src.module import QRDecomp
from jaxtyping import Int, Array
from jrystal._src.utils import vmapstack
# TODO: since we already has this file named occupations.py,
# (maybe change to singular occupation.py). We don't need to prefix
# the class name with Occ anymore.
# TODO: I prefer defining functions than classes for simple modules like this.
# e.g. occ = jrystal.occupation.uniform(ni, nk, spin) seems more natural than
# occ = jrystal.occupations.OccUniform(ni, nk, spin)()


class Gamma(nn.Module):
  """Occupation module with just Gamma point. 
  Return a mask like this:

        ni
      -----  \n
      |11100 \n
  nk  |00000 \n
      |00000 \n
      |00000 \n

  """

  nk: Int
  ni: Int
  spin: Int

  def __call__(self) -> Int[Array, '2 nk ni']:

    # TODO: currently only support polarized system.

    occ = jnp.zeros([2, self.nk, self.ni])
    occ = occ.at[0, 0, :(self.ni + self.spin) // 2].set(1)
    occ = occ.at[1, 0, :(self.ni - self.spin) // 2].set(1)

    return occ


class Uniform(nn.Module):
  """Uniform occupation number over k points. 
  Returns a mask like:

                ni
        __________          \n
        |1/k 1/k 1/k  0     \n
        |1/k 1/k 1/k  0     \n
    nk  |1/k 1/k 1/k  0     \n
        |1/k 1/k 1/k  0     \n
        |1/k 1/k 1/k  0     \n

  """

  ni: Int
  nk: Int
  spin: Int

  @nn.compact
  def __call__(self) -> Int[Array, '2 nk ni']:

    # TODO: currently only support polarized system.
    occ = jnp.zeros([2, self.nk, self.ni])
    occ = occ.at[0, :, :(self.ni + self.spin) // 2].set(1 / self.nk)
    occ = occ.at[1, :, :(self.ni - self.spin) // 2].set(1 / self.nk)

    return occ


class Orthogonal(nn.Module):
  """orthogonal occupation number over i, k
    Returns a mask like:

                ni
        __________          \n
        |o_11 o_12  0     \n
        |o_21 o_22  0     \n
    nk  |o_31 o_32  0     \n
        |o_41 o_42  0     \n
        |o_51 o_52  0     \n
  
    where \sum_i o_ij^2 = 1
  """
  ni: Int
  nk: Int
  spin: Int

  @nn.compact
  def __call__(self) -> Int[Array, '2 nk ni']:
    occ = jnp.zeros([2, self.nk, self.ni])
    shape = [1, self.nk, (self.ni + self.spin) // 2]
    qr1 = QRDecomp(shape, False, False)()
    pad = vmapstack(2)(lambda x: jnp.pad(x, (0, self.ni - qr1.shape[2])))
    qr1 = pad(qr1)

    shape = [1, self.nk, (self.ni - self.spin) // 2]
    qr2 = QRDecomp(shape, False, False)()
    pad = vmapstack(2)(lambda x: jnp.pad(x, (0, self.ni - qr2.shape[2])))
    qr2 = pad(qr2)
    occ = jnp.vstack((qr1, qr2))
    return occ**2
