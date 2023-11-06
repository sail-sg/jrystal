"""Occupation Modules
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Int, Array

# TODO: the name simple is not informative.
# maybe something like topk ?
# TODO: since we already has this file named occupations.py,
# (maybe change to singular occupation.py). We don't need to prefix
# the class name with Occ anymore.
# TODO: I prefer defining functions than classes for simple modules like this.
# e.g. occ = jrystal.occupation.uniform(ni, nk, spin) seems more natural than
# occ = jrystal.occupations.OccUniform(ni, nk, spin)()


class OccSimple(nn.Module):
  """ Simple Occupation module. Only valid for single k point calculation.
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

  @nn.compact
  def __call__(self) -> Int[Array, '2 nk ni']:

    # TODO: currently only support polarized system.

    occ = jnp.zeros([2, self.nk, self.ni])
    occ = occ.at[0, :, :(self.ni + self.spin) // 2].set(1)
    occ = occ.at[1, :, :(self.ni - self.spin) // 2].set(1)
    occ = jnp.reshape(occ, [2, self.nk, self.ni])

    return occ


class OccUniform(nn.Module):
  """Uniform occupation number over k points. Returns a mask like:

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


# TODO: remove these lines. Introduce a occupation_test.py file for the testings.
if __name__ == '__main__':
  import jrystal
  str = 'OccSimple'
  occ = getattr(jrystal.occupations, str)(1, 4, 0)
  key = jax.random.PRNGKey(123)
  params = occ.init(key)
  print(occ.apply(params))

  print(type(occ))
