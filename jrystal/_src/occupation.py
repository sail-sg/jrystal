"""Occupation Modules
"""
import jax.numpy as jnp
import flax.linen as nn
from jrystal._src.module import QRDecomp
from typing import Union
from jaxtyping import Int, Array, Float
from jrystal._src.jrystal_typing import OccupationArray
# TODO: since we already has this file named occupations.py,
# (maybe change to singular occupation.py). We don't need to prefix
# the class name with Occ anymore.
# TODO: I prefer defining functions than classes for simple modules like this.
# e.g. occ = jrystal.occupation.uniform(ni, nk, spin) seems more natural than
# occ = jrystal.occupations.OccUniform(ni, nk, spin)()


def occupation_gamma(nk: int, ni: int, spin: int, nb=None, polarize=True):
  nb = ni if nb is None else nb
  occ = jnp.zeros([2, nk, nb])
  occ = occ.at[0, 0, :(ni + spin) // 2].set(1)
  occ = occ.at[1, 0, :(ni - spin) // 2].set(1)

  if polarize is False:
    return jnp.sum(occ, axis=0, keepdims=True)

  return occ


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

  num_k: Int
  num_i: Int
  nspin: Int

  def __call__(self, *args, **kwargs) -> OccupationArray:
    return occupation_gamma(self.num_k, self.num_i, self.nspin)


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

  nk: Int
  ni: Int
  spin: Int

  @nn.compact
  def __call__(self, *args, **kwargs) -> OccupationArray:

    # TODO: currently only support polarized system.
    occ = jnp.zeros([2, self.nk, self.ni])
    occ = occ.at[0, :, :(self.ni + self.spin) // 2].set(1 / self.nk)
    occ = occ.at[1, :, :(self.ni - self.spin) // 2].set(1 / self.nk)

    return occ


class Orthogonal(nn.Module):
  """orthogonal occupation number over i, k
    Returns a mask like:

                ni
        __________        \n
        |o_11 o_12  0     \n
        |o_21 o_22  0     \n
    nk  |o_31 o_32  0     \n
        |o_41 o_42  0     \n
        |o_51 o_52  0     \n

    where \sum_i o_ij^2 = 1

  """
  nk: Int
  ni: Int
  spin: Int

  @nn.compact
  def __call__(self, *args, **kwargs) -> OccupationArray:
    if self.nk < (self.ni - self.spin) // 2:
      raise ValueError(
        'orthogonal occupation only support when the number of ',
        f'k points (now is {self.nk}) larger than number of ',
        f'orbitals (now is {(self.ni - self.spin) // 2})'
      )

    occ = jnp.zeros([2, self.nk, self.ni])

    shape = [1, self.nk, (self.ni + self.spin) // 2]
    qr1 = QRDecomp(shape, False, False)()
    occ = occ.at[0, :, :(self.ni + self.spin) // 2].set(qr1)

    shape = [1, self.nk, (self.ni - self.spin) // 2]
    qr2 = QRDecomp(shape, False, False)()
    occ = occ.at[1, :, :(self.ni - self.spin) // 2].set(qr2)

    return occ**2


def fermi_dirac(
  eigenvalues: Float[Array, '...'],
  fermi_level: Union[float, Float[Array, '']],
  width: float
) -> OccupationArray:
  """Fermi-Dirac distribution function.

    .. math::
      o_i = 1 / (exp^((eigenvalue_i - fermi_level) / width) + 1)

    Args:

      eigenvalues (Float[Array, '...']): an array of eigenvalues of any shape.
      fermi_level (Union[float, Float[Array, '']]): fermi level.
      width (float): the width.

    Returns:
      jax.Array: the occpuation numbers with the same shape of input
        eigenvalues.

    """
  o = (eigenvalues - fermi_level) / width
  o = jnp.clip(o, -100, 100)
  o = jnp.exp(o) + 1
  o = 1 / o
  return o


# TODO: Support symmetry.
def get_fermi_level(
  eigenvalues: Float[Array, '... num_k num_band'],
  num_electrons: int,
) -> Float[Array, '']:
  """get the fermi level given a array of eigenvalues.

  Args:

    eigenvalues (Float[Array, '... num_k num_band']): an array of eigenvalues.
      The dimension of the last two axes are num_k and num_band.
    num_electrons (int): number of electrons


  Returns:

      float: the fermi level.
  """

  num_k = eigenvalues.shape[-2]
  eigenvalues = eigenvalues.flatten().sort()  # ascending order.
  return eigenvalues[num_k * num_electrons - 1]


class FermiDirac(nn.Module):
  num_electrons: int
  width: float

  @nn.compact
  def __call__(
    self, eigenvalues: Float[Array, '... num_k num_band']
  ) -> OccupationArray:
    num_k = eigenvalues.shape[-2]
    fermi_level = get_fermi_level(eigenvalues, self.num_electrons)
    return fermi_dirac(eigenvalues, fermi_level, self.width) / num_k
