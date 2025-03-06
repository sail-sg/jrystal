"""Occupation module."""
from typing import Optional

import einops
import jax.numpy as jnp
from jaxtyping import Int

from ._typing import OccupationArray
from .unitary_module import unitary_matrix, unitary_matrix_param_init


def idempotent_param_init(key, num_bands, num_kpts):
  return unitary_matrix_param_init(
    key, [num_bands * num_kpts, num_bands * num_kpts], complex=False
  )


def idempotent(
  params,
  num_electrons,
  num_kpts,
  spin=0,
  spin_restricted=True,
) -> OccupationArray:
  num_bands = params["w_re"].shape[0] // num_kpts

  def o(num_e):
    u = unitary_matrix(params, False)
    indices = jnp.arange(num_bands * num_kpts)
    mask_ones = indices < num_e * num_kpts
    mask_fraction = indices == jnp.floor(num_e * num_kpts)
    fraction = num_e * num_kpts - jnp.floor(num_e * num_kpts)
    f = jnp.where(mask_ones, 1.0, jnp.where(mask_fraction, fraction, 0.0))
    occ = einops.einsum(u, f, u.T, "nk ik, ik, ik nk -> nk")
    return occ.reshape([num_kpts, num_bands]) / num_kpts

  occ = jnp.stack(
    [o((num_electrons - spin) // 2), o((num_electrons + spin) // 2)], axis=0
  )

  if spin_restricted is False:
    return occ
  else:
    return jnp.sum(occ, axis=0, keepdims=True)


def uniform(
  num_k: Int,
  num_electrons: Int,
  spin: Int = 0,
  num_bands: Optional[Int] = None,
  spin_restricted: bool = True
) -> OccupationArray:
  """uniform occupation.

    Returns a mask like:

          num_bands
         ______________     \n
        |1/k 1/k 1/k  0     \n
        |1/k 1/k 1/k  0     \n
  num_k |1/k 1/k 1/k  0     \n
        |1/k 1/k 1/k  0     \n
        |1/k 1/k 1/k  0     \n

  where k is the number of k points.

  Args:
      num_k (Int): the number of k points.
      num_electrons (Int): the number of electrons.
      spin (Int, optional): the number of unpaired electrons. Defaults to 0.
      num_bands (Int | None, optional): the number of bands(orbitals). If
        num_bands is not provided, then num_bands is the same as num_electrons.
        Defaults to None.
      spin_restricted (bool, optional): indicate of spin channel. If True, the
        first axis of output is 2. If False, the first axis of output is 1.
        Defaults to True.

  Returns:
      OccupationArray: an occupation musk with shape [s, num_k, num_bands],
        where s=2 if spin_restricted is True, else s=1. The sum of occupation mask
        equals to num_electrons.
  """
  num_bands = num_electrons if num_bands is None else num_bands

  occ = jnp.zeros([2, num_k, num_bands])
  occ = occ.at[0, :, :(num_electrons + spin) // 2].set(1 / num_k)
  occ = occ.at[1, :, :(num_electrons - spin) // 2].set(1 / num_k)

  if spin_restricted:
    return jnp.sum(occ, axis=0, keepdims=True)

  return occ


def gamma(
  num_k: Int,
  num_electrons: Int,
  spin: Int = 0,
  num_bands: Optional[Int] = None,
  spin_restricted: bool = True
) -> OccupationArray:
  """occupation on Gamma point.
  Return a mask like this:

                num_bands
               ----- \n
              |11100 \n
      num_k   |00000 \n
              |00000 \n
              |00000 \n

  Args:
      num_k (Int): the number of k points.
      num_electrons (Int): the number of electrons.
      spin (Int, optional): the number of unpaired electrons. Defaults to 0.
      num_bands (Int | None, optional): the number of bands(orbitals). If
        num_bands is not provided, then num_bands is the same as num_electrons.
        Defaults to None.
      spin_restricted (bool, optional): indicate of spin channel. If True, the
        first axis of output is 2. If False, the first axis of output is 1.
        Defaults to True.

  Returns:
      OccupationArray: an occupation musk with shape [s, num_k, num_bands],
        where s=2 if spin_restricted is True, else s=1. The sum of occupation mask
        equals to num_electrons.

  """
  num_bands = num_electrons if num_bands is None else num_bands
  occ = jnp.zeros([2, num_k, num_bands])
  occ = occ.at[0, 0, :(num_electrons + spin) // 2].set(1)
  occ = occ.at[1, 0, :(num_electrons - spin) // 2].set(1)

  if spin_restricted:
    return jnp.sum(occ, axis=0, keepdims=True)

  return occ
