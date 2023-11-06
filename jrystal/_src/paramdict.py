"""Dataclasses of arguments."""

import jax
import chex
import numpy as np

from jaxtyping import Int, Float
from typing import List
from jrystal import Crystal
from jrystal._src.grid import _grid_sizes, g_vectors, r_vectors
from jrystal._src.pw import _get_mask_radius

from ase.dft.kpoints import monkhorst_pack


@chex.dataclass
class PWDArgs:
  shape: jax.Array
  mask: jax.Array
  A: jax.Array
  k_grid: jax.Array
  spin: Int
  vol: float

  @staticmethod
  def get_PWD_args(
    crystal: Crystal,
    Ecut: Float,
    g_grid_sizes: Int | List | jax.Array,  # noqa: F821
    k_grid_sizes,
    occ: str = 'simple',
    polarize: bool = True
  ):

    crystal = crystal
    Ecut = Ecut
    polarize = polarize
    g_grid_sizes = _grid_sizes(g_grid_sizes)
    k_grid_sizes = _grid_sizes(k_grid_sizes)
    occ = occ
    nspin = 2 if polarize else 1
    ni = crystal.nelec
    nk = np.prod(k_grid_sizes)
    spin = int(ni % 2)
    mask = _get_mask_radius(crystal.A, g_grid_sizes, Ecut)

    ng = np.sum(mask)
    k_grid = monkhorst_pack(k_grid_sizes)
    g_grid = g_vectors(crystal.A, g_grid_sizes)
    r_grid = r_vectors(crystal.A, g_grid_sizes)
    cg_shape = [nspin, nk, ng, ni.item()]

    pwd_args = PWDArgs(
      shape=cg_shape,
      mask=mask,
      A=crystal.A,
      k_grid=k_grid,
      spin=spin,
      vol=crystal.vol
    )

    return pwd_args, (r_grid, g_grid)


@chex.dataclass
class EwaldArgs:
  ew_eta: Float
  ew_cut: Float
  ewald_grid: jax.Array = None


@chex.dataclass
class EnergyArgs:
  crystal: Crystal
  ng: jax.Array
  nr: jax.Array
  g_vec: jax.Array
  k_vec: jax.Array
  cg: jax.Array
  occ: jax.Array
  vol: Float
  ew_args: EwaldArgs


if __name__ == '__main__':
  pass
