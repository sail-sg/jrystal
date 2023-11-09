"""Dataclasses of arguments."""

import jax
import chex
import numpy as np

from jrystal._src.crystal import Crystal
from jrystal._src.grid import grid_sizes, g_vectors, r_vectors
from jrystal._src.functional import get_mask_radius

from ase.dft.kpoints import monkhorst_pack

from typing import Union, List
from jaxtyping import Int, Float, Array
from jrystal._src.jrystal_typing import MaskGrid, CellVector
from jrystal._src.jrystal_typing import RealVecterGrid


@chex.dataclass
class PWDArgs:
  shape: Union[Int[Array, "..."], List]  # [nspin, nk, ng, ni]
  mask: MaskGrid
  cell_vectors: CellVector
  k_vector_grid: RealVecterGrid
  spin: Int
  vol: float
  occupation_method: str = 'gamma'

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
    g_grid_sizes = grid_sizes(g_grid_sizes)
    k_grid_sizes = grid_sizes(k_grid_sizes)
    occ = occ
    nspin = 2 if polarize else 1
    ni = crystal.num_electrons
    nk = np.prod(k_grid_sizes)
    spin = int(ni % 2)
    mask = get_mask_radius(crystal.A, g_grid_sizes, Ecut)

    ng = np.sum(mask)
    k_grid = monkhorst_pack(k_grid_sizes)
    g_grid = g_vectors(crystal.A, g_grid_sizes)
    r_grid = r_vectors(crystal.A, g_grid_sizes)
    cg_shape = [nspin, nk, ng, ni.item()]

    pwd_args = PWDArgs(
      shape=cg_shape,
      mask=mask,
      cell_vectors=crystal.A,
      k_vector_grid=k_grid,
      spin=spin,
      vol=crystal.vol
    )

    return pwd_args, (r_grid, g_grid)


@chex.dataclass
class EwaldArgs:
  ewald_eta: Float
  ewald_cut: Float


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
