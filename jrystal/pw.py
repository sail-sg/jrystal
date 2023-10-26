"""The Plane-wave class.

This module defines the plane-wave functions, which is constucted by Fourier
bases. The module is built on top of the Haiku library.

"""

import jax.numpy as jnp
import numpy as np
from jrystal import Crystal

from jrystal._src import g_vectors, r_vectors
from jrystal._src import _get_mask_radius

from jaxtyping import Float, Array, Int
from typing import List
from ase.dft.kpoints import monkhorst_pack
from jrystal._src.grid import _grid_sizes
from jrystal._src import PlaneWave as PW


class PlaneWave():
  """Plane wave module.
  """

  def __init__(
    self,
    crystal: Crystal,
    Ecut: Float,
    g_grid_sizes: Int | List | Float[Array, 'nd'],  # noqa: F821
    k_grid_sizes,
    occ: str = 'simple',
    polarize: bool = True
  ):
    # TODO: get g_grid_sizes according to Ecut.
    self.crystal = crystal
    self.Ecut = Ecut
    self.polarize = polarize
    self.g_grid_sizes = _grid_sizes(g_grid_sizes)
    self.k_grid_sizes = _grid_sizes(k_grid_sizes)
    self.occ = occ
    self.nspin = 2 if self.polarize else 1
    self.ni = self.crystal.nelec
    self.nk = np.prod(self.k_grid_sizes)

    self.g_mask = _get_mask_radius(
      self.crystal.A,
      self.g_grid_sizes,
      self.Ecut
      )

    self.ng = jnp.sum(self.g_mask)
    self.k_grid = monkhorst_pack(self.k_grid_sizes)
    self.g_vec = g_vectors(self.crystal.A, self.g_grid_sizes)
    self.r_vec = r_vectors(self.crystal.A, self.g_grid_sizes)
    self.cg_shape = [self.nspin, self.nk, self.ng, self.ni]

    self.pw = PW(self.cg_shape, self.g_mask, self.crystal.A, self.k_grid)

  def init(self, key):
    return self.pw.init(key, self.r_vec)

  def get_wave(self, params):
    return self.pw.apply(params, self.r_vec)
