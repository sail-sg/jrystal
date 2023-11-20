"""Wave function module
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

import jax_xc

from jrystal._src.functional import coeff_expand, batched_fft
from jrystal._src import occupation
from jrystal._src.utils import complex_norm_square
from jrystal._src import energy
from jrystal._src.grid import r_vectors, g_vectors
from jrystal._src.crystal import Crystal

from typing import Union, List
from jaxtyping import Int, Array
from jrystal._src.jrystal_typing import MaskGrid, CellVector
from jrystal._src.jrystal_typing import ComplexVecterGrid, RealVecterGrid
from jrystal._src import errors
from jrystal._src.module import QRDecomp, BatchedBlochWave


class PlaneWaveDensity(nn.Module):
  shape: Union[Int[Array, "..."], List]  # [nspin, nk, ng, ni]
  mask: MaskGrid
  cell_vectors: CellVector
  k_vector_grid: RealVecterGrid
  spin: Int
  vol: float
  occupation_method: str = 'gamma'
  xc_method: str = 'lda_x'

  def setup(self):
    _, nk, _, ni = self.shape
    self.qr = QRDecomp(self.shape)
    self.bloch = BatchedBlochWave(self.cell_vectors, self.k_vector_grid)

    self.occupation = getattr(
      occupation, self.occupation_method.capitalize(), None
    )

    if self.occupation:
      self.occupation = self.occupation(nk, ni, self.spin)
    else:
      raise NotImplementedError(
        f"{self.occupation_method} is not included in jax-xc. "
        "Occupation method must be \{gamma, orthogonal, uniform\}"
      )

    self.xc_density = getattr(jax_xc.functionals, self.xc_method, None)
    if self.xc_density:
      self.xc_density = self.xc_density(polarized=True)
    else:
      raise NotImplementedError(
        f"{self.xc_method} is not included in jax-xc. "
        "See https://jax-xc.readthedocs.io/en/latest/index.html#"
      )

  def __call__(self, r) -> jax.Array:
    return self.density(r, reduce=True)

  def density(self, r, reduce=True) -> jax.Array:
    occ = self.occupation()
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    wave = self.bloch(r, coeff_grid)

    if (wave.ndim < occ.ndim or wave.shape[:occ.ndim] != occ.shape):
      raise errors.WavevecOccupationMismatchError(wave.shape, occ.shape)

    density = complex_norm_square(wave)
    occ = jnp.expand_dims(occ, range(occ.ndim, wave.ndim))

    if reduce:  # reduce over k, i
      density = jnp.sum(occ * density, axis=(1, 2)) / self.vol
    else:
      density = occ * density / self.vol

    return density

  def reciprocal_density(
    self, r_vector_grid: RealVecterGrid, reduce=True
  ) -> ComplexVecterGrid:
    density = self.density(r_vector_grid, reduce=reduce)
    num_grids = jnp.prod(jnp.array(self.mask.shape))
    dim = self.cell_vectors.shape[0]

    if reduce:  # reduce over k, i
      dim = density.ndim - 1
      density_fft = batched_fft(density, dim) / num_grids * self.vol
    else:
      dim = density.ndim - 3
      density_fft = batched_fft(density, dim) / num_grids * self.vol

    return density_fft

  def get_coefficient(self):
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    return coeff_grid

  def get_occupation(self):
    return self.occupation()

  def hartree(self):
    r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)
    reciprocal_density_grid = self.reciprocal_density(
      r_vector_grid, reduce=True
    )
    return energy.hartree(reciprocal_density_grid, g_vector_grid, self.vol)

  def external(self, crystal: Crystal):
    r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)
    reciprocal_density_grid = self.reciprocal_density(
      r_vector_grid, reduce=True
    )
    positions = crystal.positions
    charges = crystal.charges
    return energy.external(
      reciprocal_density_grid, positions, charges, g_vector_grid, self.vol
    )

  def kinetic(self):
    g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)
    coeff_grid = self.get_coefficient()
    occupation = self.get_occupation()
    return energy.kinetic(
      g_vector_grid, self.k_vector_grid, coeff_grid, occupation
    )

  def xc(self):
    r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    return energy.xc(self.density, r_vector_grid, self.vol, self.xc_method)
