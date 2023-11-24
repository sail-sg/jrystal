"""Wave function module
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import jax_xc

from jrystal._src.functional import coeff_expand, batched_fft
from jrystal._src import occupation
from jrystal._src.utils import complex_norm_square
from jrystal._src import energy, potential, xc_density
from jrystal._src.grid import r_vectors, g_vectors
from jrystal._src.crystal import Crystal

from typing import Tuple, Dict
from jaxtyping import Int, Array, Float
from jrystal._src.jrystal_typing import MaskGrid, CellVector, RealScalar
from jrystal._src.jrystal_typing import ComplexVecterGrid, RealVecterGrid
from jrystal._src import errors
from jrystal._src.module import QRDecomp, BatchedBlochWave


class PlaneWaveDensity(nn.Module):
  num_electrons: Int
  mask: MaskGrid
  cell_vectors: CellVector
  k_vectors: Float[Array, '... 3']
  spin: Int
  vol: float
  occupation_method: str = 'gamma'
  xc_method: str = 'lda_x'

  def setup(self):
    num_k = np.prod(np.array(self.k_vectors.shape)[:-1]).item()
    num_g = np.sum(np.array(self.mask)).item()
    shape = [2, num_k, num_g, self.num_electrons]
    self.dim = self.cell_vectors.shape[0]
    self.qr = QRDecomp(shape)
    self.bloch = BatchedBlochWave(self.cell_vectors, self.k_vectors)
    self.r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    self.g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)

    self.occupation = getattr(
      occupation, self.occupation_method.capitalize(), None
    )

    if self.occupation:
      self.occupation = self.occupation(num_k, self.num_electrons, self.spin)
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

  def __call__(self, crystal):
    return self.total_energy(crystal)

  def density(self, r, reduce=True) -> jax.Array:
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    wave = self.bloch(r, coeff_grid)

    density = complex_norm_square(wave)

    if reduce:  # reduce over k, i by occupation number
      occ = self.occupation()
      if (wave.ndim < occ.ndim or wave.shape[:occ.ndim] != occ.shape):
        raise errors.WavevecOccupationMismatchError(wave.shape, occ.shape)
      occ = jnp.expand_dims(occ, range(occ.ndim, wave.ndim))
      density = jnp.sum(occ * density, axis=(1, 2)) / self.vol
    else:
      density = density / self.vol

    return density

  def reciprocal_density(
    self, r_vector_grid: RealVecterGrid, reduce=True
  ) -> ComplexVecterGrid:
    density = self.density(r_vector_grid, reduce=reduce)
    num_grids = jnp.prod(jnp.array(self.mask.shape))
    
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

  def hartree(self) -> RealScalar:
    reciprocal_density_grid = self.reciprocal_density(
      self.r_vector_grid, reduce=True
    )
    return energy.hartree(reciprocal_density_grid, self.g_vector_grid, self.vol)

  def external(self, crystal: Crystal) -> RealScalar:
    reciprocal_density_grid = self.reciprocal_density(
      self.r_vector_grid, reduce=True
    )
    positions = crystal.positions
    charges = crystal.charges
    return energy.external(
      reciprocal_density_grid, positions, charges, self.g_vector_grid, self.vol
    )

  def kinetic(self) -> RealScalar:
    coeff_grid = self.get_coefficient()
    occupation = self.get_occupation()
    return energy.kinetic(
      self.g_vector_grid, self.k_vectors, coeff_grid, occupation
    )

  def xc(self) -> RealScalar:
    return energy.xc(self.density, self.r_vector_grid, self.vol, self.xc_method)

  def total_energy(self, crystal: Crystal) -> RealScalar:
    eh = self.hartree()
    ee = self.external(crystal)
    ek = self.kinetic()
    exc = self.xc()
    energies = {'hartree': eh, 'external': ee, 'kinetic': ek, 'xc': exc}
    return eh + ee + ek + exc, energies


class PlaneWaveFermiDirac(nn.Module):
  num_electrons: Int
  mask: MaskGrid
  cell_vectors: CellVector
  k_vectors: Float[Array, '... 3']
  spin: Int
  vol: float
  smearing: float
  xc_method: str = 'lda_x'

  def setup(self):
    num_k = self.k_vectors.shape[0]
    num_g = np.sum(np.array(self.mask)).item()
    shape = [2, num_k, num_g, self.num_electrons]
    occ_number = occupation.occupation_gamma(
      num_k, self.num_electrons, self.spin
    )
    self.occ_number = self.variable(
      "variable", "occupation", lambda x: x, occ_number
    )
    self.dim = self.cell_vectors.shape[-1]
    self.qr = QRDecomp(shape)
    self.bloch = BatchedBlochWave(self.cell_vectors, self.k_vectors)
    self.occupation_fn = occupation.FermiDirac(
      self.num_electrons, width=self.smearing
    )
    self.xc_density = getattr(jax_xc.functionals, self.xc_method, None)
    if self.xc_density:
      self.xc_density = self.xc_density(polarized=True)
    else:
      raise NotImplementedError(
        f"{self.xc_method} is not included in jax-xc. "
        "See https://jax-xc.readthedocs.io/en/latest/index.html#"
      )
    self.r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    self.g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)

  def __call__(self, crystal):
    return self.total_energy(crystal)

  def density(self, r, reduce=True) -> jax.Array:
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    wave = self.bloch(r, coeff_grid)
    density = complex_norm_square(wave)

    if reduce:
      occ = self.occ_number.value
      if (wave.ndim < occ.ndim or wave.shape[:occ.ndim] != occ.shape):
        raise errors.WavevecOccupationMismatchError(wave.shape, occ.shape)
      occ = jnp.expand_dims(occ, range(occ.ndim, wave.ndim))
      density = jnp.sum(occ * density, axis=(1, 2)) / self.vol
    else:
      density = density / self.vol
    return density.real  # [2, num_k, num_band]

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

    return density_fft  # [2, num_k, num_band]

  def get_coefficient(self):
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    return coeff_grid

  def get_occupation(
    self, eigenvalues: Float[Array, 'num_spin num_k num_bands']
  ) -> Float[Array, 'num_spin num_k num_bands']:
    return self.occupation_fn(eigenvalues)

  def hartree(self) -> Float[Array, 'num_spin num_k num_bands']:
    reciprocal_density_reduced = self.reciprocal_density(
      self.r_vector_grid, reduce=True
    )
    reciprocal_density_all_bands = self.reciprocal_density(
      self.g_vector_grid, reduce=False
    )
    v_hartree = potential.hartree_reciprocal(
      reciprocal_density_reduced, self.g_vector_grid, self.vol
    )
    hartree = jnp.sum(
      v_hartree * reciprocal_density_all_bands, 
      axis=range(-1, -self.dim - 1, -1)
    )
    hartree /= 2

    return hartree.real

  def external(self,
               crystal: Crystal) -> Float[Array, 'num_spin num_k num_bands']:
    reciprocal_density_all_bands = self.reciprocal_density(
      self.r_vector_grid, reduce=False
    )
    v_external = potential.externel_reciprocal(
      crystal.positions, crystal.charges, self.g_vector_grid, self.vol
    )
    external = jnp.sum(
      v_external * reciprocal_density_all_bands, 
      axis=range(-1, -self.dim - 1, -1)
    )

    return external.real

  def kinetic(self) -> Float[Array, 'num_spin num_k num_bands']:
    coeff_grid = self.get_coefficient()
    return energy.kinetic(self.g_vector_grid, self.k_vectors, coeff_grid)

  def xc(self) -> Float[Array, 'num_spin num_k num_bands']:
    epsilon_xc = xc_density.xc_density(
      self.density, self.r_vector_grid, self.vol, self.xc_method
    )
    density_all_bands = self.density(self.r_vector_grid, reduce=False)
    return jnp.sum(
      density_all_bands * epsilon_xc, axis=range(-1, -self.dim - 1, -1)
    )

  def total_energy(self, crystal: Crystal) -> Tuple[float, Dict]:
    eh = self.hartree()
    ee = self.external(crystal)
    ek = self.kinetic()
    exc = self.xc()

    eigenvalues = eh + ee + ek + exc  # TODO: this is wrong, need to fix
    occupation = self.occupation_fn(eigenvalues)
    self.occ_number.value = occupation

    eh = jnp.sum(occupation * eh)
    ee = jnp.sum(occupation * ee)
    ek = jnp.sum(occupation * ek)
    exc = jnp.sum(occupation * exc)

    energies = {'hartree': eh, 'external': ee, 'kinetic': ek, 'xc': exc}

    return eh + ee + ek + exc, energies
