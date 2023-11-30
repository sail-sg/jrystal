"""Wave function module
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

import jax_xc

from jrystal._src.wave_ops import coeff_expand, batched_fft
from jrystal._src.wave_ops import get_mask_cubic
from jrystal._src import occupation
from jrystal._src.utils import complex_norm_square
from jrystal._src import energy, potential, xc_density
from jrystal._src.grid import r_vectors, g_vectors
from jrystal._src.crystal import Crystal

from typing import Tuple, Dict, Callable
from jaxtyping import Int, Array, Float
from jrystal._src.jrystal_typing import CellVector, RealScalar
from jrystal._src.jrystal_typing import ComplexGrid, RealVecterGrid, RealGrid
from jrystal._src import errors
from jrystal._src.module import QRDecomp, BatchedBlochWave


class PlaneWaveDensity(nn.Module):
  num_electrons: Int
  cell_vectors: CellVector
  g_grid_sizes: Float[Array, 'd']
  k_vectors: Float[Array, '... 3']
  spin: Int
  occupation_method: str = 'gamma'
  xc_functional: str = 'lda_x'

  def setup(self):
    num_k = np.prod(np.array(self.k_vectors.shape)[:-1]).item()
    self.mask, num_g = get_mask_cubic(self.g_grid_sizes)
    shape = [2, num_k, num_g, self.num_electrons]
    self.dim = self.cell_vectors.shape[0]
    self.qr = QRDecomp(shape)
    self.bloch = BatchedBlochWave(self.cell_vectors, self.k_vectors)
    self.r_vector_grid = r_vectors(self.cell_vectors, self.g_grid_sizes)
    self.g_vector_grid = g_vectors(self.cell_vectors, self.g_grid_sizes)
    self.vol = jnp.linalg.det(self.cell_vectors)

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

    self.xc_density = getattr(jax_xc.functionals, self.xc_functional, None)
    if self.xc_density:
      self.xc_density = self.xc_density(polarized=True)
    else:
      raise NotImplementedError(
        f"{self.xc_functional} is not included in jax-xc. "
        "See https://jax-xc.readthedocs.io/en/latest/index.html#"
      )

  def __call__(self, crystal):
    return self.total_energy(crystal)

  def density(self, r=None, reduce=True) -> jax.Array:
    if r is None:
      r = self.r_vector_grid
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    wave = self.bloch(r, coeff_grid) / jnp.sqrt(self.vol)
    density = complex_norm_square(wave)

    if reduce:  # reduce over k, i by occupation number
      occ = self.occupation()
      if (wave.ndim < occ.ndim or wave.shape[:occ.ndim] != occ.shape):
        raise errors.WavevecOccupationMismatchError(wave.shape, occ.shape)
      occ = jnp.expand_dims(occ, range(occ.ndim, wave.ndim))
      density = jnp.sum(occ * density, axis=(1, 2))
    else:
      density = density

    return density

  def reciprocal_density(self, reduce=True) -> ComplexGrid:
    density = self.density(self.r_vector_grid, reduce=reduce)

    if reduce:  # reduce over k, i
      dim = density.ndim - 1
      density_fft = batched_fft(density, dim)
    else:
      dim = density.ndim - 3
      density_fft = batched_fft(density, dim)

    return density_fft

  def get_coefficient(self):
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    return coeff_grid

  def get_occupation(self):
    return self.occupation()

  def hartree(self) -> RealScalar:
    reciprocal_density_grid = self.reciprocal_density(reduce=True)
    return energy.hartree(reciprocal_density_grid, self.g_vector_grid, self.vol)

  def external(self, crystal: Crystal) -> RealScalar:
    reciprocal_density_grid = self.reciprocal_density(reduce=True)
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
    return energy.xc(
      self.density, self.r_vector_grid, self.vol, self.xc_functional
    )

  def total_energy(self, crystal: Crystal) -> RealScalar:
    eh = self.hartree()
    ee = self.external(crystal)
    ek = self.kinetic()
    exc = self.xc()
    energies = {'hartree': eh, 'external': ee, 'kinetic': ek, 'xc': exc}
    return eh + ee + ek + exc, energies


class PlaneWaveFermiDirac(nn.Module):
  num_electrons: Int
  cell_vectors: CellVector
  g_grid_sizes: Float[Array, 'd']
  k_vectors: Float[Array, '... 3']
  spin: Int
  smearing: float
  xc_functional: str = 'lda_x'

  def setup(self):
    num_k = self.k_vectors.shape[0]
    self.mask, num_g = get_mask_cubic(self.g_grid_sizes)
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
    self.xc_density = getattr(jax_xc.functionals, self.xc_functional, None)
    if self.xc_density:
      self.xc_density = self.xc_density(polarized=True)
    else:
      raise NotImplementedError(
        f"{self.xc_functional} is not included in jax-xc. "
        "See https://jax-xc.readthedocs.io/en/latest/index.html#"
      )
    self.r_vector_grid = r_vectors(self.cell_vectors, self.mask.shape)
    self.g_vector_grid = g_vectors(self.cell_vectors, self.mask.shape)
    self.vol = jnp.linalg.det(self.cell_vectors)

  def __call__(self, crystal):
    return self.total_energy(crystal)

  def density(self, r=None, reduce=True) -> jax.Array:
    if r is None:
      r = self.r_vector_grid

    coeff_grid = self.get_coefficient()
    wave = self.bloch(r, coeff_grid) / jnp.sqrt(self.vol)
    density = complex_norm_square(wave)

    if reduce:
      occ = self.occ_number.value
      if (wave.ndim < occ.ndim or wave.shape[:occ.ndim] != occ.shape):
        raise errors.WavevecOccupationMismatchError(wave.shape, occ.shape)
      occ = jnp.expand_dims(occ, range(occ.ndim, wave.ndim))
      density = jnp.sum(occ * density, axis=(1, 2))
    else:
      density = density
    return density.real  # [2, num_k, num_band]

  def reciprocal_density(
    self, r_vector_grid: RealVecterGrid, reduce=True
  ) -> ComplexGrid:
    density = self.density(r_vector_grid, reduce=reduce)
    dim = self.cell_vectors.shape[0]

    if reduce:  # reduce over k, i
      dim = density.ndim - 1
      density_fft = batched_fft(density, dim)
    else:
      dim = density.ndim - 3
      density_fft = batched_fft(density, dim)

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
      self.density, self.r_vector_grid, self.vol, self.xc_functional
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


class PlaneWaveBandStructure(nn.Module):
  hamitonian_density_fn: Callable
  num_bands: Int
  cell_vectors: CellVector
  g_grid_sizes: Float[Array, 'd']
  k_vectors: Float[Array, '... 3']
  xc_functional: str = 'lda_x'

  """

  Args:

    hamitonian_density_fn (Callable): the density function for constructing the
      effective potential.
    num_bands (Int): number of bands.

  """

  def setup(self):
    num_k = np.prod(np.array(self.k_vectors.shape)[:-1]).item()
    self.mask, num_g = get_mask_cubic(self.g_grid_sizes)
    shape = [1, num_k, num_g, self.num_bands]
    self.dim = self.cell_vectors.shape[0]
    self.qr = QRDecomp(shape)
    self.bloch = BatchedBlochWave(self.cell_vectors, self.k_vectors)
    self.r_vector_grid = r_vectors(self.cell_vectors, self.g_grid_sizes)
    self.g_vector_grid = g_vectors(self.cell_vectors, self.g_grid_sizes)
    self.vol = jnp.linalg.det(self.cell_vectors)

    self.xc_potential = potential.xc_lda

  def __call__(self, crystal):
    return self.energy_trace(crystal)

  def wave(self, r=None) -> ComplexGrid:
    if r is None:
      r = self.r_vector_grid
    coeff_grid = self.get_coefficient()
    wave = self.bloch(r, coeff_grid) / jnp.sqrt(self.vol)
    return wave

  def density(self, r=None, reduce: bool = False) -> RealGrid:
    wave = self.wave()
    density = complex_norm_square(wave)

    if reduce:
      density = jnp.sum(density, axis=range(3))
    return density

  def reciprocal_density(self, reduce: bool = False) -> ComplexGrid:
    density = self.density(reduce=False)
    dim = density.ndim - 3
    density_fft = batched_fft(density, dim)
    if reduce:
      density_fft = jnp.sum(density_fft, axis=range(3))
    return density_fft

  def get_coefficient(self):
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    return coeff_grid

  @property
  def hamitonian_density_grid(self):
    with jax.ensure_compile_time_eval():
      density = self.hamitonian_density_fn(self.r_vector_grid)
    return jnp.sum(density, axis=0)

  @property
  def reciprocal_hamitonian_density_grid(self):
    with jax.ensure_compile_time_eval():
      reciprocal_density = jnp.fft.fftn(self.hamitonian_density_grid)
    return reciprocal_density

  def hamitonian_hartree(self) -> ComplexGrid:
    return potential.hartree_reciprocal(
      self.reciprocal_hamitonian_density_grid, self.g_vector_grid
    )

  def hamitonian_xc(self) -> RealGrid:
    return potential.xc_lda(self.hamitonian_density_grid)

  def hamitonian_external(self, crystal: Crystal) -> RealGrid:
    return potential.externel_reciprocal(
      crystal.positions, crystal.charges, self.g_vector_grid, self.vol
    )

  def energy_trace(self, crystal: Crystal) -> RealScalar:

    density_grid = self.density(reduce=True)
    reciprocal_density_grid = self.reciprocal_density(reduce=True)

    hartree = energy.reciprocal_braket(
      self.hamitonian_hartree(), reciprocal_density_grid, self.vol
    )

    external = energy.reciprocal_braket(
      self.hamitonian_external(crystal), reciprocal_density_grid, self.vol
    )

    coeff_grid = self.get_coefficient()
    kinetic = energy.kinetic(
      self.g_vector_grid, self.k_vectors, coeff_grid, occupation=1
    )

    xc_energy = energy.real_braket(
      self.hamitonian_xc(), density_grid, self.vol
    )

    energies = {
      'hartree': hartree, 'external': external,
      'kinetic': kinetic, 'xc': xc_energy
    }

    total_energy = hartree + external + kinetic + xc_energy
    return total_energy, energies

  # def get_fork(self) -> Float[Array, 'd d']:
  #   reciprocal_density_band_grids = self.reciprocal_density_bands(reduce = )
  #   v_hartree = potential.hartree_reciprocal()
