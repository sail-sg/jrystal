"""Wave function modules.

The wave function module is the core to our differentiable computation
framework. It is responsible for defining the wave function ansatz and the
corresponding integrals.

NOTE: It is advisable to keep the wave function module separate from the
crystal objects, which define the external potential of the system.
NOTE: Integrals should be defined in association with the wave function,
ideally as methods within the module.
NOTE: Wherever feasible, functional programming approaches should be employed.

"""

import jax
from jax import lax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import einops
from typing import List, Union
import jax_xc

from .wave_ops import coeff_expand
from .wave_ops import get_mask_cubic
from . import occupation
from .utils import complex_norm_square
from . import energy, potential, xc_density
from .grid import (
  r_vectors,
  g_vectors,
  half_frequency_shape,
  half_frequency_pad_to,
)
from .crystal import Crystal
from .bloch import bloch_wave

from typing import Tuple, Dict, Callable
from jaxtyping import Int, Array, Float, Complex
from .jrystal_typing import CellVector, RealScalar
from .jrystal_typing import ComplexGrid, RealVecterGrid, RealGrid
from . import errors
from .module import QRDecomp


def normalize(raw_occ, num_electrons):
  raw_occ += (num_electrons -
              jnp.sum(raw_occ.flatten())) / np.prod(raw_occ.shape)
  return raw_occ


def clip(raw_occ, num_k):
  return jnp.clip(raw_occ, a_min=0, a_max=1 / num_k)


def constrain_occ(raw_occ, num_electrons, num_k):
  cond_fn = lambda x: jnp.all(jnp.logical_and(x >= 0, x <= 1 / num_k))
  body_fn = lambda x: normalize(
    clip(x, num_k), num_electrons
  )  # iteratively project
  return lax.while_loop(cond_fn, body_fn, raw_occ)


class ElectronWave(nn.Module):
  num_electrons: Int
  g_grid_sizes: List
  k_grid_sizes: List
  spin: Union[Int, None] = None

  def setup(self):
    self.half_shape = half_frequency_shape(self.g_grid_sizes)
    mask_size = np.prod(self.half_shape)
    num_k = np.prod(self.k_grid_sizes).item()
    param_shape = (2, num_k, mask_size, self.num_electrons)
    occ_shape = (2, num_k, self.num_electrons)
    self.coeff_real = self.param(
      "coeff_real", nn.initializers.normal(), param_shape
    )
    self.coeff_imag = self.param(
      "coeff_imag", nn.initializers.normal(), param_shape
    )
    self.occ = self.param("occ", nn.initializers.normal(), occ_shape)
    self.constrain_occupation()

  def k_vectors(self, cell_vectors):
    k_vector_grid = r_vectors(jnp.linalg.inv(cell_vectors), self.k_grid_sizes)
    return k_vector_grid.reshape((-1, 3))

  def coefficient(self):
    # constrain the coefficients
    num_spins = 2
    num_k = np.prod(np.array(self.k_grid_sizes)).item()
    raw_coeff = self.coeff_real + 1j * self.coeff_imag
    raw_coeff = jnp.swapaxes(
      jnp.linalg.qr(raw_coeff, mode="reduced")[0], -1, -2
    )
    raw_coeff = jnp.reshape(
      raw_coeff,
      (num_spins, num_k, self.num_electrons, *self.half_shape),
    )
    coeff = half_frequency_pad_to(raw_coeff, self.g_grid_sizes)
    return coeff

  def density(self, r, cell_vectors):
    coeff = self.coefficient()
    vol = jnp.linalg.det(cell_vectors)
    psi = bloch_wave(cell_vectors, coeff, k_vec=None)
    psir = psi(r, force_fft=True)
    dens = jnp.real(jnp.conj(psir) * psir) / vol  # (2, nk, ni, *r.shape[:-1])
    occ = self.occ
    return jnp.einsum("ski...,ski->s...", dens, occ)

  def constrain_occupation(self):
    occ = self.occ
    num_k = np.prod(np.array(self.k_grid_sizes)).item()
    occ = constrain_occ(occ, self.num_electrons, num_k)
    self.occ.value = occ

  def kinetic_energy(self, cell_vectors):
    g_vector_grid = g_vectors(cell_vectors, self.g_grid_sizes)
    k_vec = self.k_vectors(cell_vectors)
    print(k_vec)
    occ = self.get_variable("params", "occ")
    coeff = self.coefficient()
    return energy.kinetic(g_vector_grid, k_vec, coeff, occ)

  def __call__(self, r, cell_vectors):
    coeff = self.coefficient()
    # compute the wave function
    vol = jnp.linalg.det(cell_vectors)
    k_vec = self.k_vectors(cell_vectors)
    psi = bloch_wave(cell_vectors, coeff, k_vec)
    return psi(r, force_fft=True) / jnp.sqrt(vol)


class PlaneWaveDensity(nn.Module):
  num_electrons: Int
  cell_vectors: CellVector
  g_grid_sizes: Float[Array, 'd']
  k_vectors: Float[Array, '... 3']
  spin: Int
  occupation_method: str = 'gamma'
  xc_functional: str = 'lda_x'
  mask: Float[Array, '*d'] = None,
  num_g: int = None
  """
  The wave function module for total energy minimization.

  Attributes:
      num_electrons (Int): the number of electrons(orbitals).
      cell_vectors (Array): the cell vector.
      g_grid_sizes (Array): grid lattice for FFT.
      k_vectors (Array): the samples within the Brillouin zone.
      spin (Int): the number of unpaired electrons.
      occupation_method (str): occupation method.
      xc_functional (str): the xc functional.
        See https://jax-xc.readthedocs.io/en/latest/index.html#

  """

  def setup(self):
    num_k = np.prod(np.array(self.k_vectors.shape)[:-1]).item()
    if self.mask is None:
      self.mask, self.num_g = get_mask_cubic(self.g_grid_sizes)
    shape = [2, num_k, self.num_g, self.num_electrons]
    self.dim = self.cell_vectors.shape[0]
    self.qr = QRDecomp(shape)
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

  def bloch(self, coeff, r):
    return bloch_wave(self.cell_vectors, coeff)(r)

  def density(self, r=None, reduce=True) -> jax.Array:
    if r is None:
      r = self.r_vector_grid
    coeff_grid = self.get_coefficient()
    wave = self.bloch(coeff_grid, r) / jnp.sqrt(self.vol)
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
    return jnp.fft.fftn(density, axes=range(-self.dim, 0))

  def get_coefficient(self):
    coeff_dense = self.qr()
    coeff_dense = jnp.swapaxes(coeff_dense, -1, -2)
    coeff_grid = coeff_expand(coeff_dense, self.mask)
    return coeff_grid

  def get_occupation(self):
    return self.occupation()

  def hartree(self) -> Float[Array, '']:
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
  """
    The wave function module for total energy minimization with
    fermi dirac smearing.

  NOTE: this module is NOT finished. The eigenvalues are NOT the actual ones.
  Need to fix in the Future version.

  Attributes:
      num_electrons (Int): the number of electrons(orbitals).
      cell_vectors (Array): the cell vector.
      g_grid_sizes (Array): grid lattice for FFT.
      k_vectors (Array): the samples within the Brillouin zone.
      spin (Int): the number of unpaired electrons.
      smearing (float): the smearing constant.
      xc_functional (str): the xc functional.
        See https://jax-xc.readthedocs.io/en/latest/index.html#

  """

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

  def bloch(self, coeff, r):
    return bloch_wave(self.cell_vectors, coeff)(r)

  def density(self, r=None, reduce=True) -> jax.Array:
    if r is None:
      r = self.r_vector_grid
    coeff_grid = self.get_coefficient()
    wave = self.bloch(coeff_grid, r) / jnp.sqrt(self.vol)
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
    ndim = r_vector_grid.shape[-1]
    density = self.density(r_vector_grid, reduce=reduce)
    return jnp.fft.fftn(density, axes=range(-ndim, 0))

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

    # NOTE: this is wrong, need to fix.
    eigenvalues = eh + ee + ek + exc

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
  cell_vectors: Float[Array, 'd d']
  g_grid_sizes: Float[Array, 'd']
  k_vectors: Float[Array, '... 3']
  xc_functional: str = 'lda_x'
  mask: Float[Array, '*d'] = None,
  num_g: int = None
  """
  The wave function module for band structure calculation.

  NOTE: this module DOES NOT consider polarized circumstances.

  Attributes:
      hamitonian_density_fn (Callable): the converged density function on real
        space. It can be any function that take inputs with shape (..., 3) and
        return the shape (...). If flax.nn.module is used, it can be something
        like `module.apply({'params': params}, r)`
      num_bands (Int): the number of bands.
      g_grid_sizes (Array):
      k_vectors (Array): the path of special points sampled within the
        Brillouin zone.
      xc_functional: the xc functional

  TODO: Now it only supports lda for xc_functional. Next version will support
  all the xc functional provided in Jax-xc.

  """

  def setup(self):
    num_k = np.prod(np.array(self.k_vectors.shape)[:-1]).item()
    if self.mask is None:
      self.mask, self.num_g = get_mask_cubic(self.g_grid_sizes)
    shape = [1, num_k, self.num_g, self.num_bands]
    self.dim = self.cell_vectors.shape[0]
    self.qr = QRDecomp(shape)
    self.r_vector_grid = r_vectors(self.cell_vectors, self.g_grid_sizes)
    self.g_vector_grid = g_vectors(self.cell_vectors, self.g_grid_sizes)
    self.vol = jnp.linalg.det(self.cell_vectors)

    self.xc_potential = potential.xc_lda

  def bloch(self, coeff, r):
    return bloch_wave(self.cell_vectors, coeff)(r)

  def __call__(self, crystal: Crystal):
    return self.energy_trace(crystal)

  def wave(self, r: Float[Array, "... 3"] = None) -> Complex[Array, "... 3"]:
    """the bloch wave function.

    Args:
        r (Array, optional): The real space coordinate. If None, a grid mesh of
          r asscoiated with the g grid is used, where fft is performed for the
          computation.

    Returns:
        Array: the value of the bloch wave function evaluated at r.

    """
    if r is None:
      r = self.r_vector_grid
    coeff_grid = self.get_coefficient()
    wave = self.bloch(coeff_grid, r) / jnp.sqrt(self.vol)
    return wave

  def density(self, r=None, reduce: bool = False) -> RealGrid:
    wave = self.wave()
    density = complex_norm_square(wave)

    if reduce:
      density = jnp.sum(density, axis=range(3))
    return density

  def reciprocal_density(self, reduce: bool = False) -> ComplexGrid:
    density = self.density(reduce=reduce)
    return jnp.fft.fftn(density, axes=range(-self.dim, 0))

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

  def hamitonian_kinetic(
    self,
    k_vectors: Float[Array, "num_k 3"] = None,
  ) -> Float[Array, 'num_k n1 n2 n3']:
    if k_vectors is None:
      k_vectors = self.k_vectors
    k_vectors = einops.rearrange(k_vectors, "nk d -> nk 1 1 1 d")
    return jnp.sum((self.g_vector_grid + k_vectors)**2, axis=-1) / 2

  def energy_trace(self, crystal: Crystal, k_vectors=None) -> RealScalar:
    if k_vectors is None:
      k_vectors = self.k_vectors

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
      self.g_vector_grid, k_vectors, coeff_grid, occupation=1
    )

    xc_energy = energy.real_braket(self.hamitonian_xc(), density_grid, self.vol)

    energies = {
      'hartree': hartree,
      'external': external,
      'kinetic': kinetic,
      'xc': xc_energy
    }

    total_energy = hartree + external + kinetic + xc_energy
    return total_energy, energies

  def fork_matrix(self, crystal, k_vectors=None) -> Float[Array, 'd d']:
    if k_vectors is None:
      k_vectors = self.k_vectors

    v_hartree = self.hamitonian_hartree()  # [n1 n2 n3]
    wg = self.get_coefficient()  # [1 nk ni n1 n2 n3]
    wr = self.wave()

    num_grids = np.prod(np.array(v_hartree.shape))
    integral_factor = self.vol / num_grids

    fock_hartree = einops.einsum(
      jnp.conj(wr),
      jnp.fft.ifftn(v_hartree),
      wr,
      "a nk ni1 n1 n2 n3, n1 n2 n3, a nk ni2 n1 n2 n3 -> nk ni1 ni2"
    ) * integral_factor

    v_external = self.hamitonian_external(crystal)
    fock_external = einops.einsum(
      jnp.conj(wr),
      jnp.fft.ifftn(v_external),
      wr,
      "a nk ni1 n1 n2 n3, n1 n2 n3, a nk ni2 n1 n2 n3 -> nk ni1 ni2"
    ) * integral_factor

    v_lda = self.hamitonian_xc()
    fock_xc = einops.einsum(
      jnp.conj(wr),
      v_lda,
      wr,
      "a nk ni1 n1 n2 n3, n1 n2 n3, a nk ni2 n1 n2 n3 -> nk ni1 ni2"
    ) * integral_factor

    v_kin = self.hamitonian_kinetic(k_vectors)
    fock_kinetic = einops.einsum(
      jnp.conj(wg),
      v_kin,
      wg,
      "a nk ni1 n1 n2 n3, nk n1 n2 n3, a nk ni2 n1 n2 n3 -> nk ni1 ni2"
    )

    fock = fock_hartree + fock_external + fock_xc + fock_kinetic
    return fock

  def eigenvalues(self, crystal: Crystal) -> Float[Array, 'num_k num_bands']:
    fock = self.fork_matrix(crystal)
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(fock)
    return eigenvalues
