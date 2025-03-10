"""Energy functions. """
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from typing import Tuple
from jaxtyping import Array, Complex, Float, Int

from . import braket, potential, pw
from .ewald import ewald_coulomb_repulsion
from .grid import translation_vectors
from .utils import (
  absolute_square, safe_real, wave_to_density, wave_to_density_reciprocal
)


def hartree(
  density_grid_reciprocal: Complex[Array, 'x y z'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  kohn_sham: bool = False
) -> Float:
  r"""Calculate the Hartree energy in reciprocal space.

  The Hartree energy represents the classical electrostatic interaction between
  electrons. The calculation is performed in reciprocal space for efficiency.
  The Hartree potential in reciprocal space is given by:

  .. math::

      \hat{V}_H(\mathbf{G}) = 4\pi \frac{\hat{\rho}(\mathbf{G})}{|\mathbf{G}|^2},
      \quad \hat{V}_H(\mathbf{0}) = 0

  The Hartree energy is computed as:

  .. math::
  
      E_H = 2\pi \sum_{\mathbf{G}} \frac{|\hat{\rho}(\mathbf{G})|^2}{|\mathbf{G}|^2}

  Args:
    density_grid_reciprocal (Complex[Array, 'x y z']): Electron density in 
      reciprocal space.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    vol (Float): Unit cell volume.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.

  Returns:
    Float: Hartree energy.
  """
  dim = g_vector_grid.shape[-1]

  if density_grid_reciprocal.ndim == dim + 1:
    density_grid_reciprocal = jnp.sum(density_grid_reciprocal, axis=0)

  v_hartree_reciprocal = potential.hartree_reciprocal(
    density_grid_reciprocal, g_vector_grid, kohn_sham
  )

  hartree_energy = braket.reciprocal_braket(
    v_hartree_reciprocal, density_grid_reciprocal, vol
  )

  return safe_real(hartree_energy)


def external(
  density_grid_reciprocal: Complex[Array, 'x y z'],
  position: Float[Array, 'atom 3'],
  charge: Float[Array, 'atom'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float
) -> Float:
  r"""Calculate the external potential energy in reciprocal space.

  The external potential energy is computed using Parseval's identity,
  expressing the real-space integral as a sum over reciprocal lattice vectors:

  .. math::
      E = \sum_{\mathbf{G}} \hat{\rho}(\mathbf{G}) \sum_{\alpha} Z_{\alpha} 
      \exp(-i\mathbf{G}\cdot\mathbf{R}_{\alpha}) v(\mathbf{G})

  where:

  - :math:`\hat{\rho}(\mathbf{G})` is the Fourier transform of the electron density, i.e. the density in reciprocal space.
  - :math:`Z_{\alpha}` is the nuclear charge of atom :math:`\alpha`
  - :math:`\mathbf{R}_{\alpha}` is the position of atom :math:`\alpha`
  - :math:`v(\mathbf{G})` is the Fourier transform of the Coulomb potential

  Args:
    density_grid_reciprocal (Complex[Array, 'x y z']): Electron density in 
      reciprocal space.
    position (Float[Array, 'atom 3']): Atomic positions in the unit cell.
    charge (Float[Array, 'atom']): Nuclear charges.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    vol (Float): Unit cell volume.

  Returns:
    Float: External potential energy.
  """
  dim = g_vector_grid.shape[-1]
  if density_grid_reciprocal.ndim == dim + 1:
    density_grid_reciprocal = jnp.sum(density_grid_reciprocal, axis=0)

  v_external_reciprocal = potential.external_reciprocal(
    position, charge, g_vector_grid, vol
  )
  external_energy = braket.reciprocal_braket(
    v_external_reciprocal, density_grid_reciprocal, vol
  )

  return safe_real(external_energy)


def kinetic(
  g_vector_grid: Float[Array, 'x y z 3'],
  k_vector_grid: Float[Array, 'x y z 3'],
  coeff_grid: Complex[Array, 'spin kpt band x y z'],
  occupation: Optional[Float[Array, 'spin kpt band']] = None
) -> Union[Float, Float[Array, "spin kpt band"]]:
  r"""Calculate the kinetic energy of plane wave states.

  For plane wave basis functions, the kinetic energy operator is diagonal in
  reciprocal space. The kinetic energy is computed as:

  .. math::
      E_{\text{kin}} = \frac{1}{2} \sum_{\mathbf{G}} 
      |\mathbf{k} + \mathbf{G}|^2 |c_{n\mathbf{k}}(\mathbf{G})|^2

  where:
  
  - :math:`\mathbf{k}` is the k-point vector
  - :math:`\mathbf{G}` is the reciprocal lattice vector
  - :math:`c_{n\mathbf{k}}(\mathbf{G})` are the plane wave coefficients
  - :math:`n` is the band index

  Args:
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    k_vector_grid (Float[Array, 'x y z 3']): k-points in reciprocal space.
    coeff_grid (Complex[Array, 'spin kpt band x y z']): Plane wave coefficients.
    occupation (Float[Array, 'spin kpt band'], optional): Occupation numbers.
      If provided, returns the total kinetic energy weighted by occupations.

  Returns:

    Union[Float, Float[Array, "spin kpt band"]]: If occupation is provided, returns the total kinetic energy. Otherwise, returns the kinetic energy for each state.
  """

  dim = g_vector_grid.shape[-1]

  _g = jnp.expand_dims(g_vector_grid, axis=range(3))
  _k = jnp.expand_dims(
    k_vector_grid, axis=[0] + [i + 2 for i in range(dim + 1)]
  )
  e_kin = jnp.sum((_g + _k)**2, axis=-1)  # [1, nk, ni, x y z]
  e_kin = jnp.sum(e_kin * absolute_square(coeff_grid), axis=range(3, dim + 3))

  if occupation is not None:
    e_kin = jnp.sum(e_kin * occupation) / 2
  else:
    e_kin /= 2

  return safe_real(e_kin)


def xc_lda(
  density_grid: Float[Array, 'x y z'],
  vol: Float,
  kohn_sham: bool = False
) -> Float:
  r"""Calculate the LDA exchange-correlation energy.

  Implements the Local Density Approximation (LDA) for the exchange-correlation
  energy in the spin-unpolarized case. The exchange energy in LDA is given by:

  .. math::
      E_x^{\text{LDA}}[\rho] = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3}
      \int \rho(\mathbf{r})^{4/3} d\mathbf{r}

  Note that this implementation includes both exchange and correlation terms,
  though only the exchange term is shown in the equation above.

  Args:
    density_grid (Float[Array, 'x y z']): Real-space electron density.
    vol (Float): Unit cell volume.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.

  Returns:
    Float: LDA exchange-correlation energy.
  """

  assert density_grid.ndim in [3, 4]

  if density_grid.ndim == 4:  # have spin channel
    density_grid = jnp.sum(density_grid, axis=0)

  num_grid = jnp.prod(jnp.array(density_grid.shape))
  lda_density = potential.xc_lda(density_grid, kohn_sham)
  e_lda = jnp.sum(lda_density * density_grid)
  e_lda = safe_real(e_lda)

  return e_lda * vol / num_grid


def nuclear_repulsion(
  position: Float[Array, 'atom 3'],
  charge: Float[Array, 'atom'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: Float[Array, 'x y z 3'],
  vol: Float,
  ewald_eta: float,
  ewald_cutoff: float,
) -> Float:
  """Compute the nuclear repulsion energy using Ewald summation.

  This function calculates the nuclear-nuclear repulsion energy in periodic systems
  using the Ewald summation technique.

  Args:
    position (Float[Array, 'atom 3']): Coordinates of atoms in a unit cell.
    charge (Float[Array, 'atom']): Nuclear charges of atoms.
    cell_vectors (Float[Array, '3 3']): Unit cell vectors.
    g_vector_grid (Float[Array, 'x y z 3']): Grid of G-vectors in reciprocal space.
    vol (Float): Volume of the unit cell.
    ewald_eta (float): Ewald splitting parameter.
    ewald_cutoff (float): Real-space cutoff for Ewald summation.

  Returns:
    Float: Nuclear-nuclear repulsion energy.
  """
  ewald_grid = translation_vectors(cell_vectors, ewald_cutoff)
  return ewald_coulomb_repulsion(
    position, charge, g_vector_grid, vol, ewald_eta, ewald_grid
  )


def total_energy(
  coefficient: Complex[Array, "spin kpts band x y z"],
  position: Float[Array, "atom 3"],
  charge: Int[Array, "atom"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
  kohn_sham: bool = False,
  xc: str = 'lda',
  split: bool = False,
) -> Union[Float, Tuple[Float, Float, Float, Float]]:
  """Calculate the total electronic energy of the system.

  Computes the total energy as the sum of kinetic, external potential,
  Hartree, and exchange-correlation terms:

  .. math::
      E_{\text{tot}} = E_{\text{kin}} + E_{\text{ext}} + E_H + E_{xc}

  .. warning::
      This function does not include the nuclear-nuclear repulsion (Ewald) energy.
      For the complete total energy, the Ewald term must be added separately.

  Args:
    coefficient (Complex[Array, "spin kpts band x y z"]): Plane wave coefficients.
    position (Float[Array, "atom 3"]): Atomic positions in the unit cell.
    charge (Int[Array, "atom"]): Nuclear charges.
    g_vector_grid (Float[Array, "x y z 3"]): Grid of G-vectors in reciprocal space.
    kpts (Float[Array, "kpt 3"]): k-points in reciprocal space.
    vol (Float): Unit cell volume.
    occupation (Float[Array, "spin kpt band"], optional): Occupation numbers.
      If not provided, all states are considered fully occupied.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.
    xc (str, optional): Exchange-correlation functional type. Defaults to 'lda'.
    split (bool, optional): If True, return individual energy components. Defaults to False.

  Returns:
    Union[Float, Tuple[Float, Float, Float, Float]]: If split is False, returns
      the total electronic energy. If split is True, returns the individual
      components (kinetic, external, Hartree, exchange-correlation).
  """

  wave_grid_arr = pw.wave_grid(coefficient, vol)
  
  occupation = jnp.ones(
    shape=coefficient.shape[:3]
  ) if occupation is None else occupation

  density_grid = wave_to_density(wave_grid_arr, occupation)
  density_grid_rec = wave_to_density_reciprocal(wave_grid_arr, occupation)

  e_kin = kinetic(g_vector_grid, kpts, coefficient, occupation)
  e_ext = external(density_grid_rec, position, charge, g_vector_grid, vol)
  e_har = hartree(density_grid_rec, g_vector_grid, vol, kohn_sham)
  if xc == 'lda':
    e_xc = xc_lda(density_grid, vol, kohn_sham)
  else:
    raise NotImplementedError(f"xc {xc} is not supported yet.")

  if split:
    return e_kin, e_ext, e_har, e_xc

  return e_kin + e_ext + e_har + e_xc


def band_energy(
  coefficient: Complex[Array, "spin kpt band x y z"],
  position: Float[Array, "atom 3"],
  charge: Int[Array, "atom"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  occupation: Float[Array, "spin kpt band"],
  kohn_sham: bool = False,
  xc: str = 'lda'
):
  r"""Calculate the energy eigenvalues for each electronic state.

  Computes the energy eigenvalues by evaluating the expectation value of the
  single-particle Hamiltonian for each state:

  .. math::
  
      \varepsilon_{n\mathbf{k}} = \langle \psi_{n\mathbf{k}} |
      \hat{T} + \hat{V}_{\text{eff}} | \psi_{n\mathbf{k}} \rangle

  where :math:`\hat{T}` is the kinetic energy operator and
  :math:`\hat{V}_{\text{eff}}` is the effective potential.

  Args:
    coefficient (Complex[Array, "spin kpt band x y z"]): Plane wave coefficients.
    position (Float[Array, "atom 3"]): Atomic positions in the unit cell.
    charge (Int[Array, "atom"]): Nuclear charges.
    g_vector_grid (Float[Array, "x y z 3"]): Grid of G-vectors in reciprocal space.
    kpts (Float[Array, "kpt 3"]): k-points in reciprocal space.
    vol (Float): Unit cell volume.
    occupation (Float[Array, "spin kpt band"]): Occupation numbers.
    kohn_sham (bool, optional): If True, use Kohn-Sham formalism. Defaults to False.
    xc (str, optional): Exchange-correlation functional type. Defaults to 'lda'.

  Returns:
    Float[Array, "spin kpt band"]: Energy eigenvalues for each electronic state.
  """

  density_grid_sum = pw.density_grid(coefficient, vol, occupation)
  density_per_band = pw.density_grid(coefficient, vol)
  v_eff = potential.effective(
    density_grid_sum,
    position,
    charge,
    g_vector_grid,
    vol,
    split=False,
    xc=xc,
    kohn_sham=kohn_sham
  )
  e_eff = braket.real_braket(density_per_band, v_eff, vol)

  e_kin = kinetic(g_vector_grid, kpts, coefficient)
  assert np.array_equal(e_kin.shape, e_eff.shape)
  return safe_real(e_eff + e_kin)
