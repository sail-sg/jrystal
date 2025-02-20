"""Energy functions. """
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from . import braket, potential, pw
from .ewald import ewald_coulomb_repulsion
from .grid import translation_vectors
from .typing import OccupationArray, ScalarGrid, VectorGrid
from .utils import (
  absolute_square, safe_real, wave_to_density, wave_to_density_reciprocal
)


def hartree(
  density_grid_reciprocal: ScalarGrid[Complex, 3],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float,
  kohn_sham: bool = False
) -> Float:
  r"""
  Compute the Hartree energy for plane wave orbitals.

  The calculation is performed in reciprocal space. In
  :py:func:`hartree_reciprocal`, we have computed the Hartree potential in
  reciprocal space given by

  .. math::
    \hat{V}(G) = 4 \pi \dfrac{\hat{n}(G)}{\|G\|^2},
    \hat{V}(0) = 0.

  Using the Parseval's identity, the real space integration of the Hartree
  energy can be calculated as the summation over the reciprocal lattice as:

  .. math::
    E = 2\pi \sum_G \dfrac{\hat{n}(G)^2}{\|G\|^2}

  This is evaluated by the braket operation in :py:func:`reciprocal_braket`.

  Args:
    density_grid_reciprocal (ScalarGrid[Complex, 3]): The electron density on
    reciprocal lattice.
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    vol (Float): Volume of the unit cell.
    kohn_sham (bool, optional): If True, use Kohn-Sham potential. Defaults to
    False.

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
  density_grid_reciprocal: ScalarGrid[Complex, 3],
  position: Float[Array, 'atom 3'],
  charge: Float[Array, 'atom'],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float
) -> Float:
  r"""
  Compute the External energy for plane wave orbitals.

  Similar to the Hartree energy, we use the Parseval's identity to calculate
  the external energy by summation over the reciprocal lattice as:

  .. math::
      E = \sum_G \hat{n(G)}\sum_k s_k(G) v_k(G) 

  where the summation $k$ is over all atoms in the unit cell and $G$ is over
  the reciprocal lattice.

  Args:
    density_grid_reciprocal (ScalarGrid[Complex, 3]): The electron density on
    reciprocal lattice.
    position (VectorGrid[Float, 3]): Coordinates of atoms in a unit cell.
    charge (VectorGrid[Float, 1]): Charges of atoms.
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    vol (Float): the volume of unit cell.

  Return:
    Float: External energy.

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
  g_vector_grid: VectorGrid[Float, 3],
  k_vector_grid: VectorGrid[Float, 3],
  coeff_grid: ScalarGrid[Complex, 3],
  occupation: Optional[OccupationArray] = None
) -> Union[Float, Float[Array, "num_spin num_k num_bands"]]:
  r"""Kinetic energy.

  The kinetic energy is calculated by contracting the plane wave coefficients
  with the kinetic energy matrix obtained from :py:func:`kinetic`.

  .. math::
      E = 1/2 \sum_{G} |k + G|^2 c_{i,k,G}^2

  Args:
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    k_vector_grid (VectorGrid[Float, 3]): k points.
    coeff_grid (ScalarGrid[Complex, 3]): Plane wave coefficient.
    occupation (Array, optional): occupation array. If provided, then the
      function will be reduced by applying occupation number. If not provided,
      then the function will return the kinetic energy of all the orbitals.

  Returns:
    Float or Float[Array, "num_spin num_k num_bands"]]: kinetic energy.
  """

  dim = g_vector_grid.shape[-1]

  _g = jnp.expand_dims(g_vector_grid, axis=range(3))
  _k = jnp.expand_dims(
    k_vector_grid, axis=[0] + [i + 2 for i in range(dim + 1)]
  )
  e_kin = jnp.sum((_g + _k)**2, axis=-1)  # [1, nk, ni, N1, N2, N3]
  e_kin = jnp.sum(e_kin * absolute_square(coeff_grid), axis=range(3, dim + 3))

  if occupation is not None:
    e_kin = jnp.sum(e_kin * occupation) / 2
  else:
    e_kin /= 2

  return safe_real(e_kin)


def xc_lda(
  density_grid: ScalarGrid[Float, 3],
  vol: Float,
  kohn_sham: bool = False
) -> Float:
  r"""local density approximation potential.

  NOTE: this is a non-polarized lda potential

  .. math::
      E_{\rm x}^{\mathrm{LDA}}[\rho] = - \frac{3}{4}\left( \frac{3}{\pi} \right)^{1/3}\int\rho(\mathbf{r})^{4/3}  # noqa: E501

  Args:
    density_grid (ScalarGrid[Float, 3]): the density on real space grid.
    vol (Float): the volume of unit cell.
    kohn_sham (bool, optional): If True, use Kohn-Sham potential. Defaults to
    False.

  Returns:
    ScalarGrid[Float, 3]: the variation of the lda energy with respect to the density.

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
  position: Float[Array, 'num_atoms d'],
  charge: Float[Array, 'num_atoms'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float,
  ewald_eta: Float,
  ewald_cutoff: VectorGrid[Float, 3],
) -> Float:
  """
  Compute the nuclear repulsion energy.

  This function calculates the nuclear repulsion energy using Ewald summation.

  Args:
    position (VectorGrid[Float, 3]): Coordinates of atoms in a unit cell.
    charge (VectorGrid[Float, 1]): Charges of atoms.
    cell_vectors (VectorGrid[Float, 3]): cell vectors of crystal.
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    vol (Float): Volume of the unit cell.
    ewald_eta (Float): Ewald decomposition parameter.
    ewald_cutoff (VectorGrid[Float, 3]): Ewald cutoff.

  Returns:
    Float: Nuclear repulsion energy.
  """
  ewald_grid = translation_vectors(cell_vectors, ewald_cutoff)
  return ewald_coulomb_repulsion(
    position, charge, g_vector_grid, vol, ewald_eta, ewald_grid
  )


def total_energy(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  position: Float[Array, "num_atoms dim"],
  charge: Int[Array, "num_atoms"],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "num_k dim"],
  vol: Float,
  occupation: Optional[OccupationArray] = None,
  kohn_sham: bool = False,
  xc: str = 'lda',
  split: bool = False,
) -> Float:
  """
  Compute the total energy of the system.

  This function first calculates the electron density and then use it to
  calculate different components of the energy using the functions above.

  Args:
    coefficient (ScalarGrid[Complex, 3]): Coefficients of the plane-wave
    orbitals.
    position (VectorGrid[Float, 3]): Coordinates of atoms in a unit cell.
    charge (VectorGrid[Float, 1]): Charges of atoms.
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    kpts (VectorGrid[Float, 3]): k points.
    vol (Float): Volume of the unit cell.
    occupation (Array, optional): occupation array. If provided, then the
    function will be reduced by applying occupation number. If not provided,
    then the function will return the kinetic energy of all the orbitals.
    kohn_sham (bool, optional): If True, use Kohn-Sham potential. Defaults to
    False.
    xc (str): Exchange-correlation functional. Defaults to "lda".
    split (bool): If True, return split energy [e_kin, e_ext, e_har, e_xc].

  Returns:
    Float: Total energy of the system.
  """

  wave_grid_arr = pw.wave_grid(coefficient, vol)
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
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  position: Float[Array, "num_atoms dim"],
  charge: Int[Array, "num_atoms"],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "num_k dim"],
  vol: Float,
  occupation: OccupationArray,
  kohn_sham: bool = False,
  xc: str = 'lda'
):
  """
  TODO:

  Args:
    coefficient (ScalarGrid[Complex, 3]): Coefficients of the plane-wave
    orbitals.
    position (VectorGrid[Float, 3]): Coordinates of atoms in a unit cell.
    charge (VectorGrid[Float, 1]): Charges of atoms.
    g_vector_grid (VectorGrid[Float, 3]): reciprocal lattice vector.
    kpts (VectorGrid[Float, 3]): k points.
    vol (Float): Volume of the unit cell.
    occupation (Array, optional): occupation array. If provided, then the
    function will be reduced by applying occupation number. If not provided,
    then the function will return the kinetic energy of all the orbitals.
    kohn_sham (bool, optional): If True, use Kohn-Sham potential. Defaults to
    False.
    xc (str): Exchange-correlation functional. Defaults to "lda".

  Returns:

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
