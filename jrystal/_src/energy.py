"""Energy functions. """
import numpy as np
import jax.numpy as jnp
from typing import Union
from jaxtyping import Float, Array, Int, Complex

from .utils import (
  absolute_square, safe_real, wave_to_density, wave_to_density_reciprocal
)
from . import braket, potential, pw
from .typing import OccupationArray, VectorGrid, ScalarGrid
from .ewald import ewald_coulomb_repulsion
from .grid import translation_vectors


def hartree(
  density_grid_reciprocal: ScalarGrid[Complex, 3],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float,
  kohn_sham: bool = False
) -> Float:
  r"""
  Compute the Hartree energy for plane wave orbitals in reciprocal space.

  .. math::
    E = 2\pi \sum_i \sum_k \sum_G \dfrac{n(G)^2}{\|G\|^2}

  Args:
    density_grid_reciprocal (ScalarGrid[Complex, 3]): The density of grid
    points in reciprocal space. Shape: [_, n1, n2, n3].
    g_vector_grid (VectorGrid[Float, 3]): G vector grid. Shape: [n1, n2, n3, 3].
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
  Externel energy for plane waves

  .. math::
      V = \sum_G \sum_i s_i(G) v_i(G)
      E = \int V(r) \rho(r) dr

  where

  .. math::
      s_i(G) = exp(jG\tau_i)
      v_i(G) = -4 \pi z_i / \Vert G \Vert^2

  Args:
    density_grid_reciprocal (ScalarGrid[Complex, 3]): the density of grid
      points in reciprocal space.
    position (Array): Coordinates of atoms in a unit cell. Shape: [atom 3].
    charge (Array): Charges of atoms. Shape: [atom].
    g_vector_grid (VectorGrid[Float, 3]): G vectors at grid points.

  Return:
    Float: External energy.

  """
  dim = g_vector_grid.shape[-1]
  if density_grid_reciprocal.ndim == dim + 1:
    density_grid_reciprocal = jnp.sum(density_grid_reciprocal, axis=0)

  v_externel_reciprocal = potential.externel_reciprocal(
    position, charge, g_vector_grid, vol
  )
  externel_energy = braket.reciprocal_braket(
    v_externel_reciprocal, density_grid_reciprocal, vol
  )

  return safe_real(externel_energy)


def kinetic(
  g_vector_grid: VectorGrid[Float, 3],
  k_vector_grid: VectorGrid[Float, 3],
  coeff_grid: ScalarGrid[Complex, 3],
  occupation: OccupationArray | None = None
) -> Union[Float, Float[Array, "num_spin num_k num_bands"]]:
  r"""Kinetic energy.

  .. math::
      E = 1/2 \sum_{G} |k + G|^2 c_{i,k,G}^2

  Args:
      g_vector_grid (VectorGrid[Float, 3]):  G vector grid.
      k_vector_grid (VectorGrid[Float, 3]):  k vector grid.
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
      density_grid (ScalarGrid[Float, 3]): the density of grid points in
        real space.
      vol (Float): the volume of unit cell.

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
  positions: Float[Array, 'num_atoms d'],
  charges: Float[Array, 'num_atoms'],
  cell_vectors: Float[Array, '3 3'],
  g_vector_grid: VectorGrid[Float, 3],
  vol: Float,
  ewald_eta: Float,
  ewald_cutoff: VectorGrid[Float, 3],
) -> Float:
  ewald_grid = translation_vectors(cell_vectors, ewald_cutoff)
  return ewald_coulomb_repulsion(
    positions, charges, g_vector_grid, vol, ewald_eta, ewald_grid
  )


def total_energy(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  positions: Float[Array, "num_atoms dim"],
  charges: Int[Array, "num_atoms"],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "num_k dim"],
  vol: Float,
  occupation: OccupationArray | None = None,
  kohn_sham: bool = False,
  xc: str = 'lda',
  split: bool = False,
) -> Float:
  """

  Args:

  coefficient (Complex[Array, "spin kpoint band *ndim"]): the plane wave 
  coefficients. You can get it from jrystal.pw.pw_coefficient.

  """

  wave_grid_arr = pw.wave_grid(coefficient, vol)
  density_grid = wave_to_density(wave_grid_arr, occupation)
  density_grid_rec = wave_to_density_reciprocal(wave_grid_arr, occupation)

  e_kin = kinetic(g_vector_grid, kpts, coefficient, occupation)
  e_ext = external(density_grid_rec, positions, charges, g_vector_grid, vol)
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
  positions: Float[Array, "num_atoms dim"],
  charges: Int[Array, "num_atoms"],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "num_k dim"],
  vol: Float,
  occupation: OccupationArray,
  kohn_sham: bool = False,
  xc: str = 'lda'
):

  density_grid_sum = pw.density_grid(coefficient, vol, occupation)
  density_per_band = pw.density_grid(coefficient, vol)
  v_eff = potential.effective(
    density_grid_sum,
    positions,
    charges,
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
