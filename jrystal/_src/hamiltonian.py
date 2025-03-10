"""Hamiltonian matrix operations for quantum mechanical calculations.

This module provides functions to compute Hamiltonian matrix elements and related quantities in a plane wave basis. The Hamiltonian includes both kinetic and effective potential terms, supporting both standard and Kohn-Sham DFT calculations.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Complex, Bool
from typing import Optional, Union

from .kinetic import kinetic_operator
from . import pw, braket, potential, utils, grid
from .hessian import complex_hessian


def _hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: Float[Array, "x y z"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = False
) -> Float[Array, "kpt band band"]:
  r"""Compute the Hamiltonian matrix elements between orbitals.

  The Hamiltonian matrix (hamiltonian-orbital matrix) is defined as:

  .. math::

    H_{ij} = \langle \psi_i | \hat{H} | \psi_j \rangle

  where :math:`\hat{H}` is the Hamiltonian operator composed of kinetic and effective potential terms:
  
  .. math::

    \hat{H} = \hat{T} + \hat{V}_{eff}

  The kinetic term :math:`\hat{T}` is computed in reciprocal space, while the effective potential term :math:`\hat{V}_{eff}` includes both the external potential from ions and the electron-electron interaction terms.

  Args:
    coefficient (Complex[Array, "spin kpt band x y z"]): Plane wave coefficients with shape [spin, kpt, band, x, y, z]. The last three dimensions represent the spatial grid.
    positions (Float[Array, "atom 3"]): Atomic positions in Bohr units with shape [atom, 3].
    charges (Int[Array, "atom"]): Atomic numbers for each atom with shape [atom].
    effictive_density_grid (Float[Array, "x y z"]): Electron density for effective potential evaluated  on real space grid with shape [x, y, z].
    g_vector_grid (Float[Array, "x y z 3"]): G-vector grid in reciprocal space with shape [x, y, z, 3].
    kpts (Float[Array, "kpt 3"]): K-points in reciprocal space with shape [kpt, 3].
    vol (Float): Volume of the unit cell.
    xc (str): Exchange-correlation functional name. Defaults to 'lda'.
    kohn_sham: Whether to use Kohn-Sham potential. Defaults to False.

  Returns:
    Float[Array, "kpt band band"]: Float array of shape [kpt, band, band] containing the Hamiltonian matrix elements between all pairs of bands at each k-point.
  """

  v_eff = potential.effective(
    effictive_density_grid,
    positions,
    charges,
    g_vector_grid,
    vol,
    split=False,
    xc=xc,
    kohn_sham=kohn_sham
  )
  wave_grid = pw.wave_grid(coefficient, vol)
  f_eff = braket.expectation(wave_grid, v_eff, vol, diagonal=False, mode="real")

  t_kin = kinetic_operator(g_vector_grid, kpts)
  f_kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=False, mode='kinetic'
  )

  return (f_eff + f_kin)[0]


def hamiltonian_matrix_trace(
  band_coefficient: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: Float[Array, "x y z"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True,
  keep_kpts_axis: bool = False,
  keep_spin_axis: bool = False,
) -> Union[Float, Float[Array, "spin"],  Float[Array, "spin kpt"]]:
  r"""Calculate the trace of the Hamiltonian matrix.

  The trace is computed as:

  .. math::

    \text{Tr}(\hat{H}) = \sum_i H_{ii} = \sum_i \langle \psi_i | \hat{H} | \psi_i \rangle

  This quantity represents the sum of the diagonal elements of the Hamiltonian matrix,
  which is useful for various physical quantities like total energy calculations.

  Args:
    band_coefficient (Complex[Array, "spin kpt band x y z"]): Plane wave coefficients with shape [spin, kpt, band, x, y, z]. The last three dimensions represent the spatial grid.
    positions (Float[Array, "atom 3"]): Atomic positions in Bohr units with shape [atom, 3].
    charges (Int[Array, "atom"]): Atomic numbers for each atom with shape [atom].
    effictive_density_grid (Float[Array, "x y z"]): Electron density for effective potential evaluated  on real space grid with shape [x, y, z].
    g_vector_grid (Float[Array, "x y z 3"]): G-vector grid in reciprocal space with shape [x, y, z, 3].
    kpts (Float[Array, "kpt 3"]): K-points in reciprocal space with shape [kpt, 3].
    vol (Float): Volume of the unit cell.
    xc (str): Exchange-correlation functional name. Defaults to 'lda'.
    kohn_sham (bool): Whether to use Kohn-Sham potential. Defaults to True.
    keep_kpts_axis (bool): If True, retains the k-points axis in output. Defaults to False.
    keep_spin_axis (bool): If True, retains the spin axis in output. Defaults to False.

  Returns:
    Union[Float, Float[Array, "spin"],  Float[Array, "spin kpt"]]: The trace of the Hamiltonian matrix. The shape depends on keep_kpts_axis and keep_spin_axis:
  
    - If both False: returns a scalar Float
    - If keep_spin_axis True: returns Float[Array, "spin"]
    - If keep_kpts_axis True: returns Float[Array, "kpt"]
    - If both True: returns Float[Array, "spin kpt"]
  """
  num_spin = band_coefficient.shape[0]
  if num_spin == 2:
    raise NotImplementedError(
      "Spin-unrestricted hamiltonian matrix is not implemented."
    )

  v_eff = potential.effective(
    effictive_density_grid,
    positions,
    charges,
    g_vector_grid,
    vol,
    False,
    xc,
    kohn_sham
  )
  wave_grid = pw.wave_grid(band_coefficient, vol)
  f_eff = braket.expectation(wave_grid, v_eff, vol, diagonal=True, mode="real")

  t_kin = kinetic_operator(g_vector_grid, kpts)
  f_kin = braket.expectation(
    band_coefficient, t_kin, vol, diagonal=True, mode='kinetic'
  )       #  [spin, kpt, band]

  hamil_trace = (f_eff + f_kin).real
  if not keep_kpts_axis:
    hamil_trace = jnp.sum(hamil_trace, axis=1)
  if not keep_spin_axis:
    hamil_trace = jnp.sum(hamil_trace, axis=0)
  return jnp.sum(hamil_trace, axis=-1)


def hamiltonian_matrix(
  band_coefficient: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: Float[Array, "x y z"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True,
) -> Complex[Array, "kpt band band"]:
  r"""Compute the full Hamiltonian matrix in the orbital basis.
  
  This function computes the complete Hamiltonian matrix including both diagonal
  and off-diagonal elements. The matrix elements are defined as:

  .. math::

    H_{ij} = \langle \psi_i | \hat{H} | \psi_j \rangle

  where :math:`\hat{H} = \hat{T} + \hat{V}_{eff}` is the total Hamiltonian operator.

  Args:
    band_coefficient (Complex[Array, "spin kpt band x y z"]): Plane wave coefficients with shape [spin, kpt, band, x, y, z]. The last three dimensions represent the spatial grid.
    positions (Float[Array, "atom 3"]): Atomic positions in Bohr units with shape [atom, 3].
    charges (Int[Array, "atom"]): Atomic numbers for each atom with shape [atom].
    effictive_density_grid (Float[Array, "x y z"]): Electron density for effective potential evaluated on real space grid with shape [x, y, z].
    g_vector_grid (Float[Array, "x y z 3"]): G-vector grid in reciprocal space with shape [x, y, z, 3].
    kpts (Float[Array, "kpt 3"]): K-points in reciprocal space with shape [kpt, 3].
    vol (Float): Volume of the unit cell.
    xc (str): Exchange-correlation functional name. Defaults to 'lda'.
    kohn_sham (bool): Whether to use Kohn-Sham potential. Defaults to True.

  Returns:
    Complex[Array, "kpt band band"]: Complex array of shape [kpt, band, band] containing the complete Hamiltonian matrix elements between all pairs of bands at each k-point.
  """
  num_bands = band_coefficient.shape[-4]
  num_spin = band_coefficient.shape[0]

  if num_spin == 2:
    raise NotImplementedError(
      "Spin-unrestricted hamiltonian matrix is not implemented."
    )

  def hamil_k(k, coeff_k):
    k = jnp.reshape(k, [-1, 3])
    coeff_k = jnp.expand_dims(coeff_k, axis=1)

    def efun(u):
      _coeff = jnp.einsum("i, *n i a b c -> *n a b c", u, coeff_k)
      _coeff = jnp.expand_dims(_coeff, axis=-4)

      band_energies = hamiltonian_matrix_trace(
        _coeff,
        positions,
        charges,
        effictive_density_grid,
        g_vector_grid,
        k,
        vol,
        xc,
        kohn_sham,
      )

      return 0.5 * jnp.sum(band_energies).astype(band_coefficient.dtype)
    x = jnp.ones(num_bands, dtype=band_coefficient.dtype)
    return complex_hessian(efun, x)

  h1 = jax.vmap(hamil_k, in_axes=(0, 1), out_axes=0)(kpts, band_coefficient)
  return h1   #  shape: [kpt, band, band]


def _hamiltonian_matrix_basis(
  freq_mask: Int[Array, "x y z"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: Float[Array, "x y z"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True,
) -> Complex[Array, "kpt band band"]:
  r"""Compute the Hamiltonian matrix in the plane wave basis.
  
  This internal function computes the Hamiltonian matrix elements in the plane wave basis set defined by the frequency mask. It uses complex hessian calculations to efficiently compute the matrix elements.

  Args:
    freq_mask (Int[Array, "x y z"]): Integer mask of shape [x, y, z] indicating which plane waves to include in the basis (1 for included, 0 for excluded).
    positions (Float[Array, "atom 3"]): Atomic positions in Bohr units with shape [atom, 3].
    charges (Int[Array, "atom"]): Atomic numbers for each atom with shape [atom].
    effictive_density_grid (Float[Array, "x y z"]): Electron density for effective potential evaluated on real space grid with shape [x, y, z].
    g_vector_grid (Float[Array, "x y z 3"]): G-vector grid in reciprocal space with shape [x, y, z, 3].
    kpts (Float[Array, "kpt 3"]): K-points in reciprocal space with shape [kpt, 3].
    vol (Float): Volume of the unit cell.
    xc (str): Exchange-correlation functional name. Defaults to 'lda'.
    kohn_sham (bool): Whether to use Kohn-Sham potential. Defaults to True.

  Returns:
    Complex[Array, "kpt band band"]: Complex array of shape [kpt, band, band] containing the Hamiltonian matrix elements in the plane wave basis at each k-point.
  """
  
  num_basis = jnp.sum(freq_mask)
  
  if g_vector_grid.dtype == jnp.float64:
    _dtype = jnp.complex128
  elif g_vector_grid.dtype == jnp.float32:
    _dtype = jnp.complex64
  else:
    raise ValueError(f"Unsupported dtype: {g_vector_grid.dtype}")

  def hamil_k(k):
    k = k.reshape([-1, 3])

    def efun(u):
      u = jnp.expand_dims(u, axis=(0, 1, 3))  # add spin, kpt, and band axes

      _coeff = utils.expand_coefficient(u, freq_mask)
      hamil_trace = hamiltonian_matrix_trace(
        _coeff,
        positions,
        charges,
        effictive_density_grid,
        g_vector_grid,
        k,
        vol,
        xc,
        kohn_sham,
      )
      return 0.5 * jnp.sum(hamil_trace).astype(_dtype)

    x = jnp.ones(num_basis, dtype=_dtype)
    return complex_hessian(efun, x)

  hamil_basis = jax.vmap(hamil_k)(kpts)
  return hamil_basis
