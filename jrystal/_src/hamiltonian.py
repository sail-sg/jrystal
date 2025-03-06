"""Hamiltonian matrix."""
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Complex, Bool
from typing import Optional, Union

from ._typing import OccupationArray, VectorGrid, ScalarGrid
from .kinetic import kinetic
from . import pw, braket, potential, utils, grid
from .hessian import complex_hessian


def _hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpt band *ndim"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: ScalarGrid[Float, 3],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = False
) -> Float[Array, "kpt band band"]:
  """Compute the hamiltonian matrix (hamiltonian-orbital matrix).

  The hamiltonian matrix in this project is defined by

  .. math::

    H_{ij} = < \psi_i | \hat{H} | \psi_j >

  Args:
    coefficient: the plane wave coefficients.
      The shape of coefficient must be [*batch, num_bands, x y z],
      where the last 3 axes are the grid axes, and the fourth from the end
      is the bands axis.
    effictive_density_grid (ScalarGrid[Float, 3]): the electron density of  
      effective potential evaluated at grid in real space.
    positions: atomic positions. Unit in Bohr.
    charges (Int[Array, 'num_atoms']): atomic numbers
    g_vector_grid (VectorGrid[Float, 3]): a g point grid.
    kpts (Float[Array, 'num_k 3']): k points
    vol (Float): the volume of unit cell.
    xc (str, optional): the name of xc functional. Defaults to 'lda'.

  Returns:

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

  t_kin = kinetic(g_vector_grid, kpts)
  f_kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=False, mode='kinetic'
  )

  return (f_eff + f_kin)[0]


def hamiltonian_matrix_trace(
  band_coefficient: Complex[Array, "spin kpt band *ndim"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: ScalarGrid[Float, 3],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "kpts 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True,
  keep_kpts_axis: bool = False,
  keep_spin_axis: bool = False,
) -> Union[Float, Float[Array, "spin"],  Float[Array, "spin kpts"]]:
  """Calculate the trace of the hamiltonian matrix. 

  Args:
      band_coefficient (Complex[Array, "spin kpt band *ndim"]): coefficients of bands
      positions (Float[Array, "atom 3"]): positions of atoms
      charges (Int[Array, "atom"]): charges of atoms
      effictive_density_grid (ScalarGrid[Float, 3]): effective density grid
      g_vector_grid (VectorGrid[Float, 3]): g vector grid
      kpts (Float[Array, "kpt 3"]): k points
      vol (Float): volume of unit cell
      xc (str, optional): name of xc functional. Defaults to 'lda'.
      kohn_sham (bool, optional): whether to use kohn-sham potential. Defaults to True.
      keep_kpts_axis (bool, optional): whether to keep kpts axis. Defaults to False.
      keep_spin_axis (bool, optional): whether to keep spin axis. Defaults to False.

  Returns:
      jnp.ndarray: The trace of the hamiltonian matrix. The full output has shape [spin, kpts], if both keep_kpts_axis and keep_spin_axis are True. 

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

  t_kin = kinetic(g_vector_grid, kpts)
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
  band_coefficient: Complex[Array, "spin kpt band *ndim"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  effictive_density_grid: ScalarGrid[Float, 3],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True,
) -> Float[Array, "kpt band band"]:
  r"""Compute the hamiltonian matrix in the orbitals.
  
  The hamiltonian matrix in this project is defined by

  .. math::

    F_{ij} = < \psi_i | \hat{H} | \psi_j >
  
  .. note::
    Currently, spin polarization of hamiltonian matrix is not supported.

  Args:
      coefficient (ScalarGrid[Complex, 3]): the plane wave coefficients.
        The shape of coefficient must be [*batch, num_bands, x y z],
        where the last 3 axes are the grid axes, and the fourth from the end
        is the bands axis.
      wave_grid (ScalarGrid[Complex, 3]): the wave function evaluated at grid
        in real space. Must have the same shape as coefficient.
      density_grid (ScalarGrid[Float, 3]): the electron density of effective
        potential evaluated at grid in real space.
      positions (Float[Array, 'num_atoms 3']): atomic positions. Unit in Bohr.
      charges (Int[Array, 'num_atoms']): atomic numbers
      g_vector_grid (VectorGrid[Float, 3]): a g point grid.
      kpts (Float[Array, 'num_k 3']): k points
      vol (Float): the volume of unit cell.
      xc (str, optional): the name of xc functional. Defaults to 'lda'.
  
  Returns:
    Float[Array, "kpt band band"]: The hamiltonian matrix.
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


def hamiltonian_matrix_basis(
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
  cell_vectors = grid.g2cell_vectors(g_vector_grid)
  r_vecs = grid.r_vectors(cell_vectors, g_vector_grid.shape[:-1])
  g_vec_list = jnp.reshape(g_vector_grid.at[freq_mask, :].get(), [-1, 3])

  _gr = jnp.einsum("g d, *n d -> g *n", g_vec_list,)