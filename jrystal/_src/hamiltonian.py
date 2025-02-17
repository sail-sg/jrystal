"""Hamiltonian matrix."""
import jax
import jax.numpy as jnp
from jaxtyping import Float, Array, Int, Complex
import einops
from typing import Optional

from .typing import OccupationArray, VectorGrid, ScalarGrid
from .kinetic import kinetic
from . import pw, braket, potential


def __hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  density_grid: ScalarGrid[Float, 3],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  g_vector_grid: VectorGrid[Float, 3],
  k_points: Float[Array, "kpoint 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = False
) -> Float[Array, "kpoint band band"]:
  """Compute the hamiltonian matrix.

  The hamiltonian matrix in this project is defined by

  .. math::
    H_{ij} = < \psi_i | \hat{H} | \psi_j >

  Args:
    coefficient: the plane wave coefficients.
      The shape of coefficient must be [*batch, num_bands, n1, n2, n3],
      where the last 3 axes are the grid axes, and the fourth from the end
      is the bands axis.
    wave_grid: the wave function evaluated at grid
      in real space. Must have the same shape as coefficient.
    density_grid (ScalarGrid[Float, 3]): the electron density of effective
      potential evaluated at grid in real space.
    positions: atomic positions. Unit in Bohr.
    charges (Int[Array, 'num_atoms']): atomic numbers
    g_vector_grid (VectorGrid[Float, 3]): a g point grid.
    k_points (Float[Array, 'num_k 3']): k points
    vol (Float): the volume of unit cell.
    xc (str, optional): the name of xc functional. Defaults to 'lda'.

  Returns:

  """

  v_eff = potential.effective(
    density_grid,
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

  t_kin = kinetic(g_vector_grid, k_points)
  f_kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=False, mode='kinetic'
  )

  return f_eff + f_kin


def hamiltonian_matrix_trace(
  band_coefficient: Complex[Array, "spin kpoint band *ndim"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  ground_state_density_grid: ScalarGrid[Float, 3],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "kpoint 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True
):
  v_eff = potential.effective(
    ground_state_density_grid,
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
  )
  return jnp.sum(f_eff + f_kin).real


def hamiltonian_matrix(
  band_coefficient: Complex[Array, "spin kpoint band *ndim"],
  positions: Float[Array, "atom 3"],
  charges: Int[Array, "atom"],
  ground_state_density_grid: ScalarGrid[Float, 3],
  g_vector_grid: VectorGrid[Float, 3],
  kpts: Float[Array, "kpoint 3"],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True
) -> Float[Array, "kpoint band band"]:
  # TODO: Bugs exist. Need to check later.
  """Compute the hamiltonian matrix.
  The hamiltonian matrix in this project is defined by

  .. math::
    F_{ij} = < \psi_i | \hat{H} | \psi_j >

  Args:
      coefficient (ScalarGrid[Complex, 3]): the plane wave coefficients.
        The shape of coefficient must be [*batch, num_bands, n1, n2, n3],
        where the last 3 axes are the grid axes, and the fourth from the end
        is the bands axis.
      wave_grid (ScalarGrid[Complex, 3]): the wave function evaluated at grid
        in real space. Must have the same shape as coefficient.
      density_grid (ScalarGrid[Float, 3]): the electron density of effective
        potential evaluated at grid in real space.
      positions (Float[Array, 'num_atoms 3']): atomic positions. Unit in Bohr.
      charges (Int[Array, 'num_atoms']): atomic numbers
      g_vector_grid (VectorGrid[Float, 3]): a g point grid.
      k_points (Float[Array, 'num_k 3']): k points
      vol (Float): the volume of unit cell.
      xc (str, optional): the name of xc functional. Defaults to 'lda'.
  """
  num_bands = band_coefficient.shape[-4]

  def efun(u, kpts):
    kpts = kpts.reshape([-1, 3])
    u = u + 0 * 1.j

    _wave_grid = pw.wave_grid(band_coefficient, vol)
    _coeff = jnp.einsum("i, *n i a b c -> *n a b c", u, band_coefficient)
    _wave_grid = jnp.einsum("i, *n i a b c ->*n a b c", u, _wave_grid)

    _coeff = jnp.expand_dims(_coeff, axis=-4)
    _wave_grid = jnp.expand_dims(_wave_grid, axis=-4)

    band_energies = hamiltonian_matrix_trace(
      _coeff,
      positions,
      charges,
      ground_state_density_grid,
      g_vector_grid,
      kpts,
      vol,
      xc,
      kohn_sham
    )

    return jnp.sum(band_energies)

  return jax.vmap(lambda k: jax.hessian(efun)(jnp.ones(num_bands), k))(kpts)
