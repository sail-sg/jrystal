"""Norm Conserving Pseudopotential for Plane Waves. """
import numpy as np
import jax
import jax.numpy as jnp
from typing import List, Union, Optional
from jaxtyping import Float, Array, Complex, Int
from einops import einsum

from ..._src import braket
from ..._src.typing import OccupationArray, ScalarGrid, VectorGrid
from ..._src import potential, energy, hamiltonian
from ..._src.utils import wave_to_density
from ..._src import pw
from ..spherical import legendre_to_sph_harm
# from ..projector import _legendre_to_sph_har
from ..dataclass import NormConservingPseudopotential
from ..local import (
  _hamiltonian_local, hamiltonian_local, _energy_local, energy_local
)
from ..beta import beta_sbt_grid_multi_atoms


def _potential_nonlocal_square_root(
  position: Float[Array, 'num_atom d'],
  g_vector_grid: Float[Array, "*nd d"],
  k_points: Float[Array, "num_k d"],
  r_grid: List[Float[Array, "num_r"]],
  nonlocal_beta_grid: List[Float[Array, "num_beta num_r"]],
  nonlocal_angular_momentum: List[List[int]]
) -> Complex[Array, "num_k num_atom num_beta *nd nphi"]:
  """
    Square root of the nonlocal pseudopotential. The Nonlocal PP hamiltonain
    is defined by:

        < C | V_nl(K, K') | C >

    where V_nl = F D F^\dagger, where F can be obtained from this function.

  """
  assert len(nonlocal_beta_grid) == len(nonlocal_angular_momentum)

  gk_vector_grid = jnp.expand_dims(
    k_points, axis=(1, 2, 3)
  ) + jnp.expand_dims(g_vector_grid, 0)  # [nk n1 n2 n3 3]

  # sbt for beta function and intepolate
  beta_gk = beta_sbt_grid_multi_atoms(
    r_grid, nonlocal_beta_grid, nonlocal_angular_momentum, gk_vector_grid
  )  # shape [num_atom num_beta nk n1 n2 n3]

  # kernel trick for legendre polynormials.
  kappa_all = []
  for l_atom in nonlocal_angular_momentum:
    kappa_list = []
    for ln in l_atom:
      kappa_list.append(legendre_to_sph_harm(int(ln)))
    kappa = []
    for k in kappa_list:
      kappa.append(k(gk_vector_grid))
    kappa = jnp.stack(kappa)  # shape: [num_beta nk n1 n2 n3 dim_phi]
    kappa_all.append(kappa)

  kappa = jnp.stack(kappa_all)  # shape: [num_atom num_beta nk n1 n2 n3 dim_phi]

  # structure factor
  structure_factor = jnp.exp(
    1.j * jnp.matmul(gk_vector_grid, position.transpose())
  )
  # shape: [nk n1 n2 n3 na]

  return einsum(
    kappa,
    structure_factor,
    beta_gk,
    "na nb nk a b c nphi, nk a b c na, na nb nk a b c -> nk na nb a b c nphi"
  ) / jnp.sqrt(2)  # factor of 1/2 is due to the conversion of unit.


def hamiltonian_nonlocal(
  pw_coefficients: Complex[Array, "*batchs n1 n2 n3"],
  positions: Float[Array, 'na 3'],
  g_vector_grid: Float[Array, "*nd d"],
  k_points: Float[Array, "*nd d"],
  pseudo_r_grid: Float[Array, "nr"],
  nonlocal_beta_grid: Float[Array, "j nr"],
  nonlocal_angular_momentum: List[int],
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
) -> Complex[Array, "ns nk nb nb"]:
  potential_nl_sqrt = _potential_nonlocal_square_root(
    positions,
    g_vector_grid,
    k_points,
    pseudo_r_grid,
    nonlocal_beta_grid,
    nonlocal_angular_momentum
  )

  _f_matrix = einsum(
    pw_coefficients,
    potential_nl_sqrt,
    "ns nk nband a b c, nk na nbeta a b c nphi -> ns nk nband na nbeta nphi"
  )

  return 4 * jnp.pi / vol * einsum(
    jnp.conj(_f_matrix),
    jnp.stack(nonlocal_d_matrix),
    _f_matrix,
    "ns nk b1 na j1 nphi, na j1 j2, ns nk b2 na j2 nphi -> ns nk b1 b2"
  )


def _hamiltonian_nonlocal(
  pw_coefficients,
  potential_nl_sqrt,
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
):
  # this function is for seperating the potential_nl_sqrt part that may not
  # be tracable by jax.
  _f_matrix = einsum(
    pw_coefficients,
    potential_nl_sqrt,
    "s k band a b c, k atom beta a b c phi -> s k band atom beta phi"
  )

  return 4 * jnp.pi / vol * einsum(
    jnp.conj(_f_matrix),
    jnp.stack(nonlocal_d_matrix),
    _f_matrix,
    "ns nk b1 na j1 nphi, na j1 j2, ns nk b2 na j2 nphi -> ns nk b1 b2"
  )


def _energy_nonlocal(
  pw_coefficients: Complex[Array, "*batchs n1 n2 n3"],
  potential_nl_sqrt: Array,
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
  occupation: Optional[OccupationArray] = None,
) -> Float:
  # this function is for seperating the potential_nl_sqrt part that may not
  # be tracable by jax.

  hamil_nl = _hamiltonian_nonlocal(
    pw_coefficients, potential_nl_sqrt, nonlocal_d_matrix, vol
  )  # shape: [spin kpoint band band]

  return jnp.sum(jax.vmap(jax.vmap(jnp.diag))(hamil_nl) * occupation).real


def energy_nonlocal(
  pw_coefficients: Complex[Array, "*batchs n1 n2 n3"],
  positions: Float[Array, 'na 3'],
  g_vector_grid: Float[Array, "*nd d"],
  k_vec: Float[Array, "nk 3"],
  r_grid: Float[Array, "nr"],
  nonlocal_beta_grid: Float[Array, "j nr"],
  nonlocal_angular_momentum: List[int],
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
  occupation: Optional[OccupationArray] = None,
) -> Float:
  hamil_nl = hamiltonian_nonlocal(
    pw_coefficients,
    positions,
    g_vector_grid,
    k_vec,
    r_grid,
    nonlocal_beta_grid,
    nonlocal_angular_momentum,
    nonlocal_d_matrix,
    vol,
  )
  return jnp.sum(jax.vmap(jax.vmap(jnp.diag))(hamil_nl) * occupation)


def _hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: ScalarGrid[Float, 3],
  potential_local_grid_reciprocal,
  potential_nonlocal_grid_sqrt,
  g_vector_grid: VectorGrid[Float, 3],
  k_points: Float[Array, "num_k d"],
  nonlocal_d_matrix: List[Float[Array, "num_beta num_beta"]],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = False
):
  dim = k_points.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  ext_nloc = _hamiltonian_nonlocal(
    coefficient, potential_nonlocal_grid_sqrt, nonlocal_d_matrix, vol
  )
  ext_loc = _hamiltonian_local(wave_grid, potential_local_grid_reciprocal, vol)

  hamiltonian_density_grid_reciprocal = jnp.fft.fftn(
    hamiltonian_density_grid, axes=range(-dim, 0)
  )

  kin = hamiltonian.kinetic(g_vector_grid, k_points)
  h_kin = braket.expectation(
    coefficient, kin, vol, diagonal=False, mode='kinetic'
  )

  har = potential.hartree_reciprocal(
    hamiltonian_density_grid_reciprocal, g_vector_grid
  )
  har = jnp.fft.ifftn(har, axes=range(-dim, 0))
  lda = potential.xc_lda(hamiltonian_density_grid)
  v_s = har + lda
  h_s = braket.expectation(wave_grid, v_s, vol, diagonal=False, mode="real")

  return ext_nloc + ext_loc + h_s + h_kin


# TODO: define containers for pseudopotentials
def hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: ScalarGrid[Float, 3],
  r_vector_grid: Float[Array, "*nd d"],
  g_vector_grid: VectorGrid[Float, 3],
  k_points: Float[Array, "num_k 3"],
  positions: Float[Array, "num_atoms 3"],
  r_grid: Int[Array, "nr"],
  local_potential_grid: Float[Array, "nr"],
  local_potential_charge: Int[Array, "num_atoms"],
  nonlocal_beta_grid: List[Float[Array, "num_beta num_r"]],
  nonlocal_d_matrix: List[Float[Array, "num_beta num_beta"]],
  nonlocal_angular_momentum: List[List[int]],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = False
):
  dim = positions.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  ext_nloc = hamiltonian_nonlocal(
    coefficient,
    positions,
    g_vector_grid,
    k_points,
    r_grid,
    nonlocal_beta_grid,
    nonlocal_angular_momentum,
    nonlocal_d_matrix,
    vol
  )
  ext_loc = hamiltonian_local(
    wave_grid,
    positions,
    r_vector_grid,
    g_vector_grid,
    r_grid,
    local_potential_grid,
    local_potential_charge,
    vol
  )

  hamiltonian_reciprocal_density_grid = jnp.fft.fftn(
    hamiltonian_density_grid, axes=range(-dim, 0)
  )

  kin = hamiltonian.kinetic(g_vector_grid, k_points)
  h_kin = braket.expectation(
    coefficient, kin, vol, diagonal=False, mode='kinetic'
  )

  har = potential.hartree_reciprocal(
    hamiltonian_reciprocal_density_grid, g_vector_grid
  )
  har = jnp.fft.ifftn(har, axes=range(-dim, 0))
  lda = potential.xc_lda(hamiltonian_density_grid)
  v_s = har + lda
  h_s = braket.expectation(wave_grid, v_s, vol, diagonal=False, mode="real")

  return ext_nloc + ext_loc + h_s + h_kin


def _hamiltonian_trace(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: ScalarGrid[Float, 3],
  potential_local_grid_reciprocal,
  potential_nonlocal_grid_sqrt,
  g_vector_grid: VectorGrid[Float, 3],
  k_points: Float[Array, "num_k 3"],
  nonlocal_d_matrix: List[Float[Array, "num_beta num_r"]],
  vol: Float,
) -> Float:
  dim = k_points.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  occupation = jnp.ones(shape=wave_grid.shape[:3], dtype=k_points.dtype)

  density = wave_to_density(wave_grid, occupation)
  reciprocal_density_grid = jnp.fft.fftn(density, axes=range(-dim, 0))

  ext_nloc = _energy_nonlocal(
    coefficient,
    potential_nonlocal_grid_sqrt,
    nonlocal_d_matrix,
    vol,
    occupation
  )
  ext_loc = _energy_local(
    reciprocal_density_grid, potential_local_grid_reciprocal, vol
  )

  hamiltonian_reciprocal_density_grid = jnp.fft.fftn(
    hamiltonian_density_grid, axes=range(-dim, 0)
  )

  v_har_reciprocal = potential.hartree_reciprocal(
    hamiltonian_reciprocal_density_grid, g_vector_grid
  )
  v_har = jnp.fft.ifftn(v_har_reciprocal, axes=range(-3, 0))
  har = braket.expectation(wave_grid, v_har, vol, diagonal=True, mode="real")
  # har = 0
  v_lda = potential.xc_lda(hamiltonian_density_grid)
  lda = braket.expectation(wave_grid, v_lda, vol, diagonal=True, mode="real")
  h_s = jnp.sum(har + lda)

  t_kin = hamiltonian.kinetic(g_vector_grid, k_points)
  kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=True, mode='kinetic'
  )
  kin = jnp.sum(kin)

  return (ext_nloc + ext_loc + h_s + kin).real


def hamiltonian_trace(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: ScalarGrid[Float, 3],
  r_vector_grid: Float[Array, "*nd d"],
  g_vector_grid: VectorGrid[Float, 3],
  k_points: Float[Array, "num_k 3"],
  positions: Float[Array, "num_atoms 3"],
  r_grid: Int[Array, "nr"],
  local_potential_grid: Float[Array, "nr"],
  local_potential_charge: Int[Array, "num_atoms"],
  nonlocal_beta_grid: List[Float[Array, "num_beta num_r"]],
  nonlocal_d_matrix: List[Float[Array, "num_beta num_beta"]],
  nonlocal_angular_momentum: List[List[int]],
  vol: Float,
):
  dim = positions.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  occupation = np.ones(shape=wave_grid.shape[:3])
  density = wave_to_density(wave_grid, occupation)

  reciprocal_density_grid = jnp.fft.fftn(density, axes=range(-dim, 0))

  hamil_reciprocal_density_grid = jnp.fft.fftn(
    hamiltonian_density_grid, axes=range(-dim, 0)
  )
  ext_nloc = energy_nonlocal(
    coefficient,
    occupation,
    positions,
    g_vector_grid,
    k_points,
    r_grid,
    nonlocal_beta_grid,
    nonlocal_angular_momentum,
    nonlocal_d_matrix,
    vol
  )

  ext_loc = energy_local(
    reciprocal_density_grid,
    positions,
    r_vector_grid,
    g_vector_grid,
    r_grid,
    local_potential_grid,
    local_potential_charge,
    vol
  )

  v_har_reciprocal = potential.hartree_reciprocal(
    hamil_reciprocal_density_grid, g_vector_grid
  )
  v_har = jnp.fft.ifftn(v_har_reciprocal, axes=range(-3, 0))
  har = braket.expectation(wave_grid, v_har, vol, diagonal=True, mode="real")

  v_lda = potential.xc_lda(hamiltonian_density_grid)
  lda = braket.expectation(wave_grid, v_lda, vol, diagonal=True, mode="real")
  h_s = jnp.sum(har + lda)
  kin = energy.kinetic(g_vector_grid, k_points, coefficient, occupation)

  return ext_nloc + ext_loc + h_s + kin
