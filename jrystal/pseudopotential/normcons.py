# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Norm Conserving Pseudopotential for Plane Waves. """
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum
from jaxtyping import Array, Complex, Float, Int

from .._src import braket, energy, kinetic, potential, pw, xc
from .._src.utils import wave_to_density
from .beta import beta_sbt_grid_multi_atoms
from .local import (
  _energy_local, _hamiltonian_local, energy_local, hamiltonian_local
)
from .spherical import legendre_to_sph_harm


def _potential_nonlocal_square_root(
  position: Float[Array, 'atom 3'],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]]
) -> Complex[Array, "kpt atom beta x y z phi"]:
  """
  Compute the square root of the nonlocal pseudopotential. 
  
  The Nonlocal pseudopotential hamiltonian is defined by:

  .. math::

      < C | V_\text{nl}(G, G') | C >

  where :math:`V_\text{nl} = F D F^\dagger`, where :math:`F` can be obtained from this function, and :math:`D` is the diagonal matrix of the beta functions. This function returns :math:`F`.
    
  Args:
    position (Float[Array, "atom 3"]): The positions of the atoms.
    g_vector_grid (Float[Array, "x y z 3"]): The grid of the reciprocal vectors.
    kpts (Float[Array, "kpt 3"]): The grid of the k-points.
    r_grid (List[Float[Array, "r"]]): The grid of the real space.
    nonlocal_beta_grid (List[Float[Array, "beta r"]]): The grid of the beta functions.
    nonlocal_angular_momentum (List[List[int]]): The angular momentum of the beta functions.

  Returns:
    Complex[Array, "kpt atom beta x y z phi"]: The square root of the nonlocal pseudopotential.
  """
  assert len(nonlocal_beta_grid) == len(nonlocal_angular_momentum)

  gk_vector_grid = jnp.expand_dims(
    kpts, axis=(1, 2, 3)
  ) + jnp.expand_dims(g_vector_grid, 0)  # [nk x y z 3]

  # sbt for beta function and intepolate
  beta_gk = beta_sbt_grid_multi_atoms(
    r_grid, nonlocal_beta_grid, nonlocal_angular_momentum, gk_vector_grid
  )  # shape [num_atom num_beta nk x y z]

  # kernel trick for legendre polynormials.
  kappa_all = []
  for l_atom in nonlocal_angular_momentum:
    kappa_list = []
    for ln in l_atom:
      kappa_list.append(legendre_to_sph_harm(int(ln)))
    kappa = []
    for k in kappa_list:
      kappa.append(k(gk_vector_grid))
    kappa = jnp.stack(kappa)  # shape: [num_beta nk x y z dim_phi]
    kappa_all.append(kappa)

  kappa = jnp.stack(kappa_all)  # shape: [num_atom num_beta nk x y z dim_phi]

  # structure factor
  structure_factor = jnp.exp(
    1.j * jnp.matmul(gk_vector_grid, position.transpose())
  )
  # shape: [nk x y z na]

  return einsum(
    kappa,
    structure_factor,
    beta_gk,
    "na nb nk a b c nphi, nk a b c na, na nb nk a b c -> nk na nb a b c nphi"
  ) / jnp.sqrt(2)  # factor of 1/2 is due to the conversion of unit.


def hamiltonian_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, 'atom 3'],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  pseudo_r_grid: Float[Array, "r"],
  nonlocal_beta_grid: Float[Array, "beta r"],
  nonlocal_angular_momentum: List[int],
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  potential_nl_sqrt = _potential_nonlocal_square_root(
    positions,
    g_vector_grid,
    kpts,
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
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nl_sqrt: Complex[Array, "kpt atom beta x y z phi"],
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  """
  Compute the nonlocal pseudopotential hamiltonian.
  
  .. Note::
    This function is for seperating the :math:`\sqrt{V_{nl}}` part that may not be tracable by jax.
  
  Args:
    pw_coefficients (Complex[Array, "spin kpt band x y z"]): The plane wave coefficients.
    potential_nl_sqrt (Complex[Array, "kpt atom beta x y z phi"]): The square root of the nonlocal pseudopotential.
    nonlocal_d_matrix (Float[Array, "j j"]): The diagonal matrix of the beta functions.
    vol (Float): The volume of the unit cell.
    
  Returns:
    Complex[Array, "spin kpt band band"]: The nonlocal pseudopotential hamiltonian.
  """

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
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nl_sqrt: Complex[Array, "kpt atom beta x y z phi"],
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
) -> Float:
  """
  Compute the nonlocal pseudopotential energy.
  
  .. Note::
    This function is for seperating the :math:`\sqrt{V_{nl}}` part that may not be tracable by jax.
    
  Args:
    pw_coefficients (Complex[Array, "spin kpt band x y z"]): The plane wave coefficients.
    potential_nl_sqrt (Complex[Array, "kpt atom beta x y z phi"]): The square root of the nonlocal pseudopotential.
    nonlocal_d_matrix (Float[Array, "j j"]): The diagonal matrix of the beta functions.
    vol (Float): The volume of the unit cell.
    occupation (Optional[OccupationArray]): The occupation of the states.
    
  Returns:
    Float: The nonlocal pseudopotential energy.
  """

  hamil_nl = _hamiltonian_nonlocal(
    pw_coefficients, potential_nl_sqrt, nonlocal_d_matrix, vol
  )  # shape: [spin kpoint band band]

  return jnp.sum(jax.vmap(jax.vmap(jnp.diag))(hamil_nl) * occupation).real


def energy_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  positions: Float[Array, 'atom 3'],
  g_vector_grid: Float[Array, "x y z 3"],
  k_vec: Float[Array, "kpt 3"],
  r_grid: Float[Array, "r"],
  nonlocal_beta_grid: Float[Array, "beta r"],
  nonlocal_angular_momentum: List[int],
  nonlocal_d_matrix: Float[Array, "j j"],
  vol: Float,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
) -> Float:
  """
  Compute the nonlocal pseudopotential energy.
  
  Args:
    pw_coefficients (Complex[Array, "spin kpt band x y z"]): The plane wave coefficients.
    positions (Float[Array, "atom 3"]): The positions of the atoms.
    g_vector_grid (Float[Array, "x y z 3"]): The grid of the reciprocal vectors.
    k_vec (Float[Array, "kpt 3"]): The grid of the k-points.
    r_grid (Float[Array, "r"]): The grid of the real space.
    nonlocal_beta_grid (Float[Array, "beta r"]): The grid of the beta functions.

  Returns:
    Float: The nonlocal pseudopotential energy.
    
  """
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
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nonlocal_grid_sqrt: Complex[Array, "kpt atom beta x y z phi"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "num_k 3"],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True
):
  """
  Compute the nonlocal pseudopotential hamiltonian.
  
  Args:
    coefficient (Complex[Array, "spin kpt band *ndim"]): The plane wave coefficients.
    hamiltonian_density_grid (ScalarGrid[Float, 3]): The hamiltonian density grid.
    potential_local_grid_reciprocal (Float[Array, "nr"]): The local potential grid in reciprocal space.
    potential_nonlocal_grid_sqrt (Complex[Array, "kpt atom beta x y z phi"]): The square root of the nonlocal pseudopotential.
    g_vector_grid (VectorGrid[Float, 3]): The grid of the reciprocal vectors.
    kpts (Float[Array, "num_k d"]): The grid of the k-points.
  """

  dim = kpts.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  ext_nloc = _hamiltonian_nonlocal(
    coefficient, potential_nonlocal_grid_sqrt, nonlocal_d_matrix, vol
  )
  ext_loc = _hamiltonian_local(wave_grid, potential_local_grid_reciprocal, vol)

  hamiltonian_density_grid_reciprocal = jnp.fft.fftn(
    hamiltonian_density_grid, axes=range(-dim, 0)
  )

  kin = kinetic.kinetic_operator(g_vector_grid, kpts)
  h_kin = braket.expectation(
    coefficient, kin, vol, diagonal=False, mode='kinetic'
  )

  har = potential.hartree_reciprocal(
    hamiltonian_density_grid_reciprocal, g_vector_grid, kohn_sham=kohn_sham
  )
  har = jnp.fft.ifftn(har, axes=range(-dim, 0))
  lda = xc.xc_density(hamiltonian_density_grid, kohn_sham=kohn_sham)
  v_s = har + lda
  h_s = braket.expectation(wave_grid, v_s, vol, diagonal=False, mode="real")

  return ext_nloc + ext_loc + h_s + h_kin


# TODO: define containers for pseudopotentials
def hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  r_vector_grid: Float[Array, "*nd 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "num_k 3"],
  positions: Float[Array, "num_atoms 3"],
  r_grid: Int[Array, "nr"],
  local_potential_grid: Float[Array, "nr"],
  local_potential_charge: Int[Array, "atom"],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  nonlocal_angular_momentum: List[List[int]],
  vol: Float,
  xc: str = 'lda',
  kohn_sham: bool = True
):
  dim = positions.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  ext_nloc = hamiltonian_nonlocal(
    coefficient,
    positions,
    g_vector_grid,
    kpts,
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

  kin = kinetic.kinetic_operator(g_vector_grid, kpts)
  h_kin = braket.expectation(
    coefficient, kin, vol, diagonal=False, mode='kinetic'
  )

  har = potential.hartree_reciprocal(
    hamiltonian_reciprocal_density_grid, g_vector_grid, kohn_sham=kohn_sham
  )
  har = jnp.fft.ifftn(har, axes=range(-dim, 0))
  lda = xc.xc_density(hamiltonian_density_grid, kohn_sham=kohn_sham)
  v_s = har + lda
  h_s = braket.expectation(wave_grid, v_s, vol, diagonal=False, mode="real")

  return ext_nloc + ext_loc + h_s + h_kin


def _hamiltonian_trace(
  coefficient: Complex[Array, "spin kpt band x y z"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nonlocal_grid_sqrt: Complex[Array, "kpt atom beta x y z phi"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  vol: Float,
  kohn_sham: bool = True
) -> Float:
  dim = kpts.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  occupation = jnp.ones(shape=wave_grid.shape[:3], dtype=kpts.dtype)

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
    hamiltonian_reciprocal_density_grid, g_vector_grid, kohn_sham=kohn_sham
  )
  v_har = jnp.fft.ifftn(v_har_reciprocal, axes=range(-3, 0))
  har = braket.expectation(wave_grid, v_har, vol, diagonal=True, mode="real")

  v_lda = xc.xc_density(hamiltonian_density_grid, kohn_sham=kohn_sham)
  lda = braket.expectation(wave_grid, v_lda, vol, diagonal=True, mode="real")
  h_s = jnp.sum(har + lda)

  t_kin = kinetic.kinetic_operator(g_vector_grid, kpts)
  kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=True, mode='kinetic'
  )
  kin = jnp.sum(kin)

  return (ext_nloc + ext_loc + h_s + kin).real


def hamiltonian_trace(
  coefficient: Complex[Array, "spin kpt band x y z"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  r_vector_grid: Float[Array, "x y z 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  positions: Float[Array, "atom 3"],
  r_grid: Int[Array, "r"],
  local_potential_grid: Float[Array, "r"],
  local_potential_charge: Int[Array, "atom"],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  nonlocal_angular_momentum: List[List[int]],
  vol: Float,
  kohn_sham: bool = True
):
  """
  Compute the trace of the hamiltonian.
  
  Args:
    coefficient (Complex[Array, "spin kpt band x y z"]): The plane wave coefficients.
    hamiltonian_density_grid (ScalarGrid[Float, 3]): The hamiltonian density grid.
    r_vector_grid (Float[Array, "x y z 3"]): The grid of the real space.
    g_vector_grid (Float[Array, "x y z 3"]): The grid of the reciprocal vectors.
    kpts (Float[Array, "kpt 3"]): The grid of the k-points.
    positions (Float[Array, "atom 3"]): The positions of the atoms.
    r_grid (Int[Array, "r"]): The grid of the real space.
    local_potential_grid (Float[Array, "r"]): The local potential grid.
    local_potential_charge (Int[Array, "atom"]): The local potential charge.
    nonlocal_beta_grid (List[Float[Array, "beta r"]]): The grid of the beta functions.
    nonlocal_d_matrix (List[Float[Array, "beta beta"]]): The diagonal matrix of the beta functions.
    nonlocal_angular_momentum (List[List[int]]): The angular momentum of the beta functions.
    vol (Float): The volume of the unit cell.

  Returns:
    Float: The trace of the hamiltonian.
  """
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
    positions,
    g_vector_grid,
    kpts,
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

  v_lda = xc.xc_density(hamiltonian_density_grid,)
  lda = braket.expectation(wave_grid, v_lda, vol, diagonal=True, mode="real")
  h_s = jnp.sum(har + lda)
  kin = energy.kinetic(g_vector_grid, kpts, coefficient, occupation)

  return ext_nloc + ext_loc + h_s + kin
