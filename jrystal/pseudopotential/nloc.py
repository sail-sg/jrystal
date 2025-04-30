"""Nonlocal Pseudopotential. """

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import List, Optional
from jaxtyping import Float, Array, Complex, Int
from einops import einsum

from .._src import braket, kinetic
from .._src import potential, energy
from .._src.utils import wave_to_density
from .._src import pw
from .spherical import legendre_to_sph_harm
from .local import _hamiltonian_local, _energy_local
from .beta import beta_sbt_grid_multi_atoms


def _potential_nonlocal_square_root(
  position: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  beta_gk: Float[Array, "kpt beta x y z"] = None
) -> Complex[Array, "kpt beta phi x y z"]:
  """
  Compute the square root of the nonlocal pseudopotential.

  This function returns:

  .. math::

    < \beta_i | (G+k) >
  """
  assert len(nonlocal_beta_grid) == len(nonlocal_angular_momentum)

  gk_vector_grid = jnp.expand_dims(
    kpts, axis=(1, 2, 3)
  ) + jnp.expand_dims(g_vector_grid, 0)  # [nk x y z 3]

  # sbt for beta function and intepolate
  if beta_gk is None:
    beta_gk = beta_sbt_grid_multi_atoms(
      r_grid,
      nonlocal_beta_grid,
      nonlocal_angular_momentum,
      g_vector_grid,
      kpts
    )  # [kpt beta x y z]

  assert beta_gk.shape[0] == kpts.shape[0]

  output = []
  l_max = np.max(np.hstack(nonlocal_angular_momentum))

  for i in range(position.shape[0]):
    angmom = nonlocal_angular_momentum[i]
    d_matrix = nonlocal_d_matrix[i]
    # assert jnp.allclose(d_matrix, d_matrix.T)
    eigval, eigvec = jnp.linalg.eigh(d_matrix)  # shape: [beta beta]
    d_matrix_sqrt = eigvec * jnp.sqrt(eigval + 0.j)  # shape: [beta beta]
    structure_factor = jnp.exp(
      -1.j * jnp.matmul(gk_vector_grid, position[i])
    )  # shape: [nk x y z]
    kappa_list = []
    for ln in angmom:
      kappa_list.append(legendre_to_sph_harm(int(ln), int(l_max)))
    kappa = []
    for k in kappa_list:
      kappa.append(k(gk_vector_grid))
    kappa = jnp.stack(kappa)  # shape: [beta nk x y z phi]
    kappa = einsum(
      d_matrix_sqrt,
      kappa,
      structure_factor,
      "b1 b2, b2 k x y z phi, k x y z -> k b1 phi x y z"
    ) / jnp.sqrt(2)  # factor of 1/2 is due to the conversion of unit.

    output.append(kappa)

  output = jnp.concatenate(output, axis=1)
  output = einsum(
    output, beta_gk, "k beta phi x y z, k beta x y z -> k beta phi x y z"
  )
  return output


def _hamiltonian_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nl_sqrt: Complex[Array, "kpt beta x y z phi"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  _f_matrix = einsum(
    pw_coefficients,
    potential_nl_sqrt,
    "s k band x y z, k beta phi x y z -> s k band beta phi"
  )

  return 4 * jnp.pi / vol * einsum(
    jnp.conj(_f_matrix),
    _f_matrix,
    "s k b1 beta phi, s k b2 beta phi -> s k b1 b2"
  )


def _hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nonlocal_grid_sqrt: Complex[Array, "kpt atom beta x y z phi"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "num_k 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = True
) -> Complex[Array, "spin kpt band band"]:
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
    coefficient, potential_nonlocal_grid_sqrt, vol
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
  v_xc = potential.xc_density(
    hamiltonian_density_grid, g_vector_grid, kohn_sham=kohn_sham, xc_type=xc
  )
  v_s = har + v_xc
  h_s = braket.expectation(wave_grid, v_s, vol, diagonal=False, mode="real")

  return ext_nloc + ext_loc + h_s + h_kin


def _energy_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nl_sqrt: Complex[Array, "kpt beta x y z phi"],
  vol: Float,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
) -> Float:
  hamil_nl = _hamiltonian_nonlocal(pw_coefficients, potential_nl_sqrt, vol)

  return jnp.sum(jax.vmap(jax.vmap(jnp.diag))(hamil_nl) * occupation).real


def _hamiltonian_trace(
  coefficient: Complex[Array, "spin kpt band x y z"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nonlocal_grid_sqrt: Complex[Array, "kpt atom beta x y z phi"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = True
) -> Float:
  dim = kpts.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  occupation = jnp.ones(shape=wave_grid.shape[:3], dtype=kpts.dtype)

  density = wave_to_density(wave_grid, occupation)
  reciprocal_density_grid = jnp.fft.fftn(density, axes=range(-dim, 0))

  ext_nloc = _energy_nonlocal(
    coefficient, potential_nonlocal_grid_sqrt, vol, occupation
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

  v_xc = potential.xc_density(
    hamiltonian_density_grid, g_vector_grid, kohn_sham=kohn_sham, xc_type=xc
  )
  xc_energy = braket.expectation(
    wave_grid, v_xc, vol, diagonal=True, mode="real"
  )
  h_s = jnp.sum(har + xc_energy)

  t_kin = kinetic.kinetic_operator(g_vector_grid, kpts)
  kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=True, mode='kinetic'
  )
  kin = jnp.sum(kin)

  return (ext_nloc + ext_loc + h_s + kin).real
