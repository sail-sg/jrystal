"""Nonlocal Pseudopotential. """

from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum
from jaxtyping import Array, Complex, Float

from .._src import braket, kinetic, potential, pw
from .._src.utils import wave_to_density
from .beta import beta_sbt_grid
from .local import energy_local, hamiltonian_local
from .spherical import compute_spherical_harmonics_grid
from .utils import map_over_atoms


def potential_nonlocal_psi_reciprocal(
  position: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  beta_gk: Optional[List[Float[Array, "kpt beta x y z"]]] = None,
  concat: bool = True,
) -> Complex[Array, "kpt beta m x y z"]:
  """
  Compute the nonlocal pseudopotential in reciprocal space.

    This function returns:

  .. math::
    < \beta_i | (G+k) > = 4 \pi sbt(beta_i(r)) Y_lm(r)

  Args:
    position (Float[Array, "atom 3"]): The position of the atoms.
    g_vector_grid (Float[Array, "x y z 3"]): The grid of the reciprocal
    vectors.
    kpts (Float[Array, "kpt 3"]): The grid of the k-points.
    r_grid (List[Float[Array, "r"]]): The grid of the radial coordinates.
    nonlocal_beta_grid (List[Float[Array, "beta r"]]): The grid of the beta
    functions.
    nonlocal_angular_momentum (List[List[int]]): The angular momentum of the
    beta functions.
    nonlocal_d_matrix (List[Float[Array, "beta beta"]]): The matrix of the
    beta functions.
    beta_gk (Optional[Float[Array, "kpt beta x y z"]], optional): The grid of
    the beta functions. Defaults to None. This can be
    concat (bool, optional): Whether to concatenate the beta functions.
    Defaults to True. If True, the output will be concatenated along the beta
    dimension.

  Return:
    Complex[Array, "kpt beta m x y z"]: The Psi function in reciprocal space.
    The shape of the output is [kpt, beta, m, x, y, z], where m is the index of
    the magnetic momentum.

  """
  assert len(nonlocal_beta_grid) == len(nonlocal_angular_momentum)

  gk_vector_grid = jnp.expand_dims(
    kpts, axis=(1, 2, 3)
  ) + jnp.expand_dims(g_vector_grid, 0)  # [nk x y z 3]

  # sbt for beta function and intepolate
  if beta_gk is None:
    beta_gk = beta_sbt_grid(
      r_grid,
      nonlocal_beta_grid,
      nonlocal_angular_momentum,
      g_vector_grid,
      kpts,
      method="sbt",
    )  # a list of [kpt beta x y z]

  # get the spherical harmonics:
  l_max = np.max(np.hstack(nonlocal_angular_momentum))
  y_lm = compute_spherical_harmonics_grid(gk_vector_grid, l_max)
  # [l nk x y z m]

  @map_over_atoms
  def _get_psi(
    position,
    nonlocal_angular_momentum,
    nonlocal_d_matrix,
    beta_gk_single_atom
  ) -> Complex[Array, "kpt beta m x y z"]:
    y_lm_atom = y_lm[nonlocal_angular_momentum]  # [beta kpt x y z m]
    eigval, eigvec = jnp.linalg.eigh(nonlocal_d_matrix)  # shape: [beta beta]
    d_matrix_sqrt = einsum(eigvec, jnp.sqrt(eigval + 0.j), "d1 d2, d2 -> d1 d2")
    # assert jnp.min(eigval) >= 0, "The d_matrix is not positive semi-definite."
    # shape: [beta beta]
    output = einsum(
      y_lm_atom, beta_gk_single_atom, d_matrix_sqrt,
      "b1 kpt x y z m, kpt b1 x y z, b1 b2 -> kpt b2 m x y z"
    )

    structure_factor = jnp.exp(
      -1.j * jnp.matmul(gk_vector_grid, position)
    )  # shape: [kpt x y z]
    output = einsum(
      output, structure_factor,
      "kpt beta m x y z, kpt x y z -> kpt beta m x y z"
    )

    imag_factor = (1.j) ** nonlocal_angular_momentum
    # for the imaginary factor, please refer to the paper.

    output = einsum(
      output, imag_factor, "kpt beta m x y z, beta -> kpt beta m x y z"
    )
    return output * jnp.sqrt(4 * jnp.pi)
    # factor of 4 * pi is from the plane wave expansion.

  output = _get_psi(
    position, nonlocal_angular_momentum, nonlocal_d_matrix, beta_gk
  )

  if concat:
    output = jnp.concatenate(output, axis=1)

  return output


def hamiltonian_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nl_psi_reciprocal: Complex[Array, "kpt beta m x y z"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  _f_matrix = einsum(
    pw_coefficients,
    potential_nl_psi_reciprocal,
    "s k band x y z, k beta m x y z -> s k band beta m"
  )  # the factor 1/\sqrt{\omega} is omitted and handled in the next line.

  return einsum(
    jnp.conj(_f_matrix),
    _f_matrix,
    "s k b1 beta m, s k b2 beta m -> s k b1 b2"
  ) / vol  # the factor 1/vol is due to the normalization of the wavefunction.


def hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nl_psi_reciprocal: Complex[Array, "kpt atom beta x y z phi"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "num_k 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = True
) -> Complex[Array, "spin kpt band band"]:
  """

  Compute the nonlocal pseudopotential hamiltonian.

  Args:
    coefficient (Complex[Array, "spin kpt band *ndim"]): The plane wave
    coefficients.
    hamiltonian_density_grid (ScalarGrid[Float, 3]): The hamiltonian density
    grid.
    potential_local_grid_reciprocal (Float[Array, "nr"]): The local potential
    grid in reciprocal space.
    potential_nonlocal_grid_sqrt (Complex[Array, "kpt atom beta x y z phi"]):
    The square root of the nonlocal pseudopotential.
    g_vector_grid (VectorGrid[Float, 3]): The grid of the reciprocal vectors.
    kpts (Float[Array, "num_k d"]): The grid of the k-points.

  """

  dim = kpts.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  ext_nloc = hamiltonian_nonlocal(
    coefficient, potential_nl_psi_reciprocal, vol
  )
  ext_loc = hamiltonian_local(wave_grid, potential_local_grid_reciprocal, vol)

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


def energy_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nl_psi_reciprocal: Complex[Array, "kpt beta x y z phi"],
  vol: Float,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
) -> Float:
  hamil_nl = hamiltonian_nonlocal(
    pw_coefficients, potential_nl_psi_reciprocal, vol
  )

  return jnp.sum(jax.vmap(jax.vmap(jnp.diag))(hamil_nl) * occupation).real


def hamiltonian_trace(
  coefficient: Complex[Array, "spin kpt band x y z"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nl_psi_reciprocal: Complex[Array, "kpt atom beta x y z phi"],
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

  ext_nloc = energy_nonlocal(
    coefficient, potential_nl_psi_reciprocal, vol, occupation
  )
  ext_loc = energy_local(
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
