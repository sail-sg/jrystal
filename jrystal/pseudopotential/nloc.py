"""Nonlocal pseudopotential operators for separable projectors."""

from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from chex import dataclass
from einops import einsum
from jaxtyping import Array, Complex, Float

from .._src import braket, kinetic, potential, pw
from .._src.utils import wave_to_density
from .beta import beta_sbt_grid
from .local import energy_local, hamiltonian_local
from .spherical import batch_sph_harm_real, cartesian_to_spherical


def _compute_spherical_harmonics(
  r_vector_grid: Float[Array, "*n 3"], l_max: int
) -> Complex[Array, "l *n m"]:
  """Compute spherical harmonics for all angular momenta up to l_max.

    Return:
      Complex[Array, "l *n m"]: the first dimension is the angular momentum,
      l = 0, 1, 2, ..., l_max. The last dimension is the magnetic quantum
      number, m = -l, -l+1, ..., l. The remaining dimensions are the spatial
      dimensions of the r_vector_grid.
    """
  ndims = r_vector_grid.shape[:-1]
  r_sph = cartesian_to_spherical(r_vector_grid)
  _, r_theta, r_phi = r_sph[..., 0], r_sph[..., 1], r_sph[..., 2]

  y_lm = jnp.zeros((l_max + 1, *ndims, 2 * l_max + 1)) + 0.j
  for i in range(l_max + 1):
    y_lm = y_lm.at[i, ..., l_max - i:l_max + i +
                   1].set(batch_sph_harm_real(i, r_theta, r_phi))
  return y_lm


@dataclass(frozen=True)
class NonlocalProjectorGrid:
  """Atom-resolved reciprocal-space projector bundle."""

  projectors: Complex[Array, "atom kpt beta phi x y z"]
  d_matrices: Float[Array, "atom beta beta"]
  projector_mask: Float[Array, "atom beta"]


def _pad_nonlocal_projectors(
  projectors_by_atom: list[Complex[Array, "kpt beta phi x y z"]],
  d_matrices: list[Float[Array, "beta beta"]],
) -> NonlocalProjectorGrid:
  """Pad per-atom projector blocks to a single atom-major tensor."""
  max_beta = max(projector.shape[1] for projector in projectors_by_atom)
  padded_projectors = []
  padded_d = []
  projector_mask = []

  for projector, d_matrix in zip(projectors_by_atom, d_matrices):
    pad_beta = max_beta - projector.shape[1]
    padded_projectors.append(
      jnp.pad(
        projector,
        ((0, 0), (0, pad_beta), (0, 0), (0, 0), (0, 0), (0, 0)),
        mode="constant",
      )
    )
    padded_d.append(
      jnp.pad(jnp.asarray(d_matrix), ((0, pad_beta), (0, pad_beta)))
    )
    projector_mask.append(
      jnp.pad(
        jnp.ones(projector.shape[1], dtype=jnp.asarray(d_matrix).dtype),
        ((0, pad_beta),),
      )
    )

  return NonlocalProjectorGrid(
    projectors=jnp.stack(padded_projectors, axis=0),
    d_matrices=jnp.stack(padded_d, axis=0),
    projector_mask=jnp.stack(projector_mask, axis=0),
  )


def _projector_overlap(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  projector_grid: NonlocalProjectorGrid,
) -> Complex[Array, "spin atom kpt band beta phi"]:
  overlap = einsum(
    pw_coefficients,
    projector_grid.projectors,
    "s k band x y z, a k beta phi x y z -> s a k band beta phi",
  )
  return overlap * projector_grid.projector_mask[None, :, None, None, :, None]


def potential_nonlocal_psi_reciprocal(
  position: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  beta_gk: Optional[List[Float[Array, "kpt beta x y z"]]] = None,
  fourier_transform_method: str = 'sbt',
  concat: bool = True,
) -> NonlocalProjectorGrid:
  """
  Compute the nonlocal pseudopotential in reciprocal space.

    This function returns:

  .. math::
    < \beta_i | (G+k) >

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
    fourier_transform_method (str, optional): The method to compute the
    nonlocal pseudopotential in reciprocal space. Defaults to 'sbt'.

  """
  assert len(nonlocal_beta_grid) == len(nonlocal_angular_momentum)
  assert fourier_transform_method in ['fft', 'sbt', "numerical"]

  gk_vector_grid = jnp.expand_dims(kpts, axis=(1, 2, 3)) + jnp.expand_dims(
    g_vector_grid, 0
  )  # [nk x y z 3]

  # sbt for beta function and intepolate
  if beta_gk is None:
    beta_gk = beta_sbt_grid(
      r_grid,
      nonlocal_beta_grid,
      nonlocal_angular_momentum,
      g_vector_grid,
      kpts
    )  # a list of [kpt beta x y z]

  # `concat` is kept for API compatibility; the direct-D implementation always
  # returns atom-resolved projector bundles.
  del concat

  l_max = np.max(np.hstack(nonlocal_angular_momentum))
  y_lm = _compute_spherical_harmonics(gk_vector_grid, l_max)  # [l k x y z m]

  projectors_by_atom = []
  d_matrices = []
  for atom_position, atom_l, atom_d, atom_beta_gk in zip(
    position,
    nonlocal_angular_momentum,
    nonlocal_d_matrix,
    beta_gk,
  ):
    atom_l = jnp.asarray(atom_l, dtype=jnp.int32)
    y_lm_atom = y_lm[atom_l]  # [beta k x y z m]
    projector = einsum(
      y_lm_atom,
      atom_beta_gk,
      "beta kpt x y z m, kpt beta x y z -> kpt beta m x y z",
    )

    structure_factor = jnp.exp(-1.j * jnp.matmul(gk_vector_grid, atom_position))
    projector = projector * structure_factor[:, None, None, ...]
    projector = projector * (1.j)**atom_l[None, :, None, None, None, None]
    projectors_by_atom.append(projector * 4 * jnp.pi)
    d_matrices.append(jnp.asarray(atom_d))

  return _pad_nonlocal_projectors(projectors_by_atom, d_matrices)


def hamiltonian_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nl_psi_reciprocal: NonlocalProjectorGrid,
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  f_matrix = _projector_overlap(pw_coefficients, potential_nl_psi_reciprocal)
  df_matrix = einsum(
    potential_nl_psi_reciprocal.d_matrices,
    f_matrix,
    "a i j, s a k band j phi -> s a k band i phi",
  )
  return einsum(
    jnp.conj(f_matrix),
    df_matrix,
    "s a k b1 i phi, s a k b2 i phi -> s k b1 b2",
  ) / vol


def hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nl_psi_reciprocal: NonlocalProjectorGrid,
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
  ext_nloc = hamiltonian_nonlocal(coefficient, potential_nl_psi_reciprocal, vol)
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
  potential_nl_psi_reciprocal: NonlocalProjectorGrid,
  vol: Float,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
  kpts_weights: Optional[Float[Array, "kpt"]] = None,
) -> Float:
  f_matrix = _projector_overlap(pw_coefficients, potential_nl_psi_reciprocal)
  df_matrix = einsum(
    potential_nl_psi_reciprocal.d_matrices,
    f_matrix,
    "a i j, s a k band j phi -> s a k band i phi",
  )
  diag_hamil_nl = einsum(
    jnp.conj(f_matrix),
    df_matrix,
    "s a k band i phi, s a k band i phi -> s k band",
  ).real / vol
  if occupation is None:
    occupation = jnp.ones(diag_hamil_nl.shape, dtype=diag_hamil_nl.dtype)
  if kpts_weights is not None:
    occupation = occupation * kpts_weights[None, :, None]
  return jnp.sum(diag_hamil_nl * occupation).real


def hamiltonian_trace(
  coefficient: Complex[Array, "spin kpt band x y z"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nl_psi_reciprocal: NonlocalProjectorGrid,
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  vol: Float,
  kpts_weights: Optional[Float[Array, "kpt"]] = None,
  xc: str = 'lda_x',
  kohn_sham: bool = True
) -> Float:
  dim = kpts.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  occupation = jnp.ones(shape=wave_grid.shape[:3], dtype=wave_grid.real.dtype)
  if kpts_weights is not None:
    weighted_occupation = occupation * kpts_weights[None, :, None]
  else:
    weighted_occupation = occupation

  density = wave_to_density(wave_grid, weighted_occupation)
  reciprocal_density_grid = jnp.fft.fftn(density, axes=range(-dim, 0))

  ext_nloc = energy_nonlocal(
    coefficient,
    potential_nl_psi_reciprocal,
    vol,
    occupation,
    kpts_weights=kpts_weights,
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
  if kpts_weights is not None:
    har = har * kpts_weights[None, :, None]

  v_xc = potential.xc_density(
    hamiltonian_density_grid, g_vector_grid, kohn_sham=kohn_sham, xc_type=xc
  )
  xc_energy = braket.expectation(
    wave_grid, v_xc, vol, diagonal=True, mode="real"
  )
  if kpts_weights is not None:
    xc_energy = xc_energy * kpts_weights[None, :, None]
  h_s = jnp.sum(har + xc_energy)

  t_kin = kinetic.kinetic_operator(g_vector_grid, kpts)
  kin = braket.expectation(
    coefficient, t_kin, vol, diagonal=True, mode='kinetic'
  )
  if kpts_weights is not None:
    kin = kin * kpts_weights[None, :, None]
  kin = jnp.sum(kin)

  return (ext_nloc + ext_loc + h_s + kin).real
