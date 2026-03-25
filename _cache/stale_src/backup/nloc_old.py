"""Nonlocal Pseudopotential. """

from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum
from interpax import CubicSpline
from jaxtyping import Array, Complex, Float

from .._src import braket, kinetic, potential, pw
from .._src.grid import g2cell_vectors, g2r_vector_grid
from .._src.utils import wave_to_density
from .beta import beta_sbt_grid_multi_atoms

# from .beta import beta_sbt_grid_multi_atoms
from .local import energy_local, hamiltonian_local
from .spherical import batch_sph_harm, cartesian_to_spherical, legendre_to_sph_harm
from .utils import map_over_atoms


def _compute_spherical_harmonics(
    r_vector_grid: Float[Array, "x y z 3"],
    l_max: int
) -> Complex[Array, "l x y z 2*l_max+1"]:
    """Compute spherical harmonics for all angular momenta up to l_max."""
    nx, ny, nz = r_vector_grid.shape[:3]
    r_sph = cartesian_to_spherical(r_vector_grid)
    _, r_theta, r_phi = r_sph[..., 0], r_sph[..., 1], r_sph[..., 2]

    y_lm = jnp.zeros((l_max + 1, nx, ny, nz, 2 * l_max + 1)) + 0.j
    for i in range(l_max + 1):
        y_lm = y_lm.at[i, ..., :(2 * i + 1)].set(
            batch_sph_harm(i, r_theta, r_phi)
        )
    return y_lm


def _compute_psi_single_beta(
    r_grid: Float[Array, "r"],
    beta_grid: Float[Array, "r"],
    angmom: int,
    r_vector_grid: Float[Array, "x y z 3"],
    kpts: Float[Array, "kpt 3"],
    vol: Float,
    y_lm: Complex[Array, "l x y z 2*l_max+1"]
) -> Complex[Array, "kpt m x y z"]:
    """Compute Psi for a single beta function."""
    r_sph = cartesian_to_spherical(r_vector_grid)
    r_radius = r_sph[..., 0]
    nx, ny, nz = r_vector_grid.shape[:3]

    # Phase factor exp(-i k·r)
    exp_kr = jnp.exp(
        -1.j * einsum(kpts, r_vector_grid, "k d, x y z d -> k x y z")
    )

    # Interpolate beta function to real space grid
    cs = CubicSpline(r_grid, beta_grid)
    beta_r = cs(r_radius)[..., None]

    # Multiply by spherical harmonics
    output = beta_r * y_lm[angmom]
    output = einsum(exp_kr, output, "k x y z, x y z m -> k m x y z")

    # FFT to reciprocal space
    output = jnp.fft.fftn(output, axes=range(-3, 0))
    return output * vol / (nx * ny * nz)


def _process_single_atom(
    r_grid: Float[Array, "r"],
    beta_grid: List[Float[Array, "r"]],
    angmom: List[int],
    d_matrix: Float[Array, "beta beta"],
    position: Float[Array, "3"],
    r_vector_grid: Float[Array, "x y z 3"],
    kpts: Float[Array, "kpt 3"],
    vol: Float,
    gk_vector_grid: Float[Array, "kpt x y z 3"],
    y_lm: Complex[Array, "l x y z 2*l_max+1"]
) -> Complex[Array, "kpt beta m x y z"]:
    """Process a single atom's contribution to the nonlocal potential."""
    # Diagonalize D matrix
    eigval, eigvec = jnp.linalg.eigh(d_matrix)
    d_sqrt = eigvec * jnp.sqrt(eigval + 0.j)

    # Compute Psi for each beta function
    psi_list = [
        _compute_psi_single_beta(
            r_grid, b_grid, a_mom, r_vector_grid, kpts, vol, y_lm
        )
        for b_grid, a_mom in zip(beta_grid, angmom)
    ]

    # Stack and reshape
    psi = jnp.stack(psi_list)  # [beta kpt m x y z]
    psi = einsum(psi, "b k m x y z -> k b m x y z")

    structure_factor = jnp.exp(-1.j * jnp.matmul(gk_vector_grid, position))
    psi = einsum(
        psi, structure_factor, "k b m x y z, k x y z -> k b m x y z"
    ) / jnp.sqrt(2)   # factor of 1/2 is due to the conversion of unit.

    # Apply D matrix
    return einsum(d_sqrt, psi, "b1 b2, k b2 m x y z -> k b1 m x y z")


def potential_nonlocal_psi_fft(
    position: Float[Array, "atom 3"],
    g_vector_grid: Float[Array, "x y z 3"],
    kpts: Float[Array, "kpt 3"],
    r_grid: List[Float[Array, "r"]],
    nonlocal_beta_grid: List[Float[Array, "beta r"]],
    nonlocal_angular_momentum: List[List[int]],
    nonlocal_d_matrix: List[Float[Array, "beta beta"]],
) -> Complex[Array, "kpt beta m x y z"]:
    """
    Compute the Psi(G) tensor for nonlocal pseudopotential.

    This function computes the Psi(G) tensor such that:
    < G | v_{nl} | G' > = Ψ(G) ^ † Ψ(G)

    Args:
        position: Atomic positions [atom, 3]
        g_vector_grid: Reciprocal space grid [x, y, z, 3]
        kpts: K-points [kpt, 3]
        r_grid: Radial grids for each atom
        nonlocal_beta_grid: Beta function grids for each atom
        nonlocal_angular_momentum: Angular momentum for each beta function
        nonlocal_d_matrix: D matrices for each atom

    Returns:
        Psi tensor with shape [kpt, beta, m, x, y, z]
    """
    # Compute cell volume
    cell_vecs = g2cell_vectors(g_vector_grid)
    vol = jnp.abs(jnp.linalg.det(cell_vecs))

    # Setup grids
    gk_vector_grid = (
        jnp.expand_dims(kpts, axis=(1, 2, 3)) +
        jnp.expand_dims(g_vector_grid, 0)
    )   # [kpt x y z 3]
    r_vector_grid = g2r_vector_grid(g_vector_grid)

    # Compute spherical harmonics
    l_max = np.max(np.concatenate(nonlocal_angular_momentum))
    y_lm = _compute_spherical_harmonics(r_vector_grid, l_max)
    # [l x y z 2*l_max+1]

    # Process each atom
    @map_over_atoms
    def _beta_gk(r_grid, beta_grid, angmom, d_matrix, pos):
        return _process_single_atom(
            r_grid, beta_grid, angmom, d_matrix, pos,
            r_vector_grid, kpts, vol, gk_vector_grid, y_lm
        )

    beta_gk = _beta_gk(
        r_grid, nonlocal_beta_grid, nonlocal_angular_momentum,
        nonlocal_d_matrix, position
    )

    return jnp.concatenate(beta_gk, axis=1)


def potential_nonlocal_psi_sbt(
  position: Float[Array, "atom 3"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "kpt 3"],
  r_grid: List[Float[Array, "r"]],
  nonlocal_beta_grid: List[Float[Array, "beta r"]],
  nonlocal_angular_momentum: List[List[int]],
  nonlocal_d_matrix: List[Float[Array, "beta beta"]],
  beta_gk: Float[Array, "kpt beta x y z"] = None
) -> Complex[Array, "kpt beta m x y z"]:
  """
  Compute the Psi(G) tensor for nonlocal pseudopotential.

  This function computes the Psi(G) tensor such that:
  < G | v_{nl} | G' > = Ψ(G) ^ † Ψ(G)

  Args:
    position: Atomic positions [atom, 3]
    g_vector_grid: Reciprocal space grid [x, y, z, 3]
    kpts: K-points [kpt, 3]
    r_grid: Radial grids for each atom
    nonlocal_beta_grid: Beta function grids for each atom
    nonlocal_angular_momentum: Angular momentum for each beta function
    nonlocal_d_matrix: D matrices for each atom

  Returns:
      Psi tensor with shape [kpt, beta, m, x, y, z]
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
  return output * 2 * jnp.sqrt(jnp.pi)


def hamiltonian_nonlocal(
  pw_coefficients: Complex[Array, "spin kpt band x y z"],
  potential_nonlocal_psi: Complex[Array, "kpt beta m x y z"],
  vol: Float,
) -> Complex[Array, "spin kpt band band"]:
  coeff_psi = einsum(
    pw_coefficients,
    potential_nonlocal_psi,
    "s k band x y z, k beta m x y z -> s k band beta m"
  )

  return einsum(
    jnp.conj(coeff_psi),
    coeff_psi,
    "s k b1 beta phi, s k b2 beta phi -> s k b1 b2"
  ) / vol


def hamiltonian_matrix(
  coefficient: Complex[Array, "spin kpoint band *ndim"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nonlocal_psi: Complex[Array, "kpt beta m x y z"],
  g_vector_grid: Float[Array, "x y z 3"],
  kpts: Float[Array, "num_k 3"],
  vol: Float,
  xc: str = 'lda_x',
  kohn_sham: bool = True
) -> Complex[Array, "spin kpt band band"]:
  """
  Compute the nonlocal pseudopotential hamiltonian.

  Args:
    coefficient: The plane wave coefficients.
    hamiltonian_density_grid: The hamiltonian density grid.
    potential_local_grid_reciprocal: The local potential grid in reciprocal space.
    potential_nonlocal_psi: The nonlocal pseudopotential Psi tensor.
    g_vector_grid: The grid of the reciprocal vectors.
    kpts: The grid of the k-points.
    vol: Cell volume.
    xc: Exchange-correlation functional type.
    kohn_sham: Whether to use Kohn-Sham formalism.
  """

  dim = kpts.shape[-1]
  wave_grid = pw.wave_grid(coefficient, vol)
  ext_nloc = hamiltonian_nonlocal(
    coefficient, potential_nonlocal_psi, vol
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
  potential_nonlocal_psi: Complex[Array, "kpt beta m x y z"],
  vol: Float,
  occupation: Optional[Float[Array, "spin kpt band"]] = None,
) -> Float:
  hamil_nl = hamiltonian_nonlocal(pw_coefficients, potential_nonlocal_psi, vol)

  return jnp.sum(jax.vmap(jax.vmap(jnp.diag))(hamil_nl) * occupation).real


def hamiltonian_trace(
  coefficient: Complex[Array, "spin kpt band x y z"],
  hamiltonian_density_grid: Float[Array, "x y z"],
  potential_local_grid_reciprocal: Float[Array, "r"],
  potential_nonlocal_psi: Complex[Array, "kpt beta m x y z"],
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
    coefficient, potential_nonlocal_psi, vol, occupation
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
