import os

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from einops import einsum

import jrystal as jr
from jrystal.pseudopotential import ultrasoft as us
from jrystal.pseudopotential.dataclass import UltrasoftPseudopotential as USPP
from jrystal.pseudopotential.spherical import cartesian_to_spherical

# Configure JAX
jax.config.update("jax_enable_x64", True)


current_directory = os.getcwd()
pseudopotential_path = current_directory + "/"
crystal_path = current_directory + "/si.xyz"

# prepare variables.
crystal = jr.Crystal.create_from_file(crystal_path)
uspp_data = USPP.create(crystal, pseudopotential_path)

key = jax.random.PRNGKey(512)
num_bands = 20
grid_size = 48
kpts = jr.grid.k_vectors(crystal.cell_vectors, [1, 1, 1])
g_vecs = jr.grid.g_vectors(crystal.cell_vectors, [grid_size]*3)
r_vecs = jr.grid.r_vectors(crystal.cell_vectors, [grid_size]*3)
r_sph = cartesian_to_spherical(r_vecs)
r_theta = r_sph[..., 1]
r_phi = r_sph[..., 2]
freq_mask = jr.grid.spherical_mask(
  crystal.cell_vectors, g_vecs.shape[:3], cutoff_energy=60
)

params = jr.pw.param_init(key, num_bands, 1, freq_mask)
coeff = jr.pw.coeff(params, freq_mask)
occ_us = jr.occupation.uniform(
  1, np.sum(uspp_data.valence_charges), 0, num_bands=num_bands
)

beta_gk = us.beta_sbt_grid(
  uspp_data.r_grid, uspp_data.nonlocal_beta_grid, uspp_data.
  nonlocal_angular_momentum, g_vecs, kpts
)


# Calcualte the overlap matrix.
def compress(x, freq_mask):
  x = x.at[..., freq_mask].get()
  return x


def calculate_overlap_matrix(
  crystal, g_vecs, kpts, uspp_data, beta_gk, freq_mask
):
    """
    Calculate the nonlocal overlap matrix for ultrasoft pseudopotentials.

    Args:
        crystal: Crystal structure
        g_vecs: G-vectors
        kpts: K-points
        uspp_data: Ultrasoft pseudopotential data
        beta_gk: Beta functions in reciprocal space
        freq_mask: Frequency mask for compression

    Returns:
        jnp.ndarray: Overlap matrix with shape [k, g1, g2]
    """
    _iden = [
      jnp.eye(q.shape[0]) for q in uspp_data.nonlocal_augmentation_q_matrix
    ]

    _psi_g = us.potential_nonlocal_psi_reciprocal(
        crystal.positions, g_vecs, kpts, uspp_data.r_grid, uspp_data.
        nonlocal_beta_grid, uspp_data.nonlocal_angular_momentum, _iden,
        beta_gk, concat=False
    )

    m = [p.shape[2] for p in _psi_g]
    _psi_g = [compress(i, freq_mask) for i in _psi_g]  # k beta m g
    _psi_g = [jnp.reshape(p, (p.shape[0], -1, p.shape[-1])) for p in _psi_g]

    const = jnp.sqrt(4*jnp.pi)
    q_mat = [
        scipy.linalg.block_diag(*([q*const] * _m)) for q, _m in zip(
          uspp_data.nonlocal_augmentation_q_matrix, m
        )
    ]
    q_mat = scipy.linalg.block_diag(*q_mat)

    # Concatenate and calculate overlap
    _psi_g = jnp.concatenate(_psi_g, axis=1)  # [k atom_m_beta g]
    _overlap = einsum(
        _psi_g, q_mat, _psi_g.conj(), "k i1 g1, i1 i2, k i2 g2 -> k g1 g2"
    )

    return _overlap / crystal.vol + jnp.eye(_psi_g[0].shape[-1])


# Calculate the overlap matrix using the new function
overlap = calculate_overlap_matrix(
    crystal, g_vecs, kpts, uspp_data, beta_gk, freq_mask
)

print(jnp.linalg.eigvalsh(overlap))
