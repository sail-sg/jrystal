# TEST of Nonlocal Pseudopotential
# Two ways of constructing the Psi(G) tensor, which is the key to the nonlocal # pseudopotential.

# Psi (G) = < V_nl^{1/2} | G >

# and Psi(G)^\dagger Psi(G) = <  G | V_nl | G >

import jax
import jax.numpy as jnp
import numpy as np
from einops import einsum
from interpax import CubicSpline
from jaxtyping import Array, Complex, Float

import jrystal as jr
from jrystal.pseudopotential.dataclass import UltrasoftPseudopotential as USPP

np.set_printoptions(suppress=True)
jax.config.update("jax_enable_x64", True)


jr_path = jr.get_pkg_path()
pp_path = jr_path + "/pseudopotential/normcons/"
crystal_path = jr_path + "/geometry/diamond.xyz"


crystal = jr.Crystal.create_from_file(crystal_path)
uspp_data = USPP.create(crystal, pp_path)

grid_size = [48] * 3
r_vec = jr.grid.r_vectors(crystal.cell_vectors, grid_size)


def psi_fft(
  r_grid: Float[Array, "r"],
  beta_grid: Float[Array, "r"],
  angmom: int,
  r_vecs: Float[Array, "x y z 3"],
  vol: float,
  kpts: Float[Array, "kpt 3"]
) -> Complex[Array, "kpt x y z m"]:
  """interpolate in real space and then FFT to reciprocal space """
  r_sph = jr.pseudopotential.spherical.cartesian_to_spherical(r_vecs)
  r_radius, r_theta, r_phi = r_sph[..., 0], r_sph[..., 1], r_sph[..., 2]
  nx, ny, nz = r_vecs.shape[:3]

  cs = CubicSpline(r_grid, beta_grid)
  beta_r = cs(r_radius)[..., None]

  y_lm = jr.pseudopotential.spherical.batch_sph_harm(
      angmom, r_theta, r_phi
  )
  exp_kr = jnp.exp(
    -1.j * einsum(kpts, r_vecs, "k d, x y z d -> k x y z")
  )
  output = beta_r * y_lm
  output = output[None, ...] * exp_kr[..., None]
  output = jnp.fft.fftn(output, axes=range(-4, -1))
  return output * vol / (nx * ny * nz) * (1.j) ** angmom


def psi_sbt(
  r_grid: Float[Array, "r"],
  beta_grid: Float[Array, "r"],
  angmom: int,
  r_vecs: Float[Array, "x y z 3"],
  vol: float,
  kpts: Float[Array, "kpt 3"]
) -> Complex[Array, "kpt x y z m"]:
  """Use SBT for the radial potential and then transform to reciprocal space.
  And then interpolate in reciprocal space.
  """
  g_vecs = jr.grid.r2g_vector_grid(r_vecs)
  gk_vecs = jnp.expand_dims(kpts, axis=(1, 2, 3)) + g_vecs[None, ...]
  g_sph = jr.pseudopotential.spherical.cartesian_to_spherical(gk_vecs)
  g_radius, g_theta, g_phi = g_sph[..., 0], g_sph[..., 1], g_sph[..., 2]

  gg, beta_g = jr.sbt.sbt(r_grid, beta_grid, angmom)
  cs = CubicSpline(gg, beta_g)
  beta_g = cs(g_radius)[..., None]

  y_lm = jr.pseudopotential.spherical.batch_sph_harm(
      angmom, g_theta, g_phi
  ) * 4 * jnp.pi

  return beta_g * y_lm


kpts = jr._src.band.get_k_path(crystal.cell_vectors, "LGXL", 10)

for k in kpts:
  for _l in range(1):
    beta_g1 = psi_fft(
      uspp_data.r_grid[0], uspp_data.nonlocal_beta_grid[0][0], _l, r_vec,
      crystal.vol, k[None, ...]
    )
    beta_g2 = psi_sbt(
      uspp_data.r_grid[0], uspp_data.nonlocal_beta_grid[0][0], _l, r_vec,
      crystal.vol, k[None, ...]
    )

    avg_diff = jnp.mean(jnp.abs(beta_g1 - beta_g2))
    print(f"l = {_l}, k = {k}, avg_diff = {avg_diff}")