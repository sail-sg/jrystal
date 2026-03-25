import jax
import jax.numpy as jnp
import jrystal as jr
from jrystal._src.hamiltonian import (
  hamiltonian_matrix, hamiltonian_matrix_trace
)

from _view_holo import view_hlo

jax.config.update("jax_enable_x64", True)

key = jax.random.PRNGKey(123)
crystal = jr.Crystal(
  charges=jnp.array([6, 6]),  # Two carbon atoms
  positions=jnp.array([[0, 0, 0], [1.5, 1.5, 1.5]]),  # Positions in Bohr
  cell_vectors=jnp.array([[3, 0, 0], [0, 3, 0], [0, 0,
                                                 3]]),  # Cubic cell in Bohr
  spin=0  # No unpaired electrons
)
num_bands = crystal.num_electron
key = jax.random.PRNGKey(123)
kpts = jr.grid.k_vectors(crystal.A, [1, 1, 1])
g_vecs = jr.grid.g_vectors(crystal.A, [7, 8, 9])
freq_mask = jr.grid.cubic_mask([7, 8, 9])

params = jr.pw.param_init(key, num_bands, kpts.shape[0], freq_mask)
coeff = jr.pw.coeff(params, freq_mask)

occ = jr.occupation.gamma(kpts.shape[0], crystal.num_electron)
occ = jnp.ones(occ.shape)
density_grid = jr.pw.density_grid(coeff, crystal.vol, occ)


@view_hlo
@jax.jit
def f1(coeff):
  return hamiltonian_matrix_trace(
    coeff,
    crystal.positions,
    crystal.charges,
    density_grid,
    g_vecs,
    kpts,
    crystal.vol,
    kohn_sham=True
  )


@view_hlo
@jax.jit
def f2(coeff):
  return jr.energy.total_energy(
    coeff,
    crystal.positions,
    crystal.charges,
    g_vecs,
    kpts,
    crystal.vol,
    kohn_sham=True
  )


f1(coeff)
f2(coeff)
