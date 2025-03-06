import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized

import jrystal as jr
from jrystal._src.hamiltonian import (
  hamiltonian_matrix, hamiltonian_matrix_trace, _hamiltonian_matrix
)


jax.config.update("jax_enable_x64", True)


class _TestHamiltonian(parameterized.TestCase):
  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    diamond_file_path = "../geometry/diamond.xyz"
    self.crystal = jr.Crystal.create_from_file(diamond_file_path)
    self.num_bands = self.crystal.num_electron
    self.key = jax.random.PRNGKey(123)
    self.kpts = jr.grid.k_vectors(self.crystal.A, [2, 2, 1])
    self.g_vecs = jr.grid.g_vectors(self.crystal.A, [7, 8, 9])
    self.freq_mask = jr.grid.cubic_mask([7, 8, 9])
    
    self.params = jr.pw.param_init(
      self.key, self.num_bands, self.kpts.shape[0], self.freq_mask
    )
    self.coeff = jr.pw.coeff(self.params, self.freq_mask)
    
    self.occ = jr.occupation.gamma(
      self.kpts.shape[0], self.crystal.num_electron
    )
    self.occ = jnp.ones(self.occ.shape)
    self.density_grid = jr.pw.density_grid(
      self.coeff, self.crystal.vol, self.occ
    )

  def test_hamiltonian_matrix(self):
    h1 = _hamiltonian_matrix(
      self.coeff, self.crystal.positions, self.crystal.charges,  self.density_grid, self.g_vecs, self.kpts, self.crystal.vol, kohn_sham=True)

    h2 = hamiltonian_matrix(
      self.coeff, self.crystal.positions, self.crystal.charges, self.density_grid, self.g_vecs, self.kpts, self.crystal.vol, kohn_sham=True
    )

    np.testing.assert_allclose(h1, h2, atol=1e-7)

  def test_hamiltonian_matrix_trace(self):
    e1 = hamiltonian_matrix_trace(
      self.coeff, self.crystal.positions, self.crystal.charges, self.density_grid, self.g_vecs, self.kpts, self.crystal.vol, kohn_sham=True
    ).real
    e2 = jr.energy.total_energy(
      self.coeff, self.crystal.positions, self.crystal.charges, self.g_vecs, self.kpts, self.crystal.vol, kohn_sham=True
    )
    e2 = jnp.sum(e2)

    print(e1)
    print(e2)
    np.testing.assert_allclose(e1, e2, atol=1e-7)


if __name__ == '__main__':
  absltest.main()
