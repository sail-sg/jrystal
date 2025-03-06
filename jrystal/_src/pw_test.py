import jax
import jax.numpy as jnp

import jrystal as jr
import numpy as np
from absl.testing import absltest, parameterized

jax.config.update("jax_enable_x64", True)


class _TestModules(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    diamond_file_path = "../geometry/diamond.xyz"
    self.crystal = jr.Crystal.create_from_file(diamond_file_path)
    self.num_bands = self.crystal.num_electron
    self.key = jax.random.PRNGKey(123)
    self.kpts = jr.grid.k_vectors(self.crystal.A, [1, 1, 1])
    self.g_vecs = jr.grid.g_vectors(self.crystal.A, [7, 8, 9])
    self.freq_mask = jr.grid.cubic_mask([7, 8, 9])

  def test_wave(self):
    pw_param = jr.pw.pw_param_init(self.key, self.num_bands, 1, self.freq_mask)
    coeff = jr.pw.pw_coeff(pw_param, self.freq_mask)
    wave_grid1 = jr.pw.wave_grid(coeff, self.crystal.vol)

    @jr.utils.vmapstack(3)
    def wave(r):
      return jr.pw.wave_r(r, coeff, self.crystal.A, self.g_vecs)

    r_vecs = jr.grid.g2r_vector_grid(self.g_vecs, self.crystal.A)
    wave_grid2 = wave(r_vecs)
    wave_grid2 = jnp.transpose(wave_grid2, (3, 4, 5, 0, 1, 2))

    np.testing.assert_allclose(wave_grid1, wave_grid2, atol=1e-8)

  def test_nabla_n_grid(self):
    pw_param = jr.pw.pw_param_init(self.key, self.num_bands, 1, self.freq_mask)
    coeff = jr.pw.pw_coeff(pw_param, self.freq_mask)
    occupation = jr.occupation.gamma(
      1, self.crystal.num_electron, num_bands=self.num_bands
    )

    r = jnp.array([0.1, 0.05, -0.6])
    nabla_density_r1 = jr.pw.nabla_density_grid(
      r, coeff, self.crystal.A, self.g_vecs, occupation
    )

    def nabla_n(r):

      def f(r):
        return jr.pw.density_r(
          r, coeff, self.crystal.A, self.g_vecs, occupation
        )

      return jax.grad(f)(r)

    nabla_density_r2 = nabla_n(r)

    np.testing.assert_allclose(nabla_density_r1, nabla_density_r2, atol=1e-8)


if __name__ == '__main__':
  absltest.main()
