import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src.module import QRDecomp, BatchedBlochWave
from jrystal._src.module import PlaneWave, _PlaneWaveFFT
from jrystal._src.module import PlaneWaveDensity
from jrystal._src.grid import r_vectors
from jax.config import config
from flax import linen as nn

config.update("jax_enable_x64", True)


class _Test_modules(parameterized.TestCase):

  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    self.nk = 1
    self.ni = 4
    grid_size = [7, 8, 9]
    self.shape = [2, self.nk, self.ni] + grid_size

    self.cg = jax.random.normal(self.key, self.shape) / 3
    self.mask = self.cg > 0.5
    self.mask = self.mask[0, 0, 0]
    self.ng = int(jnp.sum(self.mask).item())

    self.a = jnp.eye(3)
    self.k_grid = jnp.zeros([self.nk, 3])
    self.r_vec = r_vectors(self.a, grid_size)

  def test_bloch_shape(self):
    r_vec = r_vectors(self.a, self.shape[-3:])
    cg = jnp.ones(self.shape)

    bw = BatchedBlochWave(self.a, self.k_grid)
    params = bw.init(self.key, r_vec, cg)
    psi = bw.apply(params, r_vec, cg)
    np.testing.assert_array_equal(psi.shape, cg.shape)

  def test_pw_fft_equal(self):
    shape = [2, 1, self.ng, 4]
    pw = PlaneWave(shape, self.mask, self.a, self.k_grid)
    params = pw.init(self.key, self.r_vec)
    x_bw = pw.apply(params, self.r_vec)

    pw = _PlaneWaveFFT(shape, self.mask, self.k_grid)
    params = pw.init(self.key)
    x_fft = pw.apply(params)
    np.testing.assert_array_almost_equal(x_fft, x_bw, decimal=8)

  def test_pw_shape(self):

    shape = [2, 1, self.ng, 4]
    pw = PlaneWave(shape, self.mask, self.a, self.k_grid)
    params = pw.init(self.key, self.r_vec)
    x = pw.apply(params, self.r_vec)
    np.testing.assert_array_equal(x.shape, self.shape)

  def test_qr_shape(self):
    shape = [2, 1, self.ng, 4]
    qr = QRDecomp(shape, True)
    params = qr.init(self.key)
    np.testing.assert_array_equal(params['params']['w_re'].shape, shape)
    x = qr.apply(params)
    np.testing.assert_array_equal(x.shape, shape)

  @absltest.unittest.skip("deprecated modules.")
  def test_expand_compress(self):

    class Model(nn.Module):
      mask: jax.Array

      @nn.compact
      def __call__(self, x):
        return nn.Sequential(
          [CompressCoeff(self.mask), ExpandCoeff(self.mask)]
        )(
          x
        )

    model = Model(self.mask)
    params = model.init(self.key, self.cg)
    x = model.apply(params, self.cg)
    mask = x.real**2 > 1
    np.testing.assert_array_almost_equal(self.cg[mask], x.real[mask], decimal=7)

  def test_pw_density_shape(self):
    shape = [2, 1, self.ng, 4]
    pwd = PlaneWaveDensity(shape, self.mask, self.a, self.k_grid, spin=0, vol=1)
    params = pwd.init(self.key, self.r_vec)
    nr = pwd.apply(params, self.r_vec, reduce=True, method='density')
    np.testing.assert_array_equal(nr.shape, [2, 7, 8, 9])
    nr = pwd.apply(params, self.r_vec, reduce=False, method='density')
    np.testing.assert_array_equal(nr.shape, self.shape)
    nr = pwd.apply(params, jnp.arange(3))
    np.testing.assert_array_equal(nr.shape, (2,))


if __name__ == '__main__':
  absltest.main()
