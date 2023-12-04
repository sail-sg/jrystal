import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import numpy as np
from jrystal._src.module import QRDecomp
from jrystal._src.wave_ops import coeff_compress, coeff_expand, get_mask_cubic
from jrystal._src.wave_ops import batched_fft, batched_ifft
from jax.config import config

config.update("jax_enable_x64", True)


class _Test_wave_ops(parameterized.TestCase):

  def setUp(self):
    shape = [2, 3, 4, 5]
    self.mask, self.num_g = get_mask_cubic([3, 4, 5], return_mask_num=True)
    self.shape = shape
    return super().setUp()

  @absltest.unittest.skip('')
  def test_QR(self):
    shape = [100, 100]
    qr = QRDecomp(shape)
    key = jax.random.PRNGKey(123)

    weights = jax.random.normal(key, shape)
    coeff1 = qr.apply(qr.init(key), weights)
    coeff2 = np.linalg.qr(weights)[0]
    np.testing.assert_array_equal(coeff1, coeff2)

  @parameterized.parameters((1), (2))
  def test_fft(self, seed):
    key = jax.random.PRNGKey(seed)
    weights = jax.random.normal(key, [10, 5, 20, 20, 20])
    p1 = batched_fft(weights, 3)
    p2 = batched_ifft(p1, 3)
    np.testing.assert_array_almost_equal(p2, weights, decimal=10)

  # @absltest.skip("skip")
  def test_coeff_compress_expand(self):
    coeff = jnp.zeros(self.shape)
    coeff = coeff_compress(coeff, self.mask)
    coeff = coeff_expand(coeff, self.mask)
    np.testing.assert_array_equal(self.shape, coeff.shape)

  def test_coeff_expand_compress(self):
    coeff = jnp.zeros([2, self.num_g])
    coeff = coeff_expand(coeff, self.mask)
    coeff = coeff_compress(coeff, self.mask)
    np.testing.assert_array_equal([2, self.num_g], coeff.shape)

  @parameterized.parameters((3), (10), (15), (24))
  def test_mask_cubic(self, n):
    mask, num = get_mask_cubic([n, n, n])
    np.testing.assert_array_equal(np.sum(mask), num)


if __name__ == "__main__":
  absltest.main()
