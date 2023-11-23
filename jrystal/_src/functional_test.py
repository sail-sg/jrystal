import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized

import numpy as np
from jrystal._src.module import QRDecomp
from jrystal._src.functional import coeff_compress, coeff_expand
from jax.config import config

config.update("jax_enable_x64", True)


class _Test_pw(parameterized.TestCase):

  def setUp(self):
    shape = [2, 3, 4, 5]
    key = jax.random.PRNGKey(123)
    mask = jax.random.choice(key, a=2, shape=[3, 4, 5], replace=True)
    mask = jnp.asarray(mask, dtype=jnp.int8)
    self.mask = mask
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
    ifft = BatchedInverseFFT(3)
    fft = BatchedFFT(3)

    key = jax.random.PRNGKey(seed)
    weights = jax.random.normal(key, [10, 5, 20, 20, 20])
    p1 = ifft.init(key, weights)
    p2 = fft.init(key, weights)

    output = fft.apply(p2, ifft.apply(p1, weights))
    np.testing.assert_array_almost_equal(output, weights, decimal=10)

  # @absltest.skip("skip")
  def test_coeff_compress_expand(self):
    coeff = jnp.zeros(self.shape)
    coeff = coeff_compress(coeff, self.mask)
    coeff = coeff_expand(coeff, self.mask)
    np.testing.assert_array_equal(self.shape, coeff.shape)

  @parameterized.named_parameters(('shape', [2, 35]))
  def test_coeff_expand_compress(self, shape):
    coeff = jnp.zeros(shape)
    coeff = coeff_expand(coeff, self.mask)
    coeff = coeff_compress(coeff, self.mask)
    np.testing.assert_array_equal(shape, coeff.shape)


if __name__ == "__main__":
  absltest.main()
