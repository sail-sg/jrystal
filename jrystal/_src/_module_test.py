import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jrystal._src._module import QRdecomp, BatchedBlochWave
from jrystal._src._module import BatchedInverseFFT, BatchedFFT
from jrystal._src._module import ExpandCoeff, CompressCoeff
from jax.config import config
from flax import linen as nn

config.update("jax_enable_x64", True)

class _Test_modules(parameterized.TestCase):
  
  def setUp(self):
    self.key = jax.random.PRNGKey(123)
    self.shape = [2, 3, 4, 10, 4]
    self.x = jax.random.normal(self.key, self.shape)/3
    self.mask = self.x>0
    self.mask = self.mask[0, 0, 0]
    
  
  def test_bloch(self):
    a = jnp.eye(3)
    k_grid = jnp.zeros([1, 3])
    cg = jnp.zeros([2, 4, 5, 6])
    bw = BatchedBlochWave(a, k_grid)
    r = jnp.ones([2, 3, 4, 3])
    params = bw.init(self.key, r=r, cg=cg)
    psi = bw.apply(params, r, cg)
    np.testing.assert_array_equal(psi.shape, [2, 3, 4, 2])
  
  
  def test_qr(self):
    key = self.key
    qr = QRdecomp(self.shape, True)
    params = qr.init(key)
    np.testing.assert_array_equal(params['params']['w'].shape, self.shape)
    x = qr.apply(params)
    x = jnp.swapaxes(x, -1, -2)
    np.testing.assert_array_equal(x.shape, self.shape)
    
  
  def test_fft(self):
    class Model(nn.Module):
      ndim: int
      
      @nn.compact
      def __call__(self, x):
        return nn.Sequential(
          [
          BatchedFFT(self.ndim),
          BatchedInverseFFT(self.ndim)
          ]
        )(x)

    model = Model(3)
    params = model.init(self.key, self.x)
    x = model.apply(params, self.x)
    np.testing.assert_array_almost_equal(self.x, x.real, decimal=10)
  
  
  def test_expand_compress(self):
    class Model(nn.Module):
      mask: jax.Array
      
      @nn.compact
      def __call__(self, x):
        return nn.Sequential(
          [
            CompressCoeff(self.mask),
            ExpandCoeff(self.mask)
          ]
        )(x)
      
    model = Model(self.mask)
    params =  model.init(self.key, self.x)
    x = model.apply(params, self.x)
    mask = x.real**2 > 1
    np.testing.assert_array_almost_equal(self.x[mask], x.real[mask], decimal=10)


if __name__ == '__main__':
  absltest.main()
  