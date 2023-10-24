import jax
import jax.numpy as jnp
from absl.testing import absltest, parameterized
import numpy as np
import jrystal
from jrystal._src import QRdecomp, BatchedFFT, BatchedInverseFFT
from jrystal._src._pw import _coeff_compress, _coeff_expand
from jax.config import config
from jrystal import PlaneWave
from jrystal import Crystal

config.update("jax_enable_x64", True)

class _TestPlaneWave(parameterized.TestCase):
  
  def setUp(self):
    pkg_path = jrystal._get_pkg_path()
    diamond = Crystal(xyz_file=pkg_path+'/geometries/diamond.xyz')
    self.pw = PlaneWave(diamond, 30, 24, 1)
    print(self.pw.init(jax.random.PRNGKey(123)))
    breakpoint()
    
  def test_bloch_wave(self):
    # params = self.pw.init((jax.random.PRNGKey(123)))
    psi1 = self.pw.get_wave(self.pw.r_vec)
    psi2 = self.pw._get_wave_fft()
    np.testing.assert_almost_equal(psi1, psi2, decimal=8)

if __name__ == "__main__":
  absltest.main()