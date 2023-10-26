import jax
from absl.testing import absltest, parameterized
import numpy as np
import jrystal
from jrystal._src.grid import _grid_sizes
from jax.config import config
from jrystal import Crystal
from jrystal import PlaneWave

config.update("jax_enable_x64", True)


class _TestPlaneWave(parameterized.TestCase):

  def setUp(self):
    pkg_path = jrystal._get_pkg_path()
    diamond = Crystal(xyz_file=pkg_path + '/geometries/diamond.xyz')
    self.pw = PlaneWave(diamond, 30, _grid_sizes([20, 22, 24]), _grid_sizes(1))

  def test_PW_shape(self):
    # params = self.pw.init((jax.random.PRNGKey(123)))
    key = jax.random.PRNGKey(123)
    params = self.pw.init(key)
    x = self.pw.get_wave(params)
    np.testing.assert_array_equal(x.shape, (2, 1, 12, 20, 22, 24))


if __name__ == "__main__":
  absltest.main()
