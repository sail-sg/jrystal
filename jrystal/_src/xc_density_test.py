import jax
import numpy as np
from absl.testing import absltest, parameterized
import jax_xc
from jrystal._src.energy import lda_x_raw


class _Test_xc_density(parameterized.TestCase):

  def setUp(self):
    key = jax.random.PRNGKey(123)
    self.nr = jax.random.normal(key, shape=[2, 10, 10])
    return super().setUp()

  def test_lda(self):
    lda = jax_xc.lda_x(polarized=True)
    epsilon = lda(lambda x: x, self.nr)

    _epsilon = lda_x_raw(self.nr)
    np.testing.assert_array_almost_equal(epsilon, _epsilon)


if __name__ == "__main__":
  absltest.main()
