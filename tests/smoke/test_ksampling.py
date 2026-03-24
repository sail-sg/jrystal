"""Smoke tests for unified k-point sampling."""

import numpy as np
import jax.numpy as jnp
from absl.testing import absltest

from jrystal.calc.opt_utils import create_grids
from jrystal.calc.types import KSampling
from jrystal.config import get_config


class KSamplingTest(absltest.TestCase):

  def test_mesh_ksampling_has_weights(self):
    config = get_config()
    _, _, ksampling = create_grids(config)

    self.assertIsInstance(ksampling, KSampling)
    self.assertEqual(ksampling.mode, "mesh")
    self.assertEqual(ksampling.weights.shape[0], ksampling.kpts.shape[0])
    self.assertAlmostEqual(float(np.asarray(ksampling.weights).sum()), 1.0)

  def test_path_ksampling(self):
    ksampling = KSampling(
      mode="path",
      kpts=jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
      weights=jnp.ones(2),
      labels=["G", "X"],
    )

    self.assertEqual(ksampling.mode, "path")
    self.assertEqual(ksampling.kpts.shape, (2, 3))
    self.assertEqual(ksampling.weights.shape, (2,))
    self.assertEqual(ksampling.labels, ["G", "X"])


if __name__ == "__main__":
  absltest.main()
