"""Smoke tests for runtime-context construction."""

from absl.testing import absltest

from jrystal.calc.runtime import build_runtime_context
from jrystal.config import get_config


class RuntimeContextTest(absltest.TestCase):

  def test_build_runtime_context_ae(self):
    config = get_config()
    ctx = build_runtime_context(config)

    self.assertIsNotNone(ctx.crystal)
    self.assertEqual(ctx.ksampling.mode, "mesh")
    self.assertIsNone(ctx.pseudopotential)
    self.assertIsNotNone(ctx.freq_mask)

  def test_build_runtime_context_path(self):
    config = get_config()
    config.band.k_path_special_points = "LGXL"
    ctx = build_runtime_context(config, mode="path")

    self.assertEqual(ctx.ksampling.mode, "path")
    self.assertEqual(ctx.ksampling.kpts.shape[0], config.band.num_kpoints)
    self.assertEqual(ctx.ksampling.weights.shape[0], config.band.num_kpoints)


if __name__ == "__main__":
  absltest.main()
