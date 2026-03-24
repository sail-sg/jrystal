"""Smoke tests for unified calc result dataclasses."""

import jax.numpy as jnp
from absl.testing import absltest

from jrystal.calc.types import (
  BandStructureResult,
  EnergyDecomposition,
  GroundStateResult,
  KSampling,
)


class ResultTest(absltest.TestCase):

  def test_ground_state_result_fields(self):
    result = GroundStateResult(
      config=object(),
      crystal=object(),
      params_pw={"a": 1},
      params_occ={"b": 2},
      total_energy=-1.23,
      energy_terms=EnergyDecomposition(kinetic=1.0, ewald=0.5),
      converged=True,
      density=jnp.ones((1, 2, 2, 2)),
      total_energy_history=[-1.0, -1.1, -1.23],
    )

    self.assertEqual(result.total_energy, -1.23)
    self.assertTrue(result.converged)
    self.assertEqual(result.energy_terms.kinetic, 1.0)
    self.assertEqual(result.density.shape, (1, 2, 2, 2))

  def test_band_structure_result_fields(self):
    kpath = KSampling(
      mode="path",
      kpts=jnp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
      weights=jnp.ones(2),
      labels=["G", "X"],
    )
    result = BandStructureResult(
      config=object(),
      crystal=object(),
      kpath=kpath,
      eigenvalues=jnp.zeros((1, 2, 4)),
      ground_state_energy=-10.0,
    )

    self.assertEqual(result.kpath.mode, "path")
    self.assertEqual(result.eigenvalues.shape, (1, 2, 4))
    self.assertEqual(result.ground_state_energy, -10.0)


if __name__ == "__main__":
  absltest.main()
