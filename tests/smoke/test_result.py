"""Smoke tests for unified calc result dataclasses."""

from __future__ import annotations

import jax.numpy as jnp

from jrystal.calc.types import (
  BandStructureResult,
  EnergyDecomposition,
  GroundStateResult,
  KSampling,
)


def test_ground_state_result_fields():
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

  assert result.total_energy == -1.23
  assert result.converged is True
  assert result.energy_terms.kinetic == 1.0
  assert result.density.shape == (1, 2, 2, 2)


def test_band_structure_result_fields():
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

  assert result.kpath.mode == "path"
  assert result.eigenvalues.shape == (1, 2, 4)
  assert result.ground_state_energy == -10.0
