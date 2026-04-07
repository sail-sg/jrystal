from __future__ import annotations

import numpy as np
import pytest

import jrystal as jr
from jrystal.plot._style import HARTREE_TO_EV


def _band_result():
  config = jr.config.get_config(None)
  return jr.calc.types.BandStructureResult(
    config=config,
    crystal=type("Crystal", (), {"symbols": ["Si", "Si"]})(),
    kpath=jr.calc.types.KSampling(
      mode="path",
      kpts=np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
        dtype=float,
      ),
      weights=np.ones(3, dtype=float),
      labels=["G", "X", "M"],
      segments=[(0, 1), (1, 2)],
    ),
    eigenvalues=np.asarray(
      [
        [
          [-18.0 / HARTREE_TO_EV, -1.0 / HARTREE_TO_EV, 2.0 / HARTREE_TO_EV],
          [-17.5 / HARTREE_TO_EV, -0.6 / HARTREE_TO_EV, 2.4 / HARTREE_TO_EV],
          [-17.0 / HARTREE_TO_EV, -0.2 / HARTREE_TO_EV, 2.8 / HARTREE_TO_EV],
        ]
      ],
      dtype=float,
    ),
    reference_energy=0.0,
  )


def test_band_plot_uses_frontier_window_by_default():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")

  fig = jr.plot.band_structure(_band_result())
  ymin, ymax = fig.axes[0].get_ylim()

  assert ymin > -12.0
  assert ymax < 8.5
  assert ymax > 2.5


def test_band_plot_respects_explicit_limits():
  matplotlib = pytest.importorskip("matplotlib")
  matplotlib.use("Agg")

  fig = jr.plot.band_structure(
    _band_result(),
    y_min=-4.0,
    y_max=3.0,
  )
  ymin, ymax = fig.axes[0].get_ylim()

  assert ymin == pytest.approx(-4.0)
  assert ymax == pytest.approx(3.0)
