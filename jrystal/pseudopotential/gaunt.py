"""GPAW-compatible Gaunt coefficients."""
from __future__ import annotations

import numpy as np

from .gpaw_spherical_harmonics import YL, gam

_gaunt = {}


def gaunt(lmax: int = 2) -> np.ndarray:
  r"""Gaunt coefficients for real spherical harmonics.

  Computes coefficients G_L1L2L such that:

      Y_L1(r) Y_L2(r) = sum_L G_L1L2L Y_L(r)
  """
  if lmax in _gaunt:
    return _gaunt[lmax]

  Lmax = (lmax + 1) ** 2
  L2max = (2 * lmax + 1) ** 2
  G_LLL = np.zeros((Lmax, L2max, L2max))
  for L1 in range(Lmax):
    for L2 in range(L2max):
      for L in range(L2max):
        r = 0.0
        for c1, n1 in YL[L1]:
          for c2, n2 in YL[L2]:
            for c, n in YL[L]:
              nx = n1[0] + n2[0] + n[0]
              ny = n1[1] + n2[1] + n[1]
              nz = n1[2] + n2[2] + n[2]
              r += c * c1 * c2 * gam(nx, ny, nz)
        G_LLL[L1, L2, L] = r
  _gaunt[lmax] = G_LLL
  return G_LLL
