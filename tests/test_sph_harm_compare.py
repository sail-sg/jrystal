"""Compare jrystal real spherical harmonics vs GPAW solid harmonics.

This script checks that GPAW's Y(L,x,y,z) equals jrystal's real harmonics
multiplied by r^l up to a constant normalization factor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def _import_gpaw():
  try:
    from gpaw.spherical_harmonics import Yarr  # type: ignore
    return Yarr
  except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    gpaw_path = repo_root / "jrystal" / "gpaw"
    sys.path.insert(0, str(gpaw_path))
    from gpaw.spherical_harmonics import Yarr  # type: ignore
    return Yarr


def main() -> None:
  # Local imports to avoid jax dependency at module import time.
  from jrystal.pseudopotential.spherical import (  # noqa: WPS433
    batch_sph_harm_real,
    cartesian_to_spherical,
  )

  Yarr = _import_gpaw()

  rng = np.random.default_rng(0)
  r_av = rng.normal(size=(256, 3))
  r = np.linalg.norm(r_av, axis=1)
  r = np.where(r < 1e-6, 1e-6, r)

  # GPAW L -> (l, m) mapping for L=0..8.
  l_list = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2], dtype=int)
  m_list = np.array([0, -1, 0, 1, -2, -1, 0, 1, 2], dtype=int)
  L_list = np.arange(len(l_list), dtype=int)

  gpaw_vals = Yarr(L_list, r_av).astype(float)  # (L, N)

  # Jrystal real harmonics are angular; convert to solid by r^l.
  sph = cartesian_to_spherical(r_av)
  theta = np.asarray(sph[:, 1])
  phi = np.asarray(sph[:, 2])

  print("L  l  m  ratio(mean)  max_rel_dev")
  for L, l, m in zip(L_list, l_list, m_list):
    y_real = np.asarray(batch_sph_harm_real(int(l), theta, phi))
    y_lm = y_real[:, m + l]
    y_solid = y_lm * (r ** l)
    ratio = gpaw_vals[L] / y_solid
    ratio_mean = float(np.mean(ratio))
    max_rel = float(np.max(np.abs(ratio - ratio_mean)) /
                    (abs(ratio_mean) + 1e-12))
    print(f"{L:1d}  {l:1d} {m:2d}  {ratio_mean: .8e}  {max_rel: .3e}")


if __name__ == "__main__":
  main()
