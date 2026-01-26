#!/usr/bin/env python
"""Compare PAW implementation between jrystal and GPAW.

This script uses jrystal.calc.calc_paw (jrystal-side) and GPAW's Setup
(gpaw-side) to compute the same PAW quantities and compare them.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# GPAW source (editable install)
sys.path.insert(0, "/home/aiops/zhaojx/jrystal/gpaw")

from jrystal.calc.calc_paw import setup_gpaw, calc_paw


def run_jrystal(atom: str, xc: str):
  """Run jrystal's calc_paw using its GPAW-file parser."""
  setup_data = setup_gpaw(atom, xc)
  return calc_paw(setup_data)


def run_gpaw(atom: str, xc: str):
  """Run GPAW Setup and extract comparable quantities."""
  from gpaw.setup import Setup
  from gpaw.setup_data import SetupData
  from gpaw.xc import XC

  data = SetupData(atom, xc, readxml=True)
  setup = Setup(data, XC(xc), lmax=int(max(data.l_j)), basis=None)

  results = {
    'B_ii': getattr(setup, 'B_ii', None),
    'M': getattr(setup, 'M', None),
    'M_p': getattr(setup, 'M_p', None),
    'M_pp': getattr(setup, 'M_pp', None),
    'MB': getattr(setup, 'MB', None),
    'MB_p': getattr(setup, 'MB_p', None),
    'Delta_pL': getattr(setup, 'Delta_pL', None),
    'Delta0': getattr(setup, 'Delta0', None),
    'g_lg': getattr(setup, 'g_lg', None),
    'vbar_g': data.vbar_g if hasattr(data, 'vbar_g') else None,
  }

  if hasattr(setup, 'local_corr'):
    results.update({
      'n_qg': setup.local_corr.n_qg,
      'nt_qg': setup.local_corr.nt_qg,
    })
  else:
    results.update({'n_qg': None, 'nt_qg': None})

  return results


def _to_numpy(val):
  if val is None:
    return None
  return np.asarray(val)


def _align_shapes(val_j, val_g):
  if val_j.shape == val_g.shape:
    return val_j, val_g, None
  # 1D: trim to min length
  if val_j.ndim == 1 and val_g.ndim == 1:
    n = min(val_j.shape[0], val_g.shape[0])
    return val_j[:n], val_g[:n], f"trimmed to {n}"
  # 2D: if first dim matches, trim second dim
  if val_j.ndim == 2 and val_g.ndim == 2 and val_j.shape[0] == val_g.shape[0]:
    n = min(val_j.shape[1], val_g.shape[1])
    return val_j[:, :n], val_g[:, :n], f"trimmed to {val_j.shape[0]}x{n}"
  return None, None, "shape mismatch"


def _compare_values(val_j, val_g, name):
  print(f"\n--- {name} comparison ---")

  val_j = _to_numpy(val_j)
  val_g = _to_numpy(val_g)

  if val_j is None or val_g is None:
    print(f"Missing data: jrystal={val_j is not None}, gpaw={val_g is not None}")
    return

  if val_j.shape != val_g.shape:
    val_j2, val_g2, note = _align_shapes(val_j, val_g)
    if val_j2 is None:
      print(f"Shape mismatch: jrystal {val_j.shape} vs gpaw {val_g.shape}")
      return
    val_j, val_g = val_j2, val_g2
    print(f"Shape adjusted: {val_j.shape} ({note})")
  else:
    print(f"Shape match: {val_j.shape}")

  diff = np.abs(val_j - val_g)
  print(f"Mean difference: {np.mean(diff):.6e}")
  print(f"Max difference:  {np.max(diff):.6e}")


def compare_results(results_j, results_g):
  """Compare PAW results from both implementations."""
  print("\n" + "=" * 60)
  print("Comparison Results")
  print("=" * 60)

  comparisons = [
    ('n_qg', 'Augmentation density n_qg'),
    ('nt_qg', 'Smooth augmentation density nt_qg'),
    ('Delta_pL', 'Delta_pL matrix'),
    ('Delta0', 'Delta0'),
    ('M', 'Scalar M value'),
    ('M_p', 'M_p (linear electrostatic correction)'),
    ('M_pp', 'M_pp (2nd order electrostatic correction)'),
    ('B_ii', 'Projector function overlaps B_ii'),
    ('g_lg', 'Shape functions g_lg'),
    ('vbar_g', 'Zero potential vbar_g'),
    ('MB', 'MB (core-potential integral)'),
    ('MB_p', 'MB_p (smooth core-potential)'),
  ]

  for key, display_name in comparisons:
    _compare_values(results_j.get(key), results_g.get(key), display_name)


if __name__ == "__main__":
  atom = "C"
  xc = "PBE"
  results_j = run_jrystal(atom, xc)
  results_g = run_gpaw(atom, xc)
  compare_results(results_j, results_g)
