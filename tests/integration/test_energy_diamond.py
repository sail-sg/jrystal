"""Integration test for jrystal energy minimization (diamond, CPU)."""
from __future__ import annotations

import math
import os
import re
import subprocess
from pathlib import Path

import pytest


@pytest.mark.slow
def test_energy_diamond_ci() -> None:
  repo_root = Path(__file__).resolve().parents[2]
  config_path = repo_root / "tests" / "configs" / "diamond_ci.yaml"

  env = os.environ.copy()
  env.setdefault("JAX_PLATFORMS", "cpu")
  env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
  env.setdefault("PYTHONUNBUFFERED", "1")

  result = subprocess.run(
    ["jrystal", "-m", "energy", "-c", str(config_path)],
    cwd=str(repo_root),
    env=env,
    text=True,
    capture_output=True
  )

  output = (result.stdout or "") + "\n" + (result.stderr or "")
  if result.returncode != 0:
    raise RuntimeError(
      "jrystal run failed.\n"
      f"stdout:\n{result.stdout}\n"
      f"stderr:\n{result.stderr}\n"
    )

  pattern = (
    r"Energy:\s*([-+0-9.eE]+)\|"
    r"Kinetic:\s*([-+0-9.eE]+)\|"
    r"Hartree:\s*([-+0-9.eE]+)\|"
    r"XC:\s*([-+0-9.eE]+)\|"
    r"E_zero:\s*([-+0-9.eE]+)"
  )
  match = re.findall(pattern, output)
  if not match:
    raise AssertionError(
      "Could not find energy components in jrystal output. "
      "Ensure verbose logging is enabled."
    )

  energy, kinetic, hartree, exc, e_zero = map(float, match[-1])
  expected = {
    "energy": -139.5703,
    "kinetic": 2.2003,
    "hartree": -140.7784,
    "xc": -0.9678,
    "e_zero": -0.0244
  }
  actual = {
    "energy": energy,
    "kinetic": kinetic,
    "hartree": hartree,
    "xc": exc,
    "e_zero": e_zero
  }
  tol = 5e-2
  for key, ref in expected.items():
    val = actual[key]
    if not math.isfinite(val):
      raise AssertionError(f"{key} is not finite: {val}")
    if abs(val - ref) > tol:
      raise AssertionError(
        f"{key} mismatch: {val:.4f} vs {ref:.4f} (tol {tol})"
      )

  energy = actual["energy"]
  if not math.isfinite(energy):
    raise AssertionError(f"Energy is not finite: {energy}")
  if energy >= 0:
    raise AssertionError(f"Unexpected non-negative energy: {energy}")
