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

  match = re.findall(r"Energy:\s*([-+0-9.eE]+)", output)
  if not match:
    raise AssertionError(
      "Could not find Energy in jrystal output. "
      "Ensure verbose logging is enabled."
    )

  energy = float(match[-1])
  if not math.isfinite(energy):
    raise AssertionError(f"Energy is not finite: {energy}")
  if energy >= 0:
    raise AssertionError(f"Unexpected non-negative energy: {energy}")
