"""Generate reference baselines from the public jr.calc API.

Usage:
    .venv/bin/python tests/reference/run_baseline.py
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import jax
import yaml

import jrystal as jr
from jrystal.config import JrystalConfigDict, _normalize_config

jax.config.update("jax_enable_x64", True)

REFERENCE_DIR = Path(__file__).resolve().parent


def _make_config(use_pseudopotential: bool) -> JrystalConfigDict:
  raw = {
    "system": {
      "crystal": "diamond",
      "spin": 0,
      "spin_restricted": True,
    },
    "method": {
      "xc": "lda_x",
      "use_pseudopotential": use_pseudopotential,
      "pseudopotential_type": "nc",
    },
    "basis": {
      "grid_sizes": 24,
      "cutoff_energy": 50,
    },
    "ksampling": {
      "k_grid_sizes": [1, 1, 1],
    },
    "solver": {
      "mode": "direct_opt",
      "direct_opt": {
        "max_steps": 2000,
        "optimizer": {
          "name": "adam",
          "learning_rate": 0.01,
          "b1": 0.9,
          "b2": 0.99,
        },
      },
    },
    "occupation": {
      "smearing": 0.001,
      "empty_bands": 8,
    },
    "execution": {
      "seed": 123,
      "verbose": True,
      "parallel_over_k_mesh": False,
      "parallel_over_k_path": False,
    },
  }
  return JrystalConfigDict(_normalize_config(raw))


def _num_bands(result) -> int:
  if "w_re" in result.params_pw:
    return int(result.params_pw["w_re"].shape[-1])
  return int(result.params_pw["x_re"].shape[-1])


def _serialize_energy_terms(result) -> dict[str, float]:
  terms = asdict(result.energy_terms)
  energy_terms = {
    key: round(float(value), 8)
    for key, value in terms.items()
    if abs(float(value)) > 1e-12
  }
  energy_terms["electronic_total"] = round(
    float(result.total_energy - result.energy_terms.ewald), 8,
  )
  energy_terms["total"] = round(float(result.total_energy), 8)
  return energy_terms


def run_energy_baseline(use_pseudopotential: bool) -> dict:
  config = _make_config(use_pseudopotential)
  method_label = (
    "norm-conserving pseudopotential"
    if use_pseudopotential else "all-electron"
  )

  print("=" * 60)
  print(f"Generating {method_label} baseline")
  print("=" * 60)

  start = time.time()
  result = jr.calc.energy(config)
  elapsed = time.time() - start

  baseline = {
    "system": "diamond (C2)",
    "method": method_label,
    "xc": config.method.xc,
    "grid_sizes": [24, 24, 24],
    "cutoff_energy_ha": config.basis.cutoff_energy,
    "k_grid": list(config.ksampling.k_grid_sizes),
    "num_kpts": 1,
    "num_bands": _num_bands(result),
    "optimizer": config.solver.direct_opt.optimizer.name,
    "learning_rate": config.solver.direct_opt.optimizer.learning_rate,
    "smearing": config.occupation.smearing,
    "epochs": config.solver.direct_opt.max_steps,
    "seed": config.execution.seed,
    "energy_ha": _serialize_energy_terms(result),
    "wall_time_seconds": round(elapsed, 1),
    "converged": bool(result.converged),
    "notes": (
      "Generated from jr.calc.energy() with the unified public API. "
      "Gamma-only, spin-restricted."
    ),
  }

  if use_pseudopotential and config.method.pseudopotential_file_dir is not None:
    baseline["pseudopotential_dir"] = config.method.pseudopotential_file_dir

  return baseline


def _write_yaml(path: Path, data: dict) -> None:
  with path.open("w", encoding="utf-8") as file:
    yaml.safe_dump(data, file, sort_keys=False)


def main():
  ae_baseline = run_energy_baseline(use_pseudopotential=False)
  nc_baseline = run_energy_baseline(use_pseudopotential=True)

  ae_path = REFERENCE_DIR / "baseline_ae_diamond.yaml"
  nc_path = REFERENCE_DIR / "baseline_nc_diamond.yaml"
  _write_yaml(ae_path, ae_baseline)
  _write_yaml(nc_path, nc_baseline)

  print(f"\nSaved all-electron baseline to {ae_path}")
  print(f"Saved norm-conserving baseline to {nc_path}")


if __name__ == "__main__":
  main()
