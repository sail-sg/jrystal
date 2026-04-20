"""Convergence plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..calc.types import GroundStateResult


def _load_payload(source) -> dict:
  if isinstance(source, GroundStateResult):
    columns = []
    units = []
    data = []
    if source.convergence_history:
      columns = list(source.convergence_history[0].keys())
      for column in columns:
        if column == "wall_time" or column.endswith("_s"):
          units.append("s")
        elif column in {"step", "delta_density"}:
          units.append("")
        else:
          units.append("Ha")
      data = [
        [record.get(column)
         for column in columns]
        for record in source.convergence_history
      ]
    else:
      columns = ["step", "total_energy"]
      units = ["", "Ha"]
      data = [
        [index + 1, value]
        for index, value in enumerate(source.total_energy_history)
      ]
    return {
      "solver": source.actual_solver,
      "columns": columns,
      "units": units,
      "data": data,
    }

  with open(
    Path(source) / "ground_state" / "convergence.json",
    "r",
    encoding="utf-8",
  ) as file:
    return json.load(file)


def convergence(source):
  """Plot convergence metrics from a result object or a saved output directory."""
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:  # pragma: no cover - soft dependency
    raise ImportError(
      "matplotlib is required for plotting. Install with: pip install matplotlib"
    ) from exc

  payload = _load_payload(source)
  columns = payload["columns"]
  data = np.asarray(payload["data"], dtype=object)

  fig, ax1 = plt.subplots(figsize=(8, 5))
  steps = data[:, columns.index("step")].astype(float)
  energy = data[:, columns.index("total_energy")].astype(float)
  ax1.plot(steps, energy, color="#1f77b4", label="total_energy")
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Total Energy (Ha)", color="#1f77b4")
  ax1.tick_params(axis="y", labelcolor="#1f77b4")

  ax2 = ax1.twinx()
  if "delta_energy" in columns:
    values = np.abs(data[:, columns.index("delta_energy")].astype(float))
    ax2.plot(steps, values, color="#ff7f0e", label="|dE|")
  if "delta_density" in columns:
    values = np.abs(data[:, columns.index("delta_density")].astype(float))
    ax2.plot(steps, values, color="#2ca02c", label="|dRho|")
  if "energy_std" in columns:
    values = np.abs(data[:, columns.index("energy_std")].astype(float))
    ax2.plot(steps, values, color="#d62728", label="energy_std")
  ax2.set_yscale("log")
  ax2.set_ylabel("Delta / Std")

  lines = ax1.get_lines() + ax2.get_lines()
  labels = [line.get_label() for line in lines]
  if labels:
    ax1.legend(lines, labels, loc="best")
  fig.tight_layout()
  return fig
