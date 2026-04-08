"""Band-structure plotting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..calc.types import BandStructureResult, KSampling
from ._style import energy_scale


def _load_from_directory(
  path: Path
) -> tuple[np.ndarray, KSampling, float | None]:
  eigenvalues = np.load(path / "band" / "eigenvalues.npy")
  with open(path / "band" / "kpath.json", "r", encoding="utf-8") as file:
    payload = json.load(file)
  kpath = KSampling(
    mode=payload["mode"],
    kpts=np.asarray(payload["kpts"]),
    weights=np.asarray(payload["weights"]),
    labels=payload.get("labels"),
    segments=payload.get("segments"),
  )
  reference_energy = None
  energy_json = path / "ground_state" / "energy.json"
  if energy_json.exists():
    with open(energy_json, "r", encoding="utf-8") as file:
      energy_payload = json.load(file)
    reference_energy = energy_payload.get("fermi_energy_ha")
  return eigenvalues, kpath, reference_energy


def _kpath_distance(kpts: np.ndarray) -> np.ndarray:
  diffs = np.diff(kpts, axis=0)
  lengths = np.linalg.norm(diffs, axis=-1)
  return np.concatenate([[0.0], np.cumsum(lengths)])


def _auto_energy_limits(values: np.ndarray) -> tuple[float, float]:
  finite = np.asarray(values[np.isfinite(values)], dtype=float)
  if finite.size == 0:
    return (-1.0, 1.0)
  ymin = float(np.min(finite))
  ymax = float(np.max(finite))
  if ymin == ymax:
    pad = max(abs(ymin) * 0.05, 1.0)
    return ymin - pad, ymax + pad
  pad = max((ymax - ymin) * 0.03, 1e-6)
  return ymin - pad, ymax + pad


def _resolve_energy_limits(
  values: np.ndarray,
  *,
  unit: str,
  reference_energy: float | None,
  energy_range,
  y_min,
  y_max,
) -> tuple[float, float]:
  if energy_range is not None:
    return tuple(energy_range)

  auto_min, auto_max = _auto_energy_limits(values)
  if y_min is None and y_max is None:
    return auto_min, auto_max

  lower = auto_min if y_min is None else float(y_min)
  upper = auto_max if y_max is None else float(y_max)
  if lower >= upper:
    raise ValueError("Band plot y-axis limits must satisfy y_min < y_max.")
  return lower, upper


def band_structure(
  source,
  reference_energy=None,
  energy_range=None,
  unit="eV",
  y_min=None,
  y_max=None,
  figsize=(8, 6),
  colors=None,
  save_path=None,
  ax=None,
):
  """Plot a band structure from a result object or a saved output directory."""
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:  # pragma: no cover - soft dependency
    raise ImportError(
      "matplotlib is required for plotting. Install with: pip install matplotlib"
    ) from exc

  if isinstance(source, BandStructureResult):
    eigenvalues = np.asarray(source.eigenvalues)
    kpath = source.kpath
    if reference_energy is None:
      reference_energy = source.reference_energy
  else:
    eigenvalues, kpath, detected_reference = _load_from_directory(Path(source))
    if reference_energy is None:
      reference_energy = detected_reference

  x = _kpath_distance(np.asarray(kpath.kpts))
  scale = energy_scale(unit)
  y = np.asarray(eigenvalues, dtype=float)
  if reference_energy is not None:
    y = y - float(reference_energy)
  y = y * scale

  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
  else:
    fig = ax.figure

  num_spin = y.shape[0]
  colors = colors or ["#1f77b4", "#ff7f0e"]
  for spin_index in range(num_spin):
    for band_index in range(y.shape[-1]):
      ax.plot(
        x,
        y[spin_index, :, band_index],
        color=colors[spin_index % len(colors)],
        linewidth=1.2,
      )

  if reference_energy is not None:
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)

  ax.set_xlabel("k-path")
  ax.set_ylabel(f"Energy ({unit})")
  ax.set_ylim(
    *_resolve_energy_limits(
      y,
      unit=unit,
      reference_energy=reference_energy,
      energy_range=energy_range,
      y_min=y_min,
      y_max=y_max,
    )
  )

  if kpath.segments:
    for _, end in kpath.segments[:-1]:
      ax.axvline(x[end], color="#999999", linestyle="--", linewidth=0.6)

  if kpath.labels and kpath.segments:
    tick_positions = [x[start] for start, _ in kpath.segments]
    tick_positions.append(x[kpath.segments[-1][1]])
    tick_labels = list(kpath.labels)
    if len(tick_labels) < len(tick_positions):
      tick_labels.append(tick_labels[-1] if tick_labels else "")
    ax.set_xticks(tick_positions[:len(tick_labels)])
    ax.set_xticklabels(tick_labels[:len(tick_positions)])

  fig.tight_layout()
  if save_path is not None:
    fig.savefig(save_path)
  return fig
