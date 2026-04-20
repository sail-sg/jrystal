"""Lightweight opt-in phase timing helpers."""

from __future__ import annotations

import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

from ..terminal_ui import get_output_lock


@dataclass
class _PhaseStats:
  calls: int = 0
  total_s: float = 0.0


class PhaseTimer:
  """Thread-safe wall-clock timer for coarse workflow phases."""

  def __init__(self, *, enabled: bool):
    self.enabled = bool(enabled)
    self._lock: RLock = get_output_lock()
    self._phases: "OrderedDict[str, _PhaseStats]" = OrderedDict()

  def record(self, name: str, duration_s: float) -> None:
    """Record one completed phase duration."""
    if not self.enabled:
      return
    with self._lock:
      stats = self._phases.setdefault(name, _PhaseStats())
      stats.calls += 1
      stats.total_s += float(duration_s)

  @contextmanager
  def phase(self, name: str) -> Iterator[None]:
    """Context manager for timing a phase."""
    if not self.enabled:
      yield
      return
    start = time.perf_counter()
    try:
      yield
    finally:
      self.record(name, time.perf_counter() - start)

  def summary(self, *, total_wall_time: float) -> dict[str, dict[str, float]]:
    """Return formatted timing statistics keyed by phase name."""
    if not self.enabled:
      return {}
    total_wall_time = max(float(total_wall_time), 1e-12)
    with self._lock:
      return {
        name: {
          "calls": stats.calls,
          "total_s": stats.total_s,
          "avg_s": (stats.total_s / stats.calls) if stats.calls else 0.0,
          "pct_total": 100.0 * stats.total_s / total_wall_time,
        }
        for name, stats in self._phases.items()
      }


__all__ = ["PhaseTimer"]
