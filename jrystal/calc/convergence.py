import numpy as np

from ..config import JrystalConfigDict


def create_convergence_checker(config: JrystalConfigDict):
  return ConvergenceChecker(
    window_size=config.convergence_window_size,
    threshold=config.convergence_condition,
  )


class ConvergenceChecker:

  def __init__(self, window_size: int = 20, threshold: float = 1e-5):
    self.window_size = window_size
    self.threshold = threshold
    self.history = []

  def check(self, value: float) -> bool:
    self.history.append(value)

    if len(self.history) > self.window_size:
      self.history.pop(0)

    if len(self.history) < self.window_size:
      return False

    if np.std(self.history) < self.threshold:
      return True

    return False

  def reset(self):
    self.history = []
