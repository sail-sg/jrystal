__version__ = '0.0.2'

from . import crystal
from . import occupation
from . import total_energy
from . import wave
from . import band_structure
from . import visualization
from . import config
from . import energy
from . import training_utils
from . import utils

from pathlib import Path


def get_pkg_path():
  return str(Path(__file__).parent.parent)


__path__ = get_pkg_path()

__all__ = [
  "config",
  "crystal",
  "energy",
  "occupation",
  "total_energy",
  "training_utils",
  "utils",
  "wave",
  "band_structure",
  "visualization",
]
