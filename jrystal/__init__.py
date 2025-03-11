__version__ = "0.0.1"

from . import _src
from . import crystal
from .crystal import Crystal
from . import calc

from . import pseudopotential
from . import utils
from . import sbt
from . import pw
from . import energy
from . import entropy
from . import occupation
from . import potential
from . import hamiltonian
from . import grid
from . import ewald

from . import config
from pathlib import Path


def get_pkg_path():
  return str(Path(__file__).parent.parent)


__all__ = [
  "_src",
  "crystal",
  "Crystal",
  "calc",
  "pseudopotential",
  "utils",
  "sbt",
  "pw",
  "occupation",
  "grid",
  "hamiltonian",
  "energy",
  "potential",
  "entropy",
  "get_pkg_path",
  "config",
  "ewald",
]
