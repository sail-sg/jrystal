__version__ = "0.0.1"

from . import _src
from ._src.crystal import Crystal
from . import calc

from . import pseudopotential
from . import utils
from . import sbt
from ._src import pw
from ._src import occupation
from ._src import grid
from ._src import hamiltonian
from ._src import band
from ._src import energy
from ._src import potential
from ._src import entropy
from ._src import _typing
from ._src import const
from . import config

from pathlib import Path


def get_pkg_path():
  return str(Path(__file__).parent.parent)


__all__ = [
  "_src",
  "Crystal",
  "calc",
  "pseudopotential",
  "utils",
  "sbt",
  "pw",
  "occupation",
  "grid",
  "hamiltonian",
  "band",
  "energy",
  "potential",
  "entropy",
  "_typing",
  "const",
  "get_pkg_path",
  "config",
]
