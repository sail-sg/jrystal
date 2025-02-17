"""The pseudopotential module.

Warning: The pseudopotential module is currently under development and may
undergo changes in future versions. At this time, we only support the UPF
format from Quantum Espresso. Additionally, our implementation is limited to
norm-conserving pseudopotentials. Please note that the functions in this module
are not yet fully differentiable.

"""
from .dataclass import Pseudopotential, NormConservingPseudopotential
from . import local
from . import load
from . import interpolate
from . import spherical
from . import beta
from . import utils
from .normcons import ncpp

__all__ = [
  "Pseudopotential",
  "NormConservingPseudopotential",
  'local',
  'load',
  'interpolate',
  'spherical',
  'beta',
  'utils',
  'ncpp'
]
