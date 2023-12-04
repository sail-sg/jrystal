"""Wave function modules.

The wave function module is the core to our differentiable computation
framework. It is responsible for defining the wave function ansatz and the
corresponding integrals.

NOTE: It is advisable to keep the wave function module separate from the
crystal objects, which define the external potential of the system.
NOTE: Integrals should be defined in association with the wave function,
ideally as methods within the module.
NOTE: Wherever feasible, functional programming approaches should be employed.

"""

from jrystal._src.wave import PlaneWaveDensity, PlaneWaveFermiDirac
from jrystal._src.wave import PlaneWaveBandStructure
from jrystal._src.module import PlaneWave, QRDecomp

__all__ = (
  "PlaneWaveDensity",
  "PlaneWaveFermiDirac",
  "PlaneWaveBandStructure",
  "PlaneWave",
  "QRDecomp",
)
