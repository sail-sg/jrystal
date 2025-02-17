"""Module for operations related to band structure optimization.

This module is a wrapper of ASE.BandPath modules.
See: https://wiki.fysik.dtu.dk/ase/ase/dft/kpoints.html

"""

from ase import cell
import numpy as np
from jaxtyping import Float, Array
from jrystal._src.const import BOHR2ANGSTROM


def get_k_path(
  cell_vectors: Float[Array, 'd d'],
  path: str,
  num: int,
  fractional: bool = False
) -> np.array:
  """Get k path vectors.

  See: https://wiki.fysik.dtu.dk/ase/ase/dft/kpoints.html

  Args:
      cell_vectors (Array): the cell vectors.
      path (str): a string of the special points in the Brillouin zone.
      num (int): the number of kpoints to be sampled.
      fractional (bool) Default: False. If True, the function will return
          fractional coordinates. If false, it returns absolute coordinate
          in 1/Bohr unit.

  Returns:
      np.array: the absolute coordinates of the k points in Bhor.
  """
  _cell = cell.Cell(cell_vectors * BOHR2ANGSTROM)
  kpts = _cell.bandpath(path, npoints=num).cartesian_kpts() * BOHR2ANGSTROM
  kpts = np.matmul(kpts, cell_vectors.T)

  if fractional:
    return kpts

  else:
    B = np.linalg.inv(cell_vectors).T * 2 * np.pi
    return kpts @ B
