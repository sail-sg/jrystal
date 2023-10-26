"""The Crystal class.

This module establishes the interface for the crystal structure. It's important 
to note that all values are presented in ATOMIC UNITS inside, specifically 
hartree for energy and Bohr for length. The input positions of atoms should be 
in angstrom.
"""
import ase
from ase.io import read
import jax.numpy as jnp
from jaxtyping import Float, Int, Array, ArrayLike
from typing import Union, List
import chex

from jrystal._src.const import ANGSTROM2BOHR


@chex.dataclass
class Crystal:
  """Crystal object.
  
  The crystal object represents the structure of a solid-state crystal, 
  characterized by its periodic repetition. It encompasses all the details 
  regarding the atoms and cells. It's a wrapper of the ASE
  (https://wiki.fysik.dtu.dk/ase/).
  
  
  """
  symbols: str
  scaled_positions: Float[Array, 'natom nd']
  positions: Float[Array, 'natom nd']

  charges: Int[Array, 'natom']
  spin: int

  xyz_file: str
  lattice_vectors: Float[Array, 'nd nd']
  reciprocal_vectors: Float[Array, 'nd nd']
  A: Float[Array, 'nd nd']  # alias for lattice_vectors
  B: Float[Array, 'nd nd']  # alias for reciprocal_vectors
  vol: float

  natom: int
  nelec: int

  _ase_cell: ase.Atoms

  def __init__(
    self,
    symbols: str = None,
    positions: Union[List[List], Float[ArrayLike, 'natom nd']] = None,
    lattice_vectors: Union[List[List], Float[ArrayLike, 'ndim nd']] = None,
    xyz_file: str = None,
    spin=None,
  ):

    if symbols and (positions is not None) and (lattice_vectors is not None):
      self._ase_cell = ase.Atoms(
        symbols, positions, cell=lattice_vectors, pbc=True
      )

    elif xyz_file:
      self._ase_cell = read(xyz_file)
      self.xyz_file = xyz_file

    else:
      raise ValueError(
        'Please input complete crystal parameters or a .xyz file'
      )

    self.symbols = self._ase_cell.symbols
    self.positions = jnp.array(self._ase_cell.get_positions()) * ANGSTROM2BOHR
    self.scaled_positions = jnp.array(self._ase_cell.get_scaled_positions())
    self.charges = jnp.array(self._ase_cell.get_atomic_numbers())

    self.lattice_vectors = self._ase_cell.get_cell() * ANGSTROM2BOHR
    self.A = self.lattice_vectors
    self.reciprocal_vectors = jnp.linalg.inv(self.A).T * 2 * jnp.pi
    self.B = self.reciprocal_vectors

    self.vol = self._ase_cell.get_volume() * ANGSTROM2BOHR**3
    self.natom = self._ase_cell.get_global_number_of_atoms()
    self.nelec = jnp.sum(self.charges)

    self.spin = self.nelec % 2 if spin is None else spin

  def save_xyz_file(self, path=None):
    path = path if path else str(self.cell.symbols) + '.xyz'
    self._ase_cell.write(path)
