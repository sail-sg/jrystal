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

from jrystal.const import ANGSTROM2BOHR


@chex.dataclass
class Crystal:
  """Crystal object.
  
  The crystal object represents the structure of a solid-state crystal, 
  characterized by its periodic repetition. It encompasses all the details 
  regarding the atoms and cells. It's a wrapper of the ASE
  (https://wiki.fysik.dtu.dk/ase/).
  
  Attributes:
  
    symbols: str = None
    scaled_positions: Float[Array, 'natom nd'] = None
    positions: Float[Array, 'natom nd'] = None 
    charges: Int[Array, 'natom'] = None
    spin: int = None 
    xyz_file: str = None
    lattice_vectors: Float[Array, 'nd nd'] = None
    reciprocal_vectors: Float[Array, 'nd nd'] = None
    A: Float[Array, 'nd nd'] = None  # alias for lattice_vectors
    B: Float[Array, 'nd nd'] = None  # alias for reciprocal_vectors
    vol: float = None 
    natom: int = None
    nelec: int = None 
    _ase_cell: ase.Atoms = None
  
  """

  symbols: str = None
  positions: Union[List[List], Float[ArrayLike, 'natom nd']] = None
  lattice_vectors: Union[List[List], Float[ArrayLike, 'ndim nd']] = None
  xyz_file: str = None
  spin = None

  def __post_init__(self):
    if (
      self.symbols and (self.positions is not None) and
      (self.lattice_vectors is not None)
    ):
      self._ase_cell = ase.Atoms(
        self.symbols, self.positions, cell=self.lattice_vectors, pbc=True
      )

    elif self.xyz_file:
      self._ase_cell = read(self.xyz_file)
      self.xyz_file = self.xyz_file

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

    self.spin = self.nelec % 2 if self.spin is None else self.spin

  def save_xyz_file(self, path=None):
    path = path if path else str(self.cell.symbols) + '.xyz'
    self._ase_cell.write(path)
