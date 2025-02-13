"""The Crystal class.

This module establishes the interface for the crystal structure. It's important
to note that all values are presented in ATOMIC UNITS inside, specifically
hartree for energy and Bohr for length. The input positions of atoms should be
in angstrom.
"""
import ase
from ase.io import read
import numpy as np
import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Optional

from chex import dataclass
from .const import ANGSTROM2BOHR


@dataclass
class Crystal:
  charges: Optional[Float[Array, "atom"]]
  positions: Optional[Float[Array, "atom 3"]]
  cell_vectors: Optional[Float[Array, '3 3']]
  spin: Optional[int] = None
  symbol: Optional[str] = None

  @property
  def scaled_positions(self):
    return self.positions @ jnp.linalg.inv(self.cell_vectors).T

  @property
  def vol(self):
    return jnp.abs(jnp.linalg.det(self.cell_vectors))

  @property
  def num_atom(self):
    return self.positions.shape[0]

  @property
  def num_electron(self):
    return jnp.sum(self.charges)

  @property
  def A(self):
    return self.cell_vectors

  @property
  def reciprocal_vectors(self):
    return 2 * jnp.pi * jnp.linalg.inv(self.cell_vectors).T

  @property
  def B(self):
    return self.reciprocal_vectors

  @staticmethod
  def create_from_xyz_file(xyz_file: str, spin: Optional[int] = None):
    """
    Create a crystal object from a xyz file.

    Args:
      xyz_file (str): The path to the xyz file.
      spin (int): The spin of the system. Defaults to None.

    Returns:
      Crystal: The crystal object.
    """
    _ase_cell = read(xyz_file)
    positions = np.array(_ase_cell.get_positions()) * ANGSTROM2BOHR
    charges = np.array(_ase_cell.get_atomic_numbers())
    cell_vectors = np.array(_ase_cell.get_cell()) * ANGSTROM2BOHR

    if spin is None:
      total_charges = jnp.sum(charges)
      spin = total_charges % 2

    return Crystal(
      charges=charges,
      positions=positions,
      cell_vectors=cell_vectors,
      spin=spin,
      symbol=_ase_cell.get_chemical_symbols()
    )

  @staticmethod
  def create_from_symbols(symbols, positions, cell_vectors):
    """
    Create a crystal object from symbols, positions, and cell vectors.

    Args:
      symbols (Array): The atomic symbols, following the ASE convention.
      positions (Array): The atomic positions.
      cell_vectors (Array): The cell vectors.

    Returns:
      Crystal: The crystal object.
    """
    _ase_cell = ase.Atoms(symbols, positions, cell=cell_vectors, pbc=True)
    positions = np.array(_ase_cell.get_positions()) * ANGSTROM2BOHR
    charges = np.array(_ase_cell.get_atomic_numbers())
    cell_vectors = np.array(_ase_cell.get_cell()) * ANGSTROM2BOHR

    if spin is None:
      total_charges = jnp.sum(charges)
      spin = total_charges % 2

    return Crystal(
      charges=charges,
      positions=positions,
      cell_vectors=cell_vectors,
      spin=spin,
      symbols=_ase_cell.get_chemical_symbols()
    )
