# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Crystal class.

This module establishes the interface for the crystal structure. It's important
to note that all values are presented in ATOMIC UNITS inside, specifically
hartree for energy and Bohr for length. The input positions of atoms should be
in angstrom.
"""
from typing import List, Optional, Union

import ase
import jax.numpy as jnp
import numpy as np
from ase.io import read
from chex import dataclass
from jaxtyping import Array, Float

from .const import ANGSTROM2BOHR


@dataclass
class Crystal:
  r"""Crystal Structure Dataclass.

  This class encapsulates the essential attributes of a crystal, including
  atomic charges, positions, cell vectors, etc.

  A Crystal object can be created via two methods:

  1. create from specifying the four core attributes: atomic numbers (charges),
  absolute coordinates of each atom in Bohr unit (positions), cell vectors (in
  Bohr unit), and the number of unpaired electrons (spin).
  2. create from a geometry file.

  See :doc:`Create A Crystal Structure <../examples/crystal>` for more details.


  Examples:

  .. code:: python

    from jrystal import Crystal

    # Create a crystal object from a xyz file.
    crystal = Crystal.create_from_file("diamond.xyz")

    # Create a crystal object from crystal attributes.
    crystal = Crystal(
      charges=[6, 6],
      positions=[[0, 0, 0], [1.5, 1.5, 1.5]],
      cell_vectors=[[3, 0, 0], [0, 3, 0], [0, 0, 3]],
      spin=0
    )

  Args:
    charges (Optional[Float[Array, "atom"]]): The atomic charges. Defaults to
    None.
    positions (Optional[Float[Array, "atom 3"]]): The absolute coordinates of
    each atom in Bohr unit. Defaults to None.
    cell_vectors (Optional[Float[Array, '3 3']]): The cell vectors in Bohr
    unit. Defaults to None.
    spin (Optional[int]): The number of unpaired electrons. Defaults to None.
    symbol (Optional[str], optional): The atomic symbols. Defaults to None.

  """
  charges: Optional[Float[Array, "atom"]]
  positions: Optional[Float[Array, "atom 3"]]
  cell_vectors: Optional[Float[Array, '3 3']]
  spin: Optional[int] = None
  symbols: Optional[str] = None

  @property
  def scaled_positions(self):
    r"""
    The scaled coordinate of the atoms.
    """
    return self.positions @ jnp.linalg.inv(self.cell_vectors).T

  @property
  def vol(self):
    r"""
    The volume of the unit cell in Bohr^3.
    """
    return jnp.abs(jnp.linalg.det(self.cell_vectors))

  @property
  def num_atom(self):
    r"""Total number of atoms."""
    return self.positions.shape[0]

  @property
  def num_electron(self):
    r"""Total number of electrons."""
    return jnp.sum(self.charges)

  @property
  def A(self):
    r"""Alias for cell_vectors."""
    return self.cell_vectors

  @property
  def reciprocal_vectors(self):
    r"""The reciprocal cell vectors."""
    return 2 * jnp.pi * jnp.linalg.inv(self.cell_vectors).T

  @property
  def B(self):
    r"""Alias for reciprocal vectors."""
    return self.reciprocal_vectors

  @staticmethod
  def create_from_file(file_path: str, spin: Optional[int] = None):
    r"""
    Create a crystal object from a xyz file.

    Args:
      file_path (str): The path of the xyz file.
      spin (int, optional): The number of unpaired electron.
        Defaults to None. If not provided, spin is set to 1 if the total
        number of electrons is odd, otherwise 0.

    Returns:
      A Crystal onbject.

    """
    _ase_cell = read(file_path)
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

  @staticmethod
  def create_from_symbols(
    symbols: str,
    positions: Union[List[List], Float[Array, "num_atom 3"]],
    cell_vectors: Float[Array, "3 3"],
    spin: Optional[int] = None,
  ):
    r"""
    Create a crystal object from symbols, positions, and cell vectors.

    Args:
      symbols (str): The atomic symbols.
      positions (Union[List[List], Float[Array, "num_atom 3"]): The absolute
        coordinates of each atom in Bohr unit.
      cell_vectors (Float[Array, "3 3"]): The cell vectors in Bohr unit.

    Returns:
      A Crystal object.
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
