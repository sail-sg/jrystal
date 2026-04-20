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
"""Crystal structure container and constructors.

Internal values use atomic units (Bohr for length, Hartree for energy).
"""
from typing import List, Optional, Sequence, Union

import ase
import jax.numpy as jnp
import numpy as np
from ase.io import read
from chex import dataclass
from jaxtyping import Array, Float, Int

from .const import ANGSTROM2BOHR


@dataclass
class Crystal:
  r"""Dataclass representing a periodic crystal.

  Args:
    charges (Optional[Float[Array, "atom"]]): Atomic numbers.
    positions (Optional[Float[Array, "atom 3"]]): Atomic positions in Bohr.
    cell_vectors (Optional[Float[Array, '3 3']]): Cell vectors in Bohr.
    spin (Optional[int]): Number of unpaired electrons.
    symbols (Optional[str]): Atomic symbols.
  """
  charges: Optional[Int[Array, " atom"]]
  positions: Optional[Float[Array, "atom 3"]]
  cell_vectors: Optional[Float[Array, '3 3']]
  spin: Optional[int] = None
  symbols: Optional[List[str]] = None

  @property
  def scaled_positions(self):
    r"""Return fractional atomic coordinates in the unit cell."""
    return self.positions @ jnp.linalg.inv(self.cell_vectors).T

  @property
  def vol(self):
    r"""Return unit-cell volume in Bohr^3."""
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
    r"""Create a :class:`Crystal` from a structure file.

    Args:
      file_path (str): Path to a geometry file supported by ASE.
      spin (Optional[int]): Number of unpaired electrons. If ``None``, parity
        of the total electron count is used.

    Returns:
      Crystal: Crystal instance with coordinates converted to Bohr.
    """
    _ase_cell = read(file_path)
    return Crystal.create_from_ase_atoms(_ase_cell, spin)

  @staticmethod
  def create_from_ase_atoms(atoms: ase.Atoms, spin: Optional[int] = None):
    """Build a Crystal from an ASE Atoms object."""
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    positions = positions * ANGSTROM2BOHR
    charges = np.asarray(atoms.get_atomic_numbers(), dtype=np.int32)
    cell_vectors = np.asarray(atoms.get_cell(), dtype=np.float64)
    cell_vectors = cell_vectors * ANGSTROM2BOHR

    if spin is None:
      spin = int(np.sum(charges) % 2)
    else:
      spin = int(spin)

    return Crystal(
      charges=charges,
      positions=positions,
      cell_vectors=cell_vectors,
      spin=spin,
      symbols=atoms.get_chemical_symbols(),
    )

  @staticmethod
  def create_from_symbols(
    symbols: Union[str, Sequence[str]],
    positions: Union[List[List[float]], Float[Array, "num_atom 3"]],
    cell_vectors: Float[Array, "3 3"],
    spin: Optional[int] = None,
  ):
    r"""Create a :class:`Crystal` from symbols, positions, and cell vectors.

    Args:
      symbols (str): Atomic symbols understood by ASE.
      positions (Union[List[List], Float[Array, "num_atom 3"]]): Atomic
        positions in Angstrom.
      cell_vectors (Float[Array, "3 3"]): Cell vectors in Angstrom.
      spin (Optional[int]): Number of unpaired electrons. If ``None``, parity
        of the total electron count is used.

    Returns:
      Crystal: Crystal instance with coordinates converted to Bohr.
    """
    _ase_cell = ase.Atoms(
      symbols=symbols,
      positions=np.asarray(positions, dtype=np.float64),
      cell=np.asarray(cell_vectors, dtype=np.float64),
      pbc=True,
    )
    return Crystal.create_from_ase_atoms(_ase_cell, spin)


create_from_symbols = Crystal.create_from_symbols
create_from_file = Crystal.create_from_file
create_from_ase_atoms = Crystal.create_from_ase_atoms
