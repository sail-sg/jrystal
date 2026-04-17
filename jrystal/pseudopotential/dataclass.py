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

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from jaxtyping import Array, Float, Int

from .._src.crystal import Crystal
from .kernel import (
  AtomSpeciesMap,
  PseudoSpeciesSetup,
  load_species_setups,
)


def _expand_species_field(
  species_setups: tuple[PseudoSpeciesSetup, ...],
  atom_species_map: AtomSpeciesMap,
  getter,
) -> list:
  return [getter(species_setups[int(idx)]) for idx in atom_species_map.species_index]


@dataclass
class Pseudopotential():
  """
    Pseudopotential container format.
  """
  num_atom: int
  positions: Float[Array, "atom 3"]
  charges: Int[Array, "atom"]
  atomic_symbols: List[str]
  valence_charges: List[float]
  species_setups: tuple[PseudoSpeciesSetup, ...]
  atom_species_map: Optional[AtomSpeciesMap]

  @staticmethod
  def create(
    crystal: Crystal,
    dir: Union[str, None] = None,
  ):
    # create from a crystal object
    pass


@dataclass
class NormConservingPseudopotential(Pseudopotential):
  """
    Norm Conserving Pseudopotential Container.

    Attributes:
      num_atom (int): Number of atoms.
      positions (np.ndarray): Atom positions.
      charges (np.ndarray): Atom charges.
      atomic_symbols (List[str]): Atomic symbols.
      valence_charges (List[float]): Valence charges.
      r_grid (List[np.ndarray]): r grid.
      r_cutoff (List[float]): r cutoff.
      local_potential_grid (List[np.ndarray]): Local potential grid.
      local_potential_charge (List[float]): Local potential charge.
      num_beta (List[int]): Number of beta functions.
      nonlocal_beta_grid (List[np.ndarray]): Nonlocal beta grid.
      nonlocal_beta_cutoff_radius (List[List[float]]): Nonlocal beta cutoff
        radius.
      nonlocal_d_matrix (List[np.ndarray]): Nonlocal d matrix.
      nonlocal_angular_momentum (List[List[int]]): Nonlocal angular momentum.
      nonlocal_valence_configuration (List[List[dict]]): Nonlocal valence
        configuration.

    Warning:
      Unlike the original code in Quantum Espresso where the beta functions are
      multiplied by r. In our implementation, the beta functions are the
      original beta functions (dual basis for pseudo wave function) as defined
      in the literature.

  """

  r_grid: List[Float[Array, "num_r"]]
  r_ab: List[Float[Array, "num_r"]]
  r_cutoff: List[float]
  l_max: List[int]
  l_max_rho: List[Optional[int]]
  local_potential_grid: List[Float[Array, "num_r"]]
  local_potential_charge: List[float]
  num_beta: List[int]
  nonlocal_beta_grid: List[Float[Array, "num_beta num_r"]]
  nonlocal_beta_cutoff_radius: List[List[float]]
  nonlocal_d_matrix: List[Float[Array, "num_beta num_beta"]]
  nonlocal_angular_momentum: List[List[int]]
  nonlocal_valence_configuration: List[List[dict]]

  @staticmethod
  def create(
    crystal: Crystal,
    dir: Union[str, None] = None,
  ):
    positions = crystal.positions
    charges = crystal.charges
    atomic_symbols = crystal.symbols
    num_atom = len(charges)
    species_setups, atom_species_map = load_species_setups(crystal, dir, "nc")

    valence_charges = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.valence_charge,
    )
    r_grid = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.radial.r_g,
    )
    r_ab = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.radial.dr_g,
    )
    r_cutoff = [None] * num_atom
    l_max = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.l_max,
    )
    l_max_rho = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.l_max_rho,
    )
    local_potential_grid = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.local.vloc_r,
    )
    local_potential_charge = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.local.z_valence,
    )
    nonlocal_num_beta = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: int(setup.projectors.beta_jr.shape[0]),
    )
    nonlocal_beta_grid = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.projectors.beta_jr,
    )
    nonlocal_beta_cutoff_radius = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: list(setup.projectors.cutoff_radii),
    )
    nonlocal_d_matrix = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.projectors.d_jj,
    )
    nonlocal_angular_momentum = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.projectors.l_j,
    )
    nonlocal_valence_configuration = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: list(setup.valence_configuration),
    )

    return NormConservingPseudopotential(
      num_atom,
      positions,
      charges,
      atomic_symbols,
      valence_charges,
      species_setups,
      atom_species_map,
      r_grid,
      r_ab,
      r_cutoff,
      l_max,
      l_max_rho,
      local_potential_grid,
      local_potential_charge,
      nonlocal_num_beta,
      nonlocal_beta_grid,
      nonlocal_beta_cutoff_radius,
      nonlocal_d_matrix,
      nonlocal_angular_momentum,
      nonlocal_valence_configuration
    )


@dataclass
class UltrasoftPseudopotential(NormConservingPseudopotential):
  """Ultrasoft Pseudopotential Container.

    Attributes:
      num_atom (int): Number of atoms.
      positions (np.ndarray): Atom positions.
      charges (np.ndarray): Atom charges.
      atomic_symbols (List[str]): Atomic symbols.
      valence_charges (List[float]): Valence charges.
      r_grid (List[np.ndarray]): r grid.
      r_cutoff (List[float]): r cutoff.
      local_potential_grid (List[np.ndarray]): Local potential grid.
      local_potential_charge (List[float]): Local potential charge.
      num_beta (List[int]): Number of beta functions.
      nonlocal_beta_grid (List[np.ndarray]): Nonlocal beta grid.
      nonlocal_beta_cutoff_radius (List[List[float]]): Nonlocal beta cutoff
        radius.
      nonlocal_d_matrix (List[np.ndarray]): Nonlocal d matrix.
      nonlocal_angular_momentum (List[List[int]]): Nonlocal angular momentum.
      nonlocal_valence_configuration (List[List[dict]]): Nonlocal valence
        configuration.

      nonlocal_q_matrix (List[np.ndarray]): Nonlocal q matrix.
      nonlocal_augmentation_qij (List[np.ndarray]): Nonlocal augmentation
        charge.

      Warning: the shape of nonlocal_augmentation_qij depends on the value of
      `q_with_l` in the UPF file.

      If `q_with_l` is .True., `nonlocal_augmentation_qij` is an array of
      shape (num_q, num_q, l_max) where `num_q` is the number of augmentation
      functions and `l_max` is the maximum angular momentum.

      If `q_with_l` is .False., the `nonlocal_augmentation_qij` is an array of
      shape (num_q, num_q, 1) where `num_q` is the number of augmentation
      functions.
  """
  nonlocal_augmentation_q_matrix: List[np.ndarray]
  nonlocal_augmentation_qij: List[np.ndarray]
  nonlocal_augmentation_q_with_l: List[bool]

  @staticmethod
  def create(
    crystal: Crystal,
    dir: Union[str, None] = None,
  ):
    positions = crystal.positions
    charges = crystal.charges
    atomic_symbols = crystal.symbols
    num_atom = len(charges)
    species_setups, atom_species_map = load_species_setups(crystal, dir, "us")

    valence_charges = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.valence_charge,
    )
    r_grid = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.radial.r_g,
    )
    r_ab = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.radial.dr_g,
    )
    r_cutoff = [None] * num_atom
    l_max = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.l_max,
    )
    l_max_rho = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.l_max_rho,
    )
    local_potential_grid = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.local.vloc_r,
    )
    local_potential_charge = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.local.z_valence,
    )
    nonlocal_num_beta = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: int(setup.projectors.beta_jr.shape[0]),
    )
    nonlocal_beta_grid = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.projectors.beta_jr,
    )
    nonlocal_beta_cutoff_radius = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: list(setup.projectors.cutoff_radii),
    )
    nonlocal_d_matrix = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.projectors.d_jj,
    )
    nonlocal_angular_momentum = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: setup.projectors.l_j,
    )
    nonlocal_valence_configuration = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: list(setup.valence_configuration),
    )

    def _q_matrix_or_default(setup):
      num_beta = int(setup.projectors.beta_jr.shape[0])
      if setup.augmentation is None:
        return np.zeros((num_beta, num_beta), dtype=np.float64)
      return np.asarray(setup.augmentation.q_jj, dtype=np.float64)

    def _qij_or_default(setup):
      num_beta = int(setup.projectors.beta_jr.shape[0])
      num_r = int(setup.radial.r_g.shape[0])
      if setup.augmentation is None:
        return np.zeros((num_beta, num_beta, 1, num_r), dtype=np.float64)
      return np.asarray(setup.augmentation.q_jjlr, dtype=np.float64)

    nonlocal_augmentation_q_matrix = _expand_species_field(
      species_setups,
      atom_species_map,
      _q_matrix_or_default,
    )
    nonlocal_augmentation_qij = _expand_species_field(
      species_setups,
      atom_species_map,
      _qij_or_default,
    )
    nonlocal_augmentation_q_with_l = _expand_species_field(
      species_setups,
      atom_species_map,
      lambda setup: bool(
        setup.augmentation.q_with_l if setup.augmentation is not None else False
      ),
    )

    return UltrasoftPseudopotential(
      num_atom,
      positions,
      charges,
      atomic_symbols,
      valence_charges,
      species_setups,
      atom_species_map,
      r_grid,
      r_ab,
      r_cutoff,
      l_max,
      l_max_rho,
      local_potential_grid,
      local_potential_charge,
      nonlocal_num_beta,
      nonlocal_beta_grid,
      nonlocal_beta_cutoff_radius,
      nonlocal_d_matrix,
      nonlocal_angular_momentum,
      nonlocal_valence_configuration,
      nonlocal_augmentation_q_matrix=nonlocal_augmentation_q_matrix,
      nonlocal_augmentation_qij=nonlocal_augmentation_qij,
      nonlocal_augmentation_q_with_l=nonlocal_augmentation_q_with_l,
    )
