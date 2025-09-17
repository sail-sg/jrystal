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

from typing import List, Union
from dataclasses import dataclass
import numpy as np
from jaxtyping import Float, Array, Int

from .._src.crystal import Crystal
from .load import parse_upf, find_upf


@dataclass
class Pseudopotential():
  """
    Pseudopotential container format.
  """
  num_atom: int
  positions: Float[Array, "atom 3"]
  charges: Int[Array, "atom"]
  atomic_symbols: List[str]
  valence_charges: List[int]

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
      valence_charges (List[int]): Valence charges.
      r_grid (List[np.ndarray]): r grid.
      r_cutoff (List[float]): r cutoff.
      local_potential_grid (List[np.ndarray]): Local potential grid.
      local_potential_charge (List[int]): Local potential charge.
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
  l_max: int
  l_max_rho: int
  local_potential_grid: List[Float[Array, "num_r"]]
  local_potential_charge: List[int]
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

    pp_dict_list = []
    for symbol in crystal.symbols:
      pp_path = find_upf(dir, symbol)
      pp_dict = parse_upf(pp_path)
      pp_dict_list.append(pp_dict)

    valence_charges = []
    r_grid = []
    r_ab = []
    r_cutoff = []
    l_max = []
    l_max_rho = []
    local_potential_grid = []
    local_potential_charge = []
    nonlocal_num_beta = []
    nonlocal_beta_grid = []
    nonlocal_beta_cutoff_radius = []
    nonlocal_d_matrix = []
    nonlocal_angular_momentum = []
    nonlocal_valence_configuration = []

    for pp in pp_dict_list:
      valence_charges.append(int(float(pp["PP_HEADER"]["z_valence"])))
      _r_grid = np.array(pp["PP_MESH"]["PP_R"])
      r_grid.append(_r_grid[_r_grid > 0])
      r_ab.append(np.array(pp["PP_MESH"]["PP_RAB"])[_r_grid > 0])
      # r_cutoff.append(float(pp["PP_NONLOCAL"]["PP_BETA"]["cutoff_radius"][0]))
      r_cutoff.append(None)
      l_max.append(int(pp["PP_HEADER"]["l_max"]))
      if "l_max_rho" in pp["PP_HEADER"]:
        l_max_rho.append(int(pp["PP_HEADER"]["l_max_rho"]))
      else:
        l_max_rho.append(None)
      # norm conserving pseudopotential use the same cutoff_radius for local
      # and nonlocal potentials.

      local_potential_grid.append(np.array(pp["PP_LOCAL"])[_r_grid > 0]/2)
      # 1/2 is due to the conversion from rydberg to hartree.
      local_potential_charge.append(valence_charges[-1])

      beta = []

      if "PP_BETA" in pp["PP_NONLOCAL"]:
        num_beta = len(pp["PP_NONLOCAL"]["PP_BETA"])
        nonlocal_num_beta.append(num_beta)
        beta_angular_momentum = []
        for beta_i in pp["PP_NONLOCAL"]["PP_BETA"]:
          beta.append(
            np.divide(beta_i['values'], _r_grid, where=(_r_grid > 0))
          )  # the beta function is multiplied by r in upf file
          beta_angular_momentum.append(int(beta_i["angular_momentum"]))
        nonlocal_beta_grid.append(np.stack(beta)[:, _r_grid > 0])
        nonlocal_beta_cutoff_radius.append(beta_i['cutoff_radius'])
        d_mat = np.array(pp["PP_NONLOCAL"]["PP_DIJ"])
        d_mat = np.reshape(d_mat, [num_beta, num_beta])
        nonlocal_d_matrix.append(d_mat / 2)
        # 1/2 is due to the conversion from rydberg to hartree.

      else:
        nonlocal_num_beta.append(1)
        beta_angular_momentum = [0]
        beta.append(np.zeros_like(r_grid[-1]))

        nonlocal_beta_grid.append(np.stack(beta))
        nonlocal_beta_cutoff_radius.append(0)
        d_mat = np.zeros([1, 1])
        nonlocal_d_matrix.append(d_mat)

      nonlocal_angular_momentum.append(np.array(beta_angular_momentum))
      nonlocal_valence_configuration.append(
        pp["PP_INFO"]["Valence configuration"]
      )

    return NormConservingPseudopotential(
      num_atom,
      positions,
      charges,
      atomic_symbols,
      valence_charges,
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
      valence_charges (List[int]): Valence charges.
      r_grid (List[np.ndarray]): r grid.
      r_cutoff (List[float]): r cutoff.
      local_potential_grid (List[np.ndarray]): Local potential grid.
      local_potential_charge (List[int]): Local potential charge.
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

    ncpp = NormConservingPseudopotential.create(crystal, dir)

    pp_dict_list = []
    for symbol in crystal.symbols:
      pp_path = find_upf(dir, symbol)
      pp_dict = parse_upf(pp_path)
      pp_dict_list.append(pp_dict)

    nonlocal_augmentation_q_matrix = []
    nonlocal_augmentation_qij = []
    nonlocal_augmentation_q_with_l = []

    for pp in pp_dict_list:
      if "PP_AUGMENTATION" in pp["PP_NONLOCAL"]:
        q_matrix = np.array(pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_Q"])
        # q_matrix *= 2 * np.sqrt(np.pi)
        num_q = len(pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_Q"])
        num_q = np.sqrt(num_q).astype(int)
        q_matrix = q_matrix.reshape(num_q, num_q)
        assert np.linalg.eigvalsh(q_matrix).max() >= -1, (
          "The q_matrix is not negative semi-definite."
        )
        nonlocal_augmentation_q_matrix.append(q_matrix)

        num_r_grid = len(pp["PP_MESH"]["PP_R"])
        r_grid = np.array(pp["PP_MESH"]["PP_R"])
        nonlocal_augmentation_q_with_l.append(
          pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["q_with_l"]
        )

        if nonlocal_augmentation_q_with_l[-1] is True:

          q_ij_a = np.zeros(
            [num_q, num_q, int(pp["PP_HEADER"]["l_max_rho"]) + 1, num_r_grid]
          )
          for i in range(len(pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"])):
            m = int(
              pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"][i]["first_index"]
            ) - 1
            n = int(
              pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"][i]["second_index"]
            ) - 1
            angular_momentum = int(
              pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"][i]
              ["angular_momentum"]
            )
            _q_ij = np.array(
              pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"][i]["values"]
            )
            q_ij_a[m, n, angular_momentum] = np.array(
              np.divide(_q_ij, r_grid**2, where=(r_grid > 0))
            )

            if m != n:
              q_ij_a[n, m, angular_momentum] = q_ij_a[m, n, angular_momentum]

        else:
          q_ij_a = np.zeros([num_q, num_q, 1, num_r_grid])
          for i in range(len(pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"])):
            m = int(
              pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"][i]["first_index"]
            ) - 1
            n = int(
              pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"][i]["second_index"]
            ) - 1
            _q_ij = np.array(
              pp["PP_NONLOCAL"]["PP_AUGMENTATION"]["PP_QIJ"][i]["values"]
            )
            q_ij_a[m, n, 0] = np.array(
              np.divide(_q_ij, r_grid**2, where=(r_grid > 0))
            )

            if m != n:
              q_ij_a[n, m, 0] = q_ij_a[m, n, 0]

        nonlocal_augmentation_qij.append(q_ij_a)

    return UltrasoftPseudopotential(
      **ncpp.__dict__,
      nonlocal_augmentation_q_matrix=nonlocal_augmentation_q_matrix,
      nonlocal_augmentation_qij=nonlocal_augmentation_qij,
      nonlocal_augmentation_q_with_l=nonlocal_augmentation_q_with_l
    )
