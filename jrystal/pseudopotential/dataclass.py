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
  r_cutoff: List[float]
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
    atomic_symbols = crystal.symbol
    num_atom = len(charges)

    pp_dict_list = []
    for symbol in crystal.symbol:
      pp_path = find_upf(dir, symbol)
      pp_dict = parse_upf(pp_path)
      pp_dict_list.append(pp_dict)

    valence_charges = []
    r_grid = []
    r_cutoff = []
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
      r_grid.append(np.array(pp["PP_MESH"]["PP_R"]))
      # r_cutoff.append(float(pp["PP_NONLOCAL"]["PP_BETA"]["cutoff_radius"][0]))
      r_cutoff.append(None)
      # norm conserving pseudopotential use the same cutoff_radius for local
      # and nonlocal potentials.

      local_potential_grid.append(np.array(pp["PP_LOCAL"]))
      local_potential_charge.append(valence_charges[-1] * 2)
      # The factor of 2 is due to the conversion from rydberg to hartree,
      # which is to make it consistent with QE.

      beta = []
      num_beta = len(pp["PP_NONLOCAL"]["PP_BETA"])
      nonlocal_num_beta.append(num_beta)
      beta_angular_momentum = []
      for beta_i in pp["PP_NONLOCAL"]["PP_BETA"]:
        beta.append(beta_i['values'] / r_grid[-1])
        beta_angular_momentum.append(int(beta_i["angular_momentum"]))

      nonlocal_beta_grid.append(np.stack(beta))
      nonlocal_beta_cutoff_radius.append(beta_i['cutoff_radius'])
      d_mat = np.array(pp["PP_NONLOCAL"]["PP_DIJ"])
      d_mat = np.reshape(d_mat, [num_beta, num_beta])
      nonlocal_d_matrix.append(d_mat)

      # 1/2 is due to the conversion from rydberg to hartree.
      nonlocal_angular_momentum.append(
        # [int(am) for am in beta_i["angular_momentum"]]
        beta_angular_momentum
      )
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
      r_cutoff,
      local_potential_grid,
      local_potential_charge,
      nonlocal_num_beta,
      nonlocal_beta_grid,
      nonlocal_beta_cutoff_radius,
      nonlocal_d_matrix,
      nonlocal_angular_momentum,
      nonlocal_valence_configuration
    )
