import ml_collections
from typing import Dict, Union, List, Tuple


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  # Random seed
  config.seed: int = 123

  ################################################
  # Total Energy Minimization Hyper-parameters.  #
  ################################################

  # NOTE: The following configs are valid for all calculation.

  # Crystal geometry
  config.crystal = 'diamond'
  config.crystal_xyz_file = None

  # Planewave hyperparameters
  config.g_grid_mask_method: str = "cubic"
  # g_grid_mask_method can be either "cubic" or "spherical".
  # For "cubic", the cutoff cannot be assigned, but will be estimated.
  # Only for "spherical" method, the cutoff_energy argument is valid.
  config.cutoff_energy: float = 100
  config.grid_sizes: Union[int, List[int], Tuple[int]] = 32
  config.k_grid_sizes: Union[int, List[int], Tuple[int]] = 1
  config.occupation: str = 'gamma'
  config.smearing: float = 0.01  # only valid when occupation is fermi dirac

  # Xc functional hyperparameters.
  # config.xc: str = 'gga_x_pbe'
  config.xc: str = 'lda_x'

  # Optimizer hyperparamters
  config.optimizer: str = 'yogi'
  config.optimizer_args: Dict = {'learning_rate': 1e-2}
  config.epoch: int = 5000
  config.convergence_condition: float = 1e-6
  config.ewald_args: Dict = {'ewald_eta': 0.1, 'ewald_cut': 2e4}

  # Environment setting
  config.xla_preallocate: bool = False
  config.jax_enable_x64: bool = True
  config.verbose: bool = True

  ################################################
  # Band Structure Calculation Hyper-parameters. #
  ################################################

  # NOTE: The following configs are only valid for band structure calculation.

  # Need to either indicate the k_path or the k_path file.
  config.num_unoccupied_bands: int = 4
  config.k_path: str = "LGXL"
  config.num_kpoints: int = 60
  config.k_path_file: str = None
  config.band_structure_epoch: int = 5000

  # This flag indicates whether the fine tuning process will be adopted. If
  # true, then the module will optimize with respect to the first k-point,
  # and then use the params as the initialation of the next k-point.
  # Recommend to use this trick when the input crystal system is large.
  config.k_path_fine_tuning = True
  config.k_path_fine_tuning_epoch = 200

  ################################################
  # Parallelism Setting                          #
  ################################################
  config.enable_spmd = False

  return config
