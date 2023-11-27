import ml_collections
from typing import Dict, Union, List, Tuple


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  # Random seed
  config.seed: int = 123
  
  ################################################
  # Total Energy Minimization Hyper-parameters.  #
  ################################################
  
  # Crystal geometry
  config.crystal = 'diamond'
  config.crystal_xyz_file = None

  # Planewave hyperparameters
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
  config.epoch: int = 4000
  config.convergence_condition: float = 1e-6
  config.ewald_args: Dict = {'ewald_eta': 0.1, 'ewald_cut': 2e4}

  # Environment setting
  config.xla_preallocate: bool = False
  config.jax_enable_x64: bool = True

  ################################################
  # Band Structure Calculation Hyper-parameters. #
  ################################################

  # Need to either indicate the k_path or the k_path file.
  config.num_unoccupied_bands: int = 0
  config.k_path: str = "G"
  config.num_kpoints: int = 1
  config.k_path_file: str = None
  config.band_structure_epoch: int = 2000

  return config
