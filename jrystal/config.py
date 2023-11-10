import ml_collections
from typing import Dict, Union, List, Tuple


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  # random seed
  config.seed: int = 123

  # Crystal geometry
  config.crystal = 'diamond'
  config.crystal_xyz_file = None

  # Planewave hyperparameters
  config.cutoff_energy: float = 100
  config.grid_sizes: Union[int, List[int], Tuple[int]] = 32
  config.k_grid_sizes: Union[int, List[int], Tuple[int]] = 1
  config.occupation: str = 'gamma'

  # functional hyperparameters.
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

  return config
