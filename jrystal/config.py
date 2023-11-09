import ml_collections


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  # random seed
  config.seed = 1

  # Crystal geometry
  config.crystal = 'diamond'
  config.crystal_xyz_file = None

  # Planewave hyperparameters
  config.ecut = 100
  config.grid_size = 32
  config.k_grid_size = 1

  # Optimizer hyperparamters
  config.optimizer = 'yogi'
  config.optimizer_args = {'learning_rate': 1e-3}
  config.epoch = 5000
  config.convergence_condition = 1e-6
  config.ewald_args = {'ewald_eta': 0.1, 'ewald_cut': 5e4}

  return config
