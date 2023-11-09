import ml_collections


def get_config() -> ml_collections.ConfigDict:
  config = ml_collections.ConfigDict()

  # random seed
  config.seed = 123

  # Crystal geometry
  config.crystal = 'diamond'
  config.crystal_xyz_file = None

  # Planewave hyperparameters
  config.ecut = 40
  config.grid_size = 24
  config.k_grid_size = 1

  # Optimizer hyperparamters
  config.optimizer = 'adam'
  config.optimizer_args = {'learning_rate': 1e-2}
  config.epoch = 5000
  config.convergence_condition = 1e-6
  config.ewald_args = {'ewald_eta': 0.1, 'ewald_cut': 5e4}

  return config
